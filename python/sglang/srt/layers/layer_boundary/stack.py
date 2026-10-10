"""Collect model stages, connect neighbours, and bind each stage."""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import fields, is_dataclass, replace
from typing import Optional

from sglang.srt.layers.layer_boundary.connections import (
    _connect_line,
    _ffn_on_rank_rows,
    _fork_input,
    _note_origin,
)
from sglang.srt.layers.layer_boundary.construction import _bind_stage
from sglang.srt.layers.layer_boundary.contracts import (
    ExitRows,
    ProducerReduction,
    StageKind,
)
from sglang.srt.layers.layer_boundary.declarations import StageDeclaration
from sglang.srt.layers.layer_boundary.facts import facts_of
from sglang.srt.layers.layer_boundary.stage import StageBoundary


class _Append:
    """One append_stages call: its validated stages, the boundaries it
    returned, which the stack fills in when it closes, and where it was made."""

    __slots__ = ("declarations", "bindings", "prepared_from", "boundaries", "origin")

    def __init__(self, declarations, bindings, prepared_from, boundaries, origin):
        self.declarations = declarations
        self.bindings = bindings
        self.prepared_from = prepared_from
        self.boundaries = boundaries
        self.origin = origin


class _LayerStack:
    """A layer stack under construction: what was appended, in order, and how
    to reach the layers other pipeline ranks hold on either side of it."""

    __slots__ = ("appends", "previous_layers", "next_layers", "final_read")

    def __init__(self, previous_layers=(), next_layers=(), final_read=None):
        self.appends = []
        self.previous_layers = previous_layers
        self.next_layers = next_layers
        self.final_read = final_read


# The stack being built; layer_stack saves and restores an outer one.
_stack: Optional[_LayerStack] = None


@contextlib.contextmanager
def layer_stack(*, previous_layers=(), next_layers=(), final_read=None):
    """Open a layer stack that append_stages extends in order.

    Every stage binds when the stack closes, once its producer and its
    consumer are both known: the stage appended before and after it. The last
    stage ends the model's layer stack unless a later layer declares a stage.

    Args:
        previous_layers: For the layers before this stack that another
            pipeline rank holds, nearest first, callables that return the
            stages the layer declares, in order, without building it (the
            model's shared declaration function at that layer's index; an
            empty sequence for a layer that declares none). Called only if
            this stack appended stages, after its own layers are built, until
            one of them declares a stage: its last stage is the producer of
            this stack's first. Only what binding reads of that stage is kept
            (see facts_of).
        next_layers: Likewise for the layers after this stack, whose first
            declared stage is the consumer of this stack's last.
        final_read: The model's final read of the stack's output (a
            ``FinalRead``), when it is not a plain final norm: the consumer of
            the last stage when no later layer declares one. Its
            ``attn_tp_gather`` gathers the rows that stage leaves on this
            rank's attention-TP slice, and with ``reads_attn_tp_slices`` it
            reads that slice instead.
    """
    global _stack
    outer = _stack
    stack = _LayerStack(previous_layers, next_layers, final_read)
    _stack = stack
    try:
        yield stack
        if stack.appends:
            _bind_stack(
                stack.appends,
                previous=_neighbour_stage(stack.previous_layers, last=True),
                following=_neighbour_stage(stack.next_layers, last=False),
                final_read=stack.final_read,
            )
    finally:
        _stack = outer


def check_declared_stages(stack, appended: int, expected, where: str) -> None:
    """Check that what a layer appended to ``stack`` since it held
    ``appended`` appends is what its shared declaration function says it
    declares, as far as binding reads it: another pipeline rank binds this
    layer's stages from that function alone."""
    declared = [
        facts_of(declaration)
        for append in stack.appends[appended:]
        if append.prepared_from is None
        for declaration in append.declarations
    ]
    expected = [facts_of(declaration) for declaration in expected]
    if declared != expected:
        raise ValueError(
            f"{where} declares stages its shared declaration function does "
            f"not: {_first_difference(declared, expected)}"
        )


def _first_difference(declared, expected) -> str:
    """Where the stages a layer declares first differ from those its shared
    declaration function gives, field by field."""
    if len(declared) != len(expected):
        return (
            f"it declares {[stage.kind.name for stage in declared]}, the "
            f"function {[stage.kind.name for stage in expected]}"
        )
    index, (stage, given) = next(
        (index, pair)
        for index, pair in enumerate(zip(declared, expected))
        if pair[0] != pair[1]
    )
    return f"stage {index} ({stage.kind.name}): " + "; ".join(
        f"{name} is {value!r}, the function says {given_value!r}"
        for name, value, given_value in _differing_fields(stage, given)
    )


def _differing_fields(value, given, prefix=""):
    """(dotted name, value, given value) for each leaf field that differs,
    descending into declarations and the facts they hold."""
    if is_dataclass(value):
        names = [field.name for field in fields(value)]
    else:
        names = getattr(type(value), "__struct_fields__", None)
    if names is None or type(value) is not type(given):
        return [(prefix.rstrip("."), value, given)]
    return [
        difference
        for name in names
        if getattr(value, name) != getattr(given, name)
        for difference in _differing_fields(
            getattr(value, name), getattr(given, name), f"{prefix}{name}."
        )
    ]


def _neighbour_stage(layers, *, last):
    """The stage a neighbouring layer declares next to this stack: the last
    one of the nearest layer before it, or the first of the nearest after."""
    for layer in layers:
        declared = tuple(layer())
        if declared:
            return facts_of(declared[-1] if last else declared[0])
    return None


def _require_stack() -> _LayerStack:
    if _stack is None:
        raise RuntimeError(
            "append_stages needs an open layer stack; build the layers inside "
            "make_layers or layer_stack"
        )
    return _stack


def _handed_off(declaration):
    """A stage whose output crosses to another pipeline rank: an FFN leaves it
    on the attention's rows, which is what the handoff carries and the next
    rank's first stage reads, even where it would otherwise stay on this
    rank's attention-TP slice of them. An FFN on its own rows also writes its
    output into the residual there, on every batch, so the handoff carries the
    written stream and the receiver knows it from this declaration."""
    if declaration is None or declaration.kind is not StageKind.FFN:
        return declaration
    return replace(
        declaration,
        exit_rows=ExitRows.ATTENTION,
        writes_at_handoff=_ffn_on_rank_rows(
            declaration.sparse, declaration.dense_tp_size
        ),
    )


def _return_before_trailing_attention(line):
    """When the stack ends on attentions that always leave their sum, the next
    rank or the final read takes the residual on the rows they ran on, which
    is where the FFN before them returned it: no exit moves it after them.
    That FFN returns it on the attention's rows, as one that ends the stack or
    hands off does, instead of on its own rows."""
    trailing = False
    for append in reversed(line):
        declarations = append.declarations
        for index in reversed(range(len(declarations))):
            declaration = declarations[index]
            if (
                declaration.kind is StageKind.ATTENTION
                and declaration.reduction is ProducerReduction.ALWAYS_PARTIAL
            ):
                trailing = True
                continue
            if (
                trailing
                and declaration.kind is StageKind.FFN
                and _ffn_on_rank_rows(declaration.sparse, declaration.dense_tp_size)
            ):
                declarations[index] = replace(declaration, exit_rows=ExitRows.ATTENTION)
            return


def _bind_stack(appends, *, previous, following, final_read=None):
    """Bind every appended stage and fill in the boundaries each append
    returned.

    The stages of the appends that extend the stack form one line, chained in
    append order, and every connection between two of them is built once:
    each stage's outgoing connection is the next one's incoming. A branch
    then starts from the incoming connection of the stage it branches from.
    """
    line = [a for a in appends if a.prepared_from is None]
    if following is not None:
        # The last stage hands off to the next rank.
        line[-1].declarations[-1] = _handed_off(line[-1].declarations[-1])
    _return_before_trailing_attention(line)
    if following is None:
        # Without a stage on a later rank, the last one ends the model's stack,
        # read by the final read.
        line[-1].declarations[-1] = _ended(
            replace(line[-1].declarations[-1], terminal=True), final_read
        )
    stages = [
        (append, index) for append in line for index in range(len(append.bindings))
    ]
    bound = {id(append): [None] * len(append.bindings) for append in appends}
    # A returned declaration's stage as bound, for the branches that start from it.
    sources = {}
    producer = facts_of(_handed_off(previous))
    chained = []
    for append, index in stages:
        chained.append(
            replace(append.declarations[index], previous=producer, prepared_from=None)
        )
        producer = facts_of(chained[-1])
    connections = _connect_line(
        chained,
        [append.origin for append, _ in stages],
        before=chained[0].previous,
        after=following,
    )
    _bind_line(
        stages,
        chained,
        connections,
        bound,
        sources,
        final_read=final_read if following is None else None,
    )
    for append in appends:
        if append.prepared_from is None:
            continue
        source = sources.get(id(append.prepared_from))
        if source is None:
            raise ValueError(
                "prepared_from must be the declaration of a stage appended "
                f"earlier to the same layer stack (at {append.origin})"
            )
        source_declaration, source_incoming = source
        branch = [(append, index) for index in range(len(append.bindings))]
        chained = []
        for index, declaration in enumerate(append.declarations):
            chained.append(
                replace(
                    declaration,
                    previous=facts_of(chained[-1]) if chained else None,
                    prepared_from=None if chained else facts_of(source_declaration),
                )
            )
        try:
            arrivals = _fork_input(source_incoming)
        except Exception as error:
            _note_origin(error, append.origin)
            raise
        connections = _connect_line(
            chained,
            [append.origin] * len(chained),
            before=source_declaration,
            arrivals=arrivals,
        )
        _bind_line(branch, chained, connections, bound, sources)
    bound = [tuple(bound[id(append)]) for append in appends]
    _check_declared_gathers(appends, bound, remote_producer=previous is not None)
    for append, boundaries in zip(appends, bound):
        for returned, boundary in zip(append.boundaries, boundaries):
            returned.plan = boundary.plan
            returned.declaration = boundary.declaration


def _bind_line(stages, chained, connections, bound, sources, *, final_read=None):
    """Bind a chain of stages from its connections, ``connections[i]`` into
    ``chained[i]`` and ``connections[i + 1]`` out of it. An attention that
    always leaves its sum binds after the FFN that follows it in the same
    append, so the attention's entry keeps the residual that FFN's capture
    needs."""
    last = len(chained) - 1
    boundaries = [None] * len(chained)

    def bind(position):
        append, index = stages[position]
        norm, options = append.bindings[index]
        consumer = None
        if position < last and stages[position + 1][0] is append:
            consumer = chained[position + 1]
            if _carries_capture(chained[position], consumer):
                bind(position + 1)
        try:
            boundary = _bind_stage(
                chained[position],
                norm,
                connections[position],
                connections[position + 1],
                final_read=final_read if position == last else None,
                capture_preserves_residual=(
                    _preserves_residual(boundaries[position + 1])
                    if consumer is not None
                    and _carries_capture(chained[position], consumer)
                    else None
                ),
                **options,
            )
        except Exception as error:
            _note_origin(error, append.origin)
            raise
        boundaries[position] = boundary
        bound[id(append)][index] = boundary
        sources[id(append.boundaries[index].declaration)] = (
            boundary.declaration,
            connections[position],
        )

    for position in range(len(chained)):
        if boundaries[position] is None:
            bind(position)


def _carries_capture(producer, consumer):
    """Whether an attention that always leaves its sum keeps the residual its
    FFN's entry needs for capture. Only between stages of one append."""
    return (
        producer.kind is StageKind.ATTENTION
        and producer.reduction is ProducerReduction.ALWAYS_PARTIAL
        and consumer.kind is StageKind.FFN
    )


def _preserves_residual(consumer):
    """Each variant's check that the FFN's entry leaves the residual in place."""
    return {
        variant: path.entry.preserves_residual
        for variant, path in consumer.plan.paths.items()
        if path.entry.preserves_residual is not None
    }


def _check_declared_gathers(appends, bound, *, remote_producer):
    """A stage that declares its gather over attention TP must have one: at
    its own entry, or at its producer's exit. A producer on another pipeline
    rank runs it there."""
    line = [
        (append.origin, boundary)
        for append, boundaries in zip(appends, bound)
        if append.prepared_from is None
        for boundary in boundaries
    ]
    for (origin, consumer), producer in zip(line, [None, *(b for _, b in line)]):
        if consumer.declaration.attn_tp_gather is None or any(
            path.entry.input_gather_declared for path in consumer.plan.paths.values()
        ):
            continue
        if producer is None and remote_producer:
            continue
        if producer is not None and any(
            path.output_gathers_attn_tp for path in producer.plan.paths.values()
        ):
            continue
        error = ValueError(
            "an attention-TP gather is declared, but neither this stage's entry "
            "nor its producer's exit gathers over attention TP"
        )
        _note_origin(error, origin)
        raise error


def _ended(declaration, final_read):
    """The stack's last stage as the final read takes it: an FFN on this
    rank's attention-TP slice leaves its output there for a read that reads
    the slice and gathers what it read."""
    if declaration.kind is StageKind.FFN and getattr(
        final_read, "reads_attn_tp_slices", False
    ):
        return replace(declaration, exit_rows=ExitRows.SLICE)
    return declaration


_STAGE_OPTIONS = {
    StageKind.ATTENTION: frozenset({"qkv_latent_func", "fusions"}),
    StageKind.FFN: frozenset({"fusions"}),
}


def _stage_items(stages):
    declarations, bindings = [], []
    for item in stages:
        if len(item) not in (2, 3):
            raise ValueError("a stage needs (declaration, norm[, options])")
        declaration, norm = item[:2]
        if not isinstance(declaration, StageDeclaration):
            raise TypeError("append_stages requires stage declarations")
        if declaration.previous is not None or declaration.prepared_from is not None:
            raise ValueError("a declaration takes its source from the layer stack")
        if declaration.terminal:
            raise ValueError("the layer stack marks the terminal stage")
        options = dict(item[2] if len(item) == 3 else {})
        unexpected = options.keys() - _STAGE_OPTIONS[declaration.kind]
        if unexpected:
            raise TypeError(
                f"unsupported {declaration.kind.name} stage options: "
                f"{sorted(unexpected)}"
            )
        declarations.append(declaration)
        bindings.append((norm, options))
    return declarations, bindings


def _origin(stack: _LayerStack) -> str:
    """Where the current append_stages call was made, and its place in the stack."""
    frame = sys._getframe(2)
    return (
        f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno} "
        f"(append {len(stack.appends)} of its layer stack)"
    )


def append_stages(*stages, prepared_from=None):
    """Append a local linear sequence of stages to the open layer stack.

    Args:
        *stages: Items of (declaration, norm) or (declaration, norm, options).
            Declarations carry no source or terminal flag. Options are
            constructor keywords: fusions, and qkv_latent_func for attention.
        prepared_from: The ``declaration`` of a boundary an earlier call
            returned on this stack: the already-read stage this sequence
            branches from. A branch is a side path the model merges back
            explicitly: it neither extends the stack nor waits for a consumer,
            so each of its stages binds with only the stages of the branch.

    Returns:
        A tuple of StageBoundary objects in declaration order. Their
        declarations are usable at once and gain their place in the stack when
        it closes, which is also when their plans are bound.
    """
    if not stages:
        raise ValueError("append_stages needs at least one stage")
    stack = _require_stack()
    declarations, bindings = _stage_items(stages)
    declarations = [replace(d) for d in declarations]
    boundaries = tuple(StageBoundary(None, declaration=d) for d in declarations)
    stack.appends.append(
        _Append(declarations, bindings, prepared_from, boundaries, _origin(stack))
    )
    return boundaries
