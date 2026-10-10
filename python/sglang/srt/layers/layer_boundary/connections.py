"""Resolve token rows and contracts between adjacent stage declarations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional

from sglang.srt.environ import envs
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.layer_boundary.adapters.overlap import resolve_exit_rows
from sglang.srt.layers.layer_boundary.boundary import _cp_moves
from sglang.srt.layers.layer_boundary.contracts import (
    BatchVariant,
    EdgeContract,
    ExitRows,
    InputContract,
    OutputContract,
    ProducerReduction,
    StageContract,
    StageKind,
)
from sglang.srt.layers.layer_boundary.declarations import StageDeclaration
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    _cp_gathers_over_attn_cp,
    _prefill_cp_shards_tokens,
    batches_are_unpadded,
    input_scattered_configured,
    is_dense_ffn_fully_dp,
    moe_gathers_over_moe_cp,
    token_axis_sizes,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import NORM_READOUT, PLAIN_ADD
from sglang.srt.layers.moe import is_moe_input_scattered_across_dp_ranks
from sglang.srt.runtime_context import get_exec, get_parallel

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


def _reject_unsupported_cp_moe(moe_on_local_rows: bool, cp_shards: bool) -> None:
    """A MoE layer under attention CP whose tokens the steps cannot bring
    to it: under a prefill CP, one dispatched per DP shard under attention DP
    and GQA CP, and one on the TP group whose data-parallel groups are the CP
    ranks under DSA or MLA CP; and under attention DP, one on the TP group
    whose data-parallel groups are the CP ranks."""
    parallel = get_parallel()
    gqa = not _cp_gathers_over_attn_cp()
    if moe_on_local_rows:
        if cp_shards and gqa and parallel.attn_dp_size > 1:
            raise NotImplementedError(
                "a MoE dispatched per DP shard under attention DP and GQA prefill CP"
            )
    elif parallel.moe_dp_size == parallel.attn_cp_size:
        if cp_shards and not gqa:
            raise NotImplementedError(
                "a MoE on the TP group with moe_dp_size == attn_cp_size under "
                "DSA or MLA prefill CP"
            )
        if parallel.attn_dp_size > 1:
            raise NotImplementedError(
                "a MoE on the TP group with moe_dp_size == attn_cp_size under "
                "attention DP and attention CP"
            )


def _unpadded_possible() -> bool:
    """Whether a batch whose rows do not divide over attention TP may reach an
    FFN that would run on this rank's attention-TP slice, so that FFN needs a
    variant that stays on the attention's rows."""
    return batches_are_unpadded() and (
        is_moe_input_scattered_across_dp_ranks() or is_dense_ffn_fully_dp()
    )


def _active_variants():
    yield BatchVariant.ORDINARY
    if _prefill_cp_shards_tokens():
        yield BatchVariant.CONTEXT_PARALLEL
    if input_scattered_configured():
        yield BatchVariant.INPUT_SCATTERED
    if layernorm_sp.layernorm_sp_enabled():
        yield BatchVariant.SEQUENCE_PARALLEL
    if _unpadded_possible():
        yield BatchVariant.UNPADDED


def _row_layouts(variant):
    axes = token_axis_sizes(cp_active=variant is BatchVariant.CONTEXT_PARALLEL)
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axes
    )
    local = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, TokenAxis.ATTN_TP, axis_sizes=axes
    )
    return axes, attention, local, Layout.sharded_over(axis_sizes=axes)


def _ffn_on_rank_rows(sparse, dense_tp_size) -> bool:
    """Whether an FFN computes on this rank's own rows (its attention-TP slice
    of them, or an unpadded batch's rows) rather than on the attention's."""
    if sparse:
        return is_moe_input_scattered_across_dp_ranks()
    if dense_tp_size is None:
        return is_dense_ffn_fully_dp()
    return dense_tp_size == 1


def _resolve_ffn(
    variant,
    *,
    sparse,
    terminal=False,
    output_transform=None,
    read=NORM_READOUT,
    update=PLAIN_ADD,
    dense_tp_size=None,
    output_complete=False,
):
    parallel = get_parallel()
    if dense_tp_size not in (None, 1, parallel.attn_tp_size, parallel.tp_size):
        raise ValueError(
            "dense FFN rows support local compute, the attention TP group or the "
            "full TP group"
        )
    strategy = get_exec().comm.boundary_reduction
    if strategy not in ("ar", "rs", "rsv", "rs+rsv"):
        raise ValueError(
            "boundary_reduction must be resolved before model construction"
        )
    can_move_output = output_transform is None or output_transform.before_reduce_scatter
    use_reduce_scatter = strategy in ("rs", "rs+rsv") and can_move_output
    use_reduce_scatterv = strategy in ("rsv", "rs+rsv") and can_move_output
    cp_shards = _prefill_cp_shards_tokens()
    axes, attention, local, full = _row_layouts(variant)
    on_rank_rows = _ffn_on_rank_rows(sparse, dense_tp_size)
    if variant is BatchVariant.UNPADDED and on_rank_rows:
        # Rows that do not divide over attention TP stay whole: the FFN runs on
        # the attention's rows (an a2a MoE dispatches them from every
        # attention-TP rank) and its output is complete, as a2a's combine or
        # local compute leaves it.
        return (
            StageContract(
                InputContract(attention, read=read),
                OutputContract(attention, update=update, transform=output_transform),
            ),
            attention,
            attention,
        )
    on_cp_shards = (
        sparse
        and parallel.attn_cp_size > 1
        and parallel.moe_dp_size == parallel.attn_cp_size
        and not _cp_gathers_over_attn_cp()
    )
    # Over attention TP, short of the full TP group: the FFN owes only the
    # attention-TP sum, and nothing moves over DP.
    on_attention_rows = (
        not sparse
        and dense_tp_size == parallel.attn_tp_size
        and dense_tp_size not in (1, parallel.tp_size)
    )
    if sparse:
        group = SumGroup.MOE_OUTPUT
    else:
        group = SumGroup.ATTN_TP if on_attention_rows else SumGroup.TP
    # An FFN that writes the next stream itself hands on a complete output.
    complete = on_rank_rows or update.writes_stream or output_complete
    if variant is BatchVariant.SEQUENCE_PARALLEL:
        return (
            StageContract(
                InputContract(
                    full,
                    gathered_by_compute=frozenset({TokenAxis.ATTN_TP}),
                    read=read,
                ),
                OutputContract(local, update=update, transform=output_transform),
            ),
            local,
            local,
        )
    if variant is BatchVariant.INPUT_SCATTERED:
        scattered_residual = update.applied_at_exit
        residual = local if scattered_residual else attention
        returned = local if scattered_residual and not terminal else attention
        return (
            StageContract(
                InputContract(full, read=read),
                OutputContract(
                    attention,
                    group=None if complete else group,
                    may_reduce_scatter=not complete
                    and use_reduce_scatter
                    and (scattered_residual or not terminal),
                    update=update,
                    transform=output_transform,
                ),
            ),
            residual,
            returned,
        )
    if cp_shards and _cp_moves().reduce_scatter is not None:
        may_leave = variant is not BatchVariant.CONTEXT_PARALLEL
        may_scatter = True
    elif cp_shards or parallel.attn_dp_size > 1:
        may_leave = may_scatter = parallel.attn_cp_size == 1 or on_cp_shards
    else:
        may_leave = may_scatter = True
    if on_rank_rows:
        rows = local
    elif on_attention_rows:
        rows = attention
        # The output is already on the residual's rows: nothing to scatter.
        use_reduce_scatter = use_reduce_scatterv = False
    else:
        rows = Layout.sharded_over(
            *((TokenAxis.ATTN_CP,) if on_cp_shards else ()), axis_sizes=axes
        )
    produced = (
        OutputContract(rows, update=update, transform=output_transform)
        if complete
        else OutputContract(
            rows,
            group=group,
            may_defer_to_next=may_leave
            and not terminal
            and not update.applied_at_exit
            and update.outlives_layer
            and output_transform is None,
            may_reduce_scatter=use_reduce_scatter and may_scatter,
            may_reduce_scatterv=use_reduce_scatterv and may_leave,
            update=update,
            transform=output_transform,
        )
    )
    returned = local if on_rank_rows and not terminal else attention
    return (
        StageContract(InputContract(rows, read=read), produced),
        local if on_rank_rows else attention,
        returned,
    )


@dataclass(frozen=True)
class StageConnection:
    """Immutable, norm-free contracts for the two sides of a connection.

    Fields:
        producer: Source declaration, or None at stack entry.
        consumer: Destination declaration, or None at a layer or stack exit.
        exits: Producer-side EdgeContract for each supported BatchVariant.
        entries: Consumer-side EdgeContract for each supported BatchVariant.

    Equivalent declarations can be reconstructed across pipeline partitions;
    adjacent layers need not share the same connection object.
    """

    producer: Optional[StageDeclaration]
    consumer: Optional[StageDeclaration]
    exits: Mapping[BatchVariant, EdgeContract]
    entries: Mapping[BatchVariant, EdgeContract]

    def __post_init__(self):
        object.__setattr__(self, "exits", MappingProxyType(dict(self.exits)))
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))


def _reject_unsupported(stage):
    """Reject a stage whose declared capabilities this parallel configuration
    cannot bind, naming the combination. The rejections that depend on the
    rows of one edge are made where that edge binds (``_bind_stage``)."""
    parallel = get_parallel()
    if stage.update.applied_at_exit:
        if stage.sparse and moe_gathers_over_moe_cp():
            raise NotImplementedError(
                "an update applied at the stage's exit with a MoE gathered over "
                "the MoE-CP group"
            )
        if parallel.attn_cp_size > 1 and input_scattered_configured():
            raise NotImplementedError(
                "an update applied at the stage's exit with input-scattered "
                "attention under attention CP"
            )
    if stage.kind is StageKind.FFN and stage.sparse and parallel.attn_cp_size > 1:
        _reject_unsupported_cp_moe(
            _ffn_on_rank_rows(stage.sparse, stage.dense_tp_size),
            _prefill_cp_shards_tokens(),
        )


class _Arrival(NamedTuple):
    """What reaches a stage's entry from its producer, for one batch variant.
    ``plain_add`` is None after an attention that always leaves its sum: the
    entry receives that update with the output."""

    produced: OutputContract
    residual: Layout
    plain_add: Optional[bool]
    written: bool


class _Flow(NamedTuple):
    """One stage's residual data flow for one batch variant."""

    contract: StageContract
    arrival: _Arrival
    during: Layout
    returned: Layout


def _stack_arrival(variant):
    """What reaches the first stage of the model's layer stack: its input, on
    the attention's rows (this rank's slice of them under sequence
    parallelism), owing the TP sum an input-scattered batch leaves."""
    _, attention, local, _ = _row_layouts(variant)
    rows = local if variant is BatchVariant.SEQUENCE_PARALLEL else attention
    owes = variant is BatchVariant.INPUT_SCATTERED
    return _Arrival(
        OutputContract(
            rows,
            group=SumGroup.TP if owes else None,
            always_partial=owes,
            update=None,
        ),
        rows,
        True,
        False,
    )


def _resolve_stage(stage, variant, arrival, following=None):
    """``stage``'s data flow for ``variant``, given what arrives at its entry
    and the stage after it (None when it ends the stack or a branch)."""
    axes, attention, local, full = _row_layouts(variant)
    _reject_unsupported(stage)
    if stage.kind is StageKind.FFN:
        declaration, residual, returned = _resolve_ffn(
            variant,
            sparse=stage.sparse,
            terminal=stage.terminal,
            output_transform=stage.output_transform,
            read=stage.read,
            update=stage.update,
            dense_tp_size=stage.dense_tp_size,
            output_complete=stage.output_complete,
        )
        exit_rows = resolve_exit_rows(stage.exit_rows)
        if exit_rows is ExitRows.ATTENTION or (
            following is not None and following.read.reads_after_attn_tp_gather
        ):
            returned = attention
        elif exit_rows is ExitRows.SLICE:
            # The residual's rows during the FFN: this rank's slice for an FFN
            # on its own rows, else the attention's.
            returned = residual
        return _Flow(declaration, arrival, residual, returned)
    sp = variant is BatchVariant.SEQUENCE_PARALLEL
    scattered = variant is BatchVariant.INPUT_SCATTERED
    gathers = (
        frozenset({TokenAxis.ATTN_TP})
        if stage.gathers_attn_tp_input and (sp or scattered or _use_ag_after_qlora)
        else frozenset()
    )
    owes = not sp and axes[TokenAxis.ATTN_TP] > 1
    group = SumGroup.ATTN_TP
    compute_rows = attention
    if stage.tp_group is SumGroup.TP:
        parallel = get_parallel()
        if parallel.attn_dp_size != 1 or sp or scattered:
            raise NotImplementedError(
                "full-TP attention requires unscattered input without attention DP"
            )
        compute_rows = full
        owes = parallel.tp_size > 1
        # Canonicalize equal groups to the CP transport's reduction contract.
        # initialize_model_parallel aliases ATTN_CP to TP at this topology.
        group = (
            SumGroup.ATTN_CP
            if parallel.attn_cp_size == parallel.tp_size and parallel.attn_cp_size > 1
            else SumGroup.TP
        )
    elif stage.tp_group is not SumGroup.ATTN_TP:
        raise ValueError("attention heads must be partitioned over ATTN_TP or TP")
    declaration = StageContract(
        InputContract(compute_rows, gathered_by_compute=gathers, read=stage.read),
        OutputContract(
            local if sp else compute_rows,
            group=group if owes else None,
            always_partial=owes
            and (
                stage.reduction is ProducerReduction.ALWAYS_PARTIAL
                or (following is not None and following.kind is StageKind.FFN)
            ),
            may_defer_to_next=owes
            and stage.reduction is ProducerReduction.EXIT_SCOPED
            and following is not None
            and following.kind is StageKind.ATTENTION,
            update=stage.update,
            transform=stage.output_transform,
        ),
    )
    # An attention moves no residual: it keeps the rows it arrives on, except
    # that one that always leaves its sum completes an input-scattered sum
    # it receives onto this rank's slice of them.
    during = arrival.residual
    if (
        variant is BatchVariant.INPUT_SCATTERED
        and arrival.produced.always_partial
        and arrival.produced.update is None
        and stage.reduction is ProducerReduction.ALWAYS_PARTIAL
    ):
        during = Layout(during.sharded | {TokenAxis.ATTN_TP})
    return _Flow(declaration, arrival, during, attention)


def _arrival(producer, flow, variant):
    """What reaches the stage after ``producer`` from it, given its flow."""
    output = flow.contract.output
    if (
        producer.kind is StageKind.ATTENTION
        and producer.reduction is ProducerReduction.ALWAYS_PARTIAL
    ):
        # The output owes its sum, and the residual is where it ran.
        return _Arrival(output, flow.during, None, False)
    if (
        producer.kind is StageKind.ATTENTION
        and producer.reduction is ProducerReduction.EXIT_SCOPED
        and (output.always_partial or output.may_defer_to_next)
    ):
        # A mixer's exit leaves the next entry the sum it declares; a sum it
        # may carry on arrives with the value, if at all.
        owes, group = output.always_partial, output.group
    else:
        # On an input-scattered batch, the next entry's reduce-scatter
        # completes the TP sum of an output not written at its exit.
        owes = (
            variant is BatchVariant.INPUT_SCATTERED
            and not producer.update.applied_at_exit
        )
        group = SumGroup.TP
    return _Arrival(
        OutputContract(
            flow.returned,
            group=group if owes else None,
            always_partial=owes,
            update=None,
        ),
        flow.returned,
        producer.update.is_plain_add,
        producer.update.applied_at_exit,
    )


def _entry_edge(stage, flow, variant):
    """The edge into ``stage`` as its entry binds it."""
    arrival = flow.arrival
    joins = (
        variant is BatchVariant.INPUT_SCATTERED
        and arrival.produced.update is not None
        and arrival.produced.update.is_plain_add
        and stage.kind is StageKind.FFN
    )
    return EdgeContract(
        arrival.produced,
        flow.contract.input,
        arrival.residual,
        flow.during,
        residual_joins_sum=joins,
        arriving_plain_add=arrival.plain_add,
        arrives_written=arrival.written,
    )


def _exit_edge(stage, flow, entry):
    """The edge out of ``stage`` as its exit binds it. An attention that always
    leaves its sum has no exit: its edge is its consumer's entry edge, when
    there is a consumer."""
    if entry is not None and (
        stage.kind is StageKind.ATTENTION
        and stage.reduction is ProducerReduction.ALWAYS_PARTIAL
    ):
        return entry
    return EdgeContract(
        flow.contract.output, InputContract(flow.returned), flow.during, flow.returned
    )


def _connect_line(line, origins, *, before=None, after=None, arrivals=None):
    """The connections along a chain of stages, each resolved once per batch
    variant: into ``line[0]``, between each two, and out of ``line[-1]`` to
    ``after`` (a stage on the next rank, or None at the end of the stack or a
    branch). What arrives at ``line[0]`` is ``arrivals``, else what ``before``
    (a stage on the previous rank, or None for the stack input) hands on; that
    rank's own entry is taken as the stack input's. An error resolving a stage
    names the append at ``origins[i]`` that declared ``line[i]``, or the one
    next to the neighbouring rank's stage."""
    variants = tuple(_active_variants())

    def resolve(position, stage, arrivals, following=None):
        try:
            return {
                v: _resolve_stage(stage, v, arrivals[v], following) for v in variants
            }
        except Exception as error:
            _note_origin(error, origins[position])
            raise

    if arrivals is None:
        arrivals = {v: _stack_arrival(v) for v in variants}
        if before is not None:
            if (
                before.output_transform is not None
                and before.kind is StageKind.ATTENTION
                and before.reduction is ProducerReduction.ALWAYS_PARTIAL
            ):
                # Its transform runs at the next stage's input, which would be
                # on this rank, without the module that declares it.
                error = NotImplementedError(
                    "a pipeline rank that ends on an attention transforming the "
                    "output whose sum it leaves"
                )
                _note_origin(error, origins[0])
                raise error
            flow = resolve(0, before, arrivals, line[0])
            arrivals = {v: _arrival(before, flow[v], v) for v in variants}
    flows = []
    for position, stage in enumerate(line):
        following = line[position + 1] if position + 1 < len(line) else after
        flow = resolve(position, stage, arrivals, following)
        flows.append(flow)
        arrivals = {v: _arrival(stage, flow[v], v) for v in variants}
    if after is not None:
        flows.append(resolve(len(line) - 1, after, arrivals))
    entries = [
        {v: _entry_edge(stage, flow[v], v) for v in variants}
        for stage, flow in zip([*line, after], flows)
    ]
    connections = [StageConnection(before, line[0], {}, entries[0])]
    for position, stage in enumerate(line):
        consumer = line[position + 1] if position + 1 < len(line) else after
        following = entries[position + 1] if consumer is not None else {}
        connections.append(
            StageConnection(
                stage,
                consumer,
                {
                    v: _exit_edge(stage, flows[position][v], following.get(v))
                    for v in variants
                },
                following,
            )
        )
    return connections


def _fork_input(prepared):
    """What reaches a branch's first stage: an already-read input, placed onto
    the branch's computation rows by the branch adapter, which forks the
    stream without running the branch's input norm again."""
    arrivals = {}
    for variant in _active_variants():
        if variant is not BatchVariant.ORDINARY:
            raise NotImplementedError(
                "prepared branch transport requires ordinary token rows"
            )
        source = prepared.entries[variant]
        arrivals[variant] = _Arrival(
            OutputContract(source.need.layout, update=None),
            source.residual_to,
            True,
            False,
        )
    return arrivals


def _note_origin(error: Exception, origin: str) -> None:
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(f"while binding the stages appended at {origin}")
