"""Model stage declarations and required residual capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from sglang.srt.layers.layer_boundary.adapters.overlap import tbo_exit_rows
from sglang.srt.layers.layer_boundary.contracts import (
    ExitRows,
    ProducerReduction,
    StageKind,
)
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import ResidualReadout, ResidualUpdate
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_QUANT_READOUT,
    NORM_READOUT,
    PLAIN_ADD,
)


@dataclass(frozen=True)
class StageDeclaration:
    """Describe one computation's boundaries without binding kernels or norms.

    Fields:
        kind: Selects attention/mixer or FFN boundary adapters; does not run
            the corresponding computation.
        read: Consumer operation that derives compute input from the residual.
        update: Producer operation that writes compute output into the residual.
        sparse: Whether this FFN is a MoE; used to resolve input/output rows.
            Expert routing and all-to-all remain inside the MoE computation.
        terminal: Whether this stage ends the model's layer stack, as the
            stack records it. Prevents leaving work that requires a following
            layer; a finalize handoff may still reach the terminal norm when
            the fusion provider allows it.
        output_transform: Optional operation on the contribution before the
            residual update. An FFN's exit runs it under an explicit
            reduction-order contract; an attention's is run by the input of
            the stage that follows, once the attention's sum is complete.
        reduction: Whether the next stage's input always completes the sum
            (an attention's), or the exit decides (an FFN's or a mixer's).
        gathers_attn_tp_input: Whether attention gathers TP-sharded input itself.
        tp_group: Group partitioning attention heads. Full TP consumers require
            full token rows, including when TP shares the prefill CP group.
        dense_tp_size: Dense FFN compute width: None uses the configured width,
            1 means local compute, the attention TP size means compute on the
            attention's rows over attention TP, and the full TP size means TP
            compute.
        exit_rows: Required FFN output rows at the layer or branch exit.
        writes_at_handoff: Whether this FFN writes its output into the residual
            at its exit because the next stage is on another pipeline rank.
        output_complete: Whether the FFN's compute completes its own output
            sum, so the exit owes none. For an FFN whose output sum is fused
            with a reduction its computation needs anyway.
        attn_tp_gather: Optional implementation of the gather over attention
            TP that brings rows into this stage: at its own entry, or at its
            producer's exit, which runs its consumer's gather. Given this rank's
            contiguous slice of the rows, it returns them all, in rank order,
            or None to leave the gather to the boundary. Called on every batch,
            so it must be CUDA-graph safe. Across a pipeline boundary the
            producer's rank takes it from the model's shared declaration
            function, without the declaring layer, so it may use only the
            communication state of the rank it runs on, not that layer's
            weights or buffers.
        previous: The stage whose output this stage consumes, as the stack
            records it: only the facts binding reads of it (see facts_of),
            on this rank or another.
        prepared_from: Declaration whose already-read input a branch reuses.
            Mutually exclusive with previous; avoids a second update/read.

    Sources contain declarations, never executable boundary objects.
    """

    kind: StageKind
    read: ResidualReadout
    update: ResidualUpdate
    sparse: bool = False
    terminal: bool = False
    output_transform: Optional[OutputTransform] = None
    reduction: ProducerReduction = ProducerReduction.EXIT_SCOPED
    gathers_attn_tp_input: bool = False
    tp_group: SumGroup = SumGroup.ATTN_TP
    dense_tp_size: Optional[int] = None
    exit_rows: Optional[ExitRows] = None
    writes_at_handoff: bool = False
    output_complete: bool = False
    attn_tp_gather: Optional[Callable] = None
    # Only declarations participate in construction, never executable stages.
    previous: Optional[StageDeclaration] = None
    prepared_from: Optional[StageDeclaration] = None

    def __post_init__(self):
        if self.previous is not None and self.prepared_from is not None:
            raise ValueError("choose a previous output or a prepared branch input")
        for source in (self.previous, self.prepared_from):
            if source is not None and not isinstance(source, StageDeclaration):
                raise TypeError("stage sources must be declarations")
        _check_update(self.update)
        _check_read(self.read)


# The members every ResidualUpdate and ResidualReadout must set: the binding
# reads each of them, and none has a default that is correct for every
# implementation.
_UPDATE_FACTS = (
    "is_plain_add",
    "applied_at_exit",
    "outlives_layer",
    "writes_stream",
    "quantized_sum",
)


_READ_FACTS = ("is_plain_norm", "reads_before_dp_gather", "reads_after_attn_tp_gather")


# The kernels a read supplies, which the binding also reads; empty when it
# supplies none.
_READ_KERNELS = ("completing_fusions", "gathering_reads")


def _require(protocol, implementation, facts):
    missing = [name for name in facts if not hasattr(implementation, name)]
    if missing:
        raise TypeError(
            f"{type(implementation).__name__} does not set {', '.join(missing)}, "
            f"which every {protocol} must"
        )


def _check_update(update):
    _require("ResidualUpdate", update, _UPDATE_FACTS)
    name = type(update).__name__
    if update.writes_stream and (update.is_plain_add or not update.applied_at_exit):
        raise ValueError(
            f"{name} writes the next stream itself, which only an update "
            "applied at its exit and other than a plain add does"
        )
    if update.quantized_sum and not update.is_plain_add:
        raise ValueError(
            f"{name} lets its sum run quantized, which only a plain add does"
        )


def _check_read(read):
    _require("ResidualReadout", read, _READ_FACTS + _READ_KERNELS)


def declare_attn(
    *,
    read=NORM_QUANT_READOUT,
    update=PLAIN_ADD,
    reduction=ProducerReduction.ALWAYS_PARTIAL,
    gathers_attn_tp_input=True,
    output_transform=None,
    tp_group=SumGroup.ATTN_TP,
    attn_tp_gather=None,
):
    """Declare attention or a mixer; construct its executable boundary later.

    Args:
        read: Input operation; defaults to normalization with quantization support.
        update: Operation that adds this stage's output to the residual.
        reduction: ALWAYS_PARTIAL for an attention whose sum the next stage's
            input always completes; EXIT_SCOPED for a mixer whose exit decides.
        gathers_attn_tp_input: Whether compute gathers attention-TP input slices itself.
        output_transform: Optional operation on the output once its sum is
            complete, before the residual update (a sandwich norm). The next
            stage's input runs it, so no fused add + norm takes that input.
            Requires ALWAYS_PARTIAL.
        tp_group: Head partition; ATTN_TP by default, TP for full-TP consumers.
        attn_tp_gather: Implementation of the gather over attention TP into
            this stage, tried before the boundary's own (see StageDeclaration).

    Returns:
        A StageDeclaration with no norm, tensors or execution plan.
    """
    if (
        output_transform is not None
        and reduction is not ProducerReduction.ALWAYS_PARTIAL
    ):
        raise ValueError("an attention output transform requires ALWAYS_PARTIAL")
    return StageDeclaration(
        StageKind.ATTENTION,
        read,
        update,
        output_transform=output_transform,
        reduction=reduction,
        gathers_attn_tp_input=gathers_attn_tp_input,
        tp_group=tp_group,
        attn_tp_gather=attn_tp_gather,
    )


def declare_ffn(
    *,
    sparse=False,
    read=NORM_READOUT,
    update=PLAIN_ADD,
    output_transform=None,
    next_layer_sparse=False,
    dense_tp_size=None,
    exit_rows=None,
    output_complete=False,
):
    """Declare a dense or MoE FFN independently of its compute module.

    Args:
        sparse: Whether the FFN is a MoE; EP dispatch stays inside compute.
        read: Operation deriving FFN input from the residual.
        update: Operation writing FFN output into the residual.
        output_transform: Optional contribution transform before residual update.
        next_layer_sparse: Whether the next decoder layer's FFN is sparse; used only
            to derive the TBO exit rows when exit_rows is not supplied.
        dense_tp_size: Dense compute width: None for configuration, 1 for local
            compute, the attention TP size, or the full TP size.
        exit_rows: Explicit output-row requirement; otherwise derived from
            the adjacent FFN kinds and TBO configuration.
        output_complete: Whether the compute completes its own output sum,
            fused with a reduction it needs anyway; the exit then owes none.

    Returns:
        A StageDeclaration with no norm, tensors or execution plan.
    """
    return StageDeclaration(
        StageKind.FFN,
        read,
        update,
        sparse=sparse,
        output_transform=output_transform,
        dense_tp_size=dense_tp_size,
        exit_rows=exit_rows or tbo_exit_rows(sparse, next_layer_sparse),
        output_complete=output_complete,
    )
