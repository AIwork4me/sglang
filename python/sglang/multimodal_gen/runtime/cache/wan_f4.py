"""Request-scoped, independent block-group caching for Wan ModelSlim F4."""

import functools
import math
from collections import Counter
from dataclasses import replace

import torch

from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp_utils import (
    default_mxfp4_policy,
    resolve_precision,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def validate_layer_groups(raw: list) -> list[dict]:
    from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
        CACHE_DIT_REQUEST_KNOB_KEYS,
    )

    if not isinstance(raw, list):
        raise ValueError("layer_groups must be a list of half-open layer ranges")
    groups = []
    previous_end = 0
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) - (
            CACHE_DIT_REQUEST_KNOB_KEYS | {"start", "end"}
        ):
            raise ValueError("Invalid layer_groups entry or cache knob")
        start, end = entry.get("start"), entry.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or start < previous_end
            or start >= end
        ):
            raise ValueError("layer_groups must contain ordered, nonoverlapping ranges")
        previous_end = end
        groups.append(dict(entry))
    return groups


def default_layer_groups(num_layers: int) -> list[dict]:
    return [
        {"start": start, "end": min(start + 8, num_layers - 2)}
        for start in range(5, num_layers - 2, 8)
    ]


def wan_f4_cache_mode(params, legacy_enabled: bool) -> str | None:
    if params.enable_cache_dit is False:
        return None
    if "quality" in getattr(params, "_explicit_fields", ()):
        return "grouped" if params.quality == "high" else None
    enabled = (
        legacy_enabled if params.enable_cache_dit is None else params.enable_cache_dit
    )
    if not enabled:
        return None
    raw = params.cache_dit_params or {}
    secondary = raw.get("secondary") or {}
    return (
        "grouped" if "layer_groups" in raw or "layer_groups" in secondary else "legacy"
    )


def _precision_signature(blocks, num_steps):
    result = []
    for block in blocks:
        description = getattr(block, "attention_quant_description", {})
        scheme = getattr(block, "attention_quant_scheme", None)
        if scheme in ("FP8", "MXFP4"):
            result.append(resolve_precision(description, "fa", scheme))
        for module in block.modules():
            linear = getattr(module, "scheme", None)
            if hasattr(linear, "is_dual_scale"):
                result.append(
                    resolve_precision(
                        linear.quant_config,
                        "w4a4_linear",
                        "W4A4",
                        None
                        if linear.is_dual_scale
                        else default_mxfp4_policy(num_steps),
                    )
                )
    return tuple(result)


class _GroupHost(torch.nn.Module):
    def __init__(self, transformers, configs, groups):
        super().__init__()
        self.segments = []
        self.cached_groups = []
        self.original_blocks = [transformer.blocks for transformer in transformers]
        self.precisions = [None] * len(transformers)
        self.force_steps = [None] * len(transformers)
        for expert, (blocks, config, ranges) in enumerate(
            zip(self.original_blocks, configs, groups)
        ):
            segments = []
            cursor = 0
            for index, entry in enumerate(ranges):
                start, end = entry["start"], entry["end"]
                if start < 5 or end > len(blocks) - 2:
                    raise ValueError(
                        "Wan F4 groups must preserve the first five and last two layers"
                    )
                if cursor < start:
                    segments.append((None, blocks[cursor:start]))
                name = f"expert_{expert}_group_{index}"
                group = torch.nn.ModuleList(blocks[start:end])
                group_config = replace(
                    config,
                    **{
                        key: value
                        for key, value in entry.items()
                        if key not in ("start", "end")
                    },
                )
                for key in (
                    "Fn_compute_blocks",
                    "Bn_compute_blocks",
                    "max_warmup_steps",
                    "max_continuous_cached_steps",
                ):
                    value = getattr(group_config, key)
                    minimum = (
                        1 if key in ("Fn_compute_blocks", "Bn_compute_blocks") else 0
                    )
                    if type(value) is not int or value < minimum:
                        raise ValueError(f"Invalid grouped cache {key}: {value}")
                threshold = group_config.residual_diff_threshold
                if (
                    isinstance(threshold, bool)
                    or not isinstance(threshold, (float, int))
                    or not math.isfinite(threshold)
                    or threshold <= 0
                ):
                    raise ValueError(
                        "Grouped cache threshold must be finite and positive"
                    )
                if group_config.steps_computation_policy != "dynamic":
                    raise ValueError("Wan F4 grouped cache requires dynamic SCM")
                if (
                    group_config.Fn_compute_blocks + group_config.Bn_compute_blocks
                    >= len(group)
                ):
                    segments.append((None, group))
                else:
                    self.add_module(name, group)
                    segments.append((name, group))
                    self.cached_groups.append((expert, name, start, end, group_config))
                cursor = end
            if cursor < len(blocks):
                segments.append((None, blocks[cursor:]))
            self.segments.append(segments)
        self.counts = {name: [0, 0] for _, name, *_ in self.cached_groups}

    def refresh(self, expert=None):
        import cache_dit

        names = self._context_names
        try:
            if expert is not None:
                self._context_names = [
                    context_name
                    for context_name, group in zip(names, self.cached_groups)
                    if group[0] == expert
                ]
            cache_dit.refresh_context(self)
        finally:
            self._context_names = names

    def forward(self, hidden_states, encoder_hidden_states, expert, *args, **kwargs):
        context = get_forward_context()
        batch = context.forward_batch
        total_steps = batch.num_inference_steps
        signature = _precision_signature(self.original_blocks[expert], total_steps)
        previous = self.precisions[expert]
        changed = previous is not None and previous != signature
        if changed:
            self.refresh(expert)
            self.force_steps[expert] = context.current_timestep
            logger.info(
                "Wan F4 expert=%d step=%d precision=%s; invalidating grouped residuals",
                expert,
                context.current_timestep,
                dict(Counter(signature)),
            )
        if previous is None:
            logger.info(
                "Wan F4 expert=%d step=%d precision=%s",
                expert,
                context.current_timestep,
                dict(Counter(signature)),
            )
        self.precisions[expert] = signature
        step = context.current_timestep
        force = step == self.force_steps[expert] or step < 2 or step >= total_steps - 2
        contexts = {
            group[1]: self._context_manager.get_context(context_name)
            for context_name, group in zip(self._context_names, self.cached_groups)
        }
        for name, original in self.segments[expert]:
            group_context = contexts.get(name)
            before = 0
            mask = None
            if group_context is not None:
                before = len(group_context.cached_steps) + len(
                    group_context.cfg_cached_steps
                )
                mask = group_context.cache_config.steps_computation_mask
                if force:
                    # Dynamic SCM must override residual similarity at protected steps.
                    group_context.cache_config.steps_computation_mask = [
                        1
                    ] * total_steps
            try:
                for block in getattr(self, name) if name else original:
                    hidden_states = block(
                        hidden_states, encoder_hidden_states, *args, **kwargs
                    )
            finally:
                if group_context is not None:
                    group_context.cache_config.steps_computation_mask = mask
            if group_context is not None:
                after = len(group_context.cached_steps) + len(
                    group_context.cfg_cached_steps
                )
                self.counts[name][int(after > before)] += 1
        return hidden_states


class _ExpertBlocks(torch.nn.Module):
    def __init__(self, host, expert):
        super().__init__()
        self.host = host
        self.expert = expert

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        return self.host(
            hidden_states, encoder_hidden_states, self.expert, *args, **kwargs
        )


class WanF4GroupedCache:
    def __init__(
        self,
        transformers,
        configs,
        groups,
        *,
        has_separate_cfg,
        sp_group=None,
        tp_group=None,
    ):
        import cache_dit

        from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
            BlockAdapter,
            DBCacheConfig,
            DMDCalibratorConfig,
            ForwardPattern,
            ParamsModifier,
            TaylorSeerCalibratorConfig,
            _assert_calibrator_exclusive,
            _assert_dmd_supported,
            _build_parallelism_config,
            _mark_transformer_parallelized,
            _patch_cache_dit_similarity,
        )

        self.transformers = transformers
        self.original_forwards = [transformer.forward for transformer in transformers]
        self.host = _GroupHost(transformers, configs, groups)
        self.adapter = None
        if not self.host.cached_groups:
            return
        modifiers = []
        for expert, name, start, end, config in self.host.cached_groups:
            _assert_calibrator_exclusive(config, label=name)
            _assert_dmd_supported(config, label=name)
            calibrator = None
            if config.enable_taylorseer:
                calibrator = TaylorSeerCalibratorConfig(
                    taylorseer_order=config.taylorseer_order
                )
            elif config.enable_dmd:
                calibrator = DMDCalibratorConfig(
                    dmd_history=config.dmd_history,
                    dmd_rank=config.dmd_rank,
                    dmd_ridge=config.dmd_ridge,
                    dmd_svd_precision=config.dmd_svd_precision,
                )
            modifiers.append(
                ParamsModifier(
                    cache_config=DBCacheConfig(
                        num_inference_steps=config.num_inference_steps,
                        Fn_compute_blocks=config.Fn_compute_blocks,
                        Bn_compute_blocks=config.Bn_compute_blocks,
                        max_warmup_steps=config.max_warmup_steps,
                        residual_diff_threshold=config.residual_diff_threshold,
                        max_continuous_cached_steps=config.max_continuous_cached_steps,
                        steps_computation_mask=config.steps_computation_mask,
                        steps_computation_policy="dynamic",
                        enable_separate_cfg=has_separate_cfg,
                    ),
                    calibrator_config=calibrator,
                )
            )
            logger.info(
                "Wan F4 expert=%d layers=[%d,%d): Fn=%d Bn=%d W=%d R=%.3f MC=%d TaylorSeer=%s DMD=%s",
                expert,
                start,
                end,
                config.Fn_compute_blocks,
                config.Bn_compute_blocks,
                config.max_warmup_steps,
                config.residual_diff_threshold,
                config.max_continuous_cached_steps,
                config.enable_taylorseer,
                config.enable_dmd,
            )
        names = [group[1] for group in self.host.cached_groups]
        # One adapter host also supports unequal group counts in cache-dit 1.3.5.
        self.adapter = BlockAdapter(
            transformer=self.host,
            blocks=[[getattr(self.host, name) for name in names]],
            blocks_name=[names],
            forward_pattern=[[ForwardPattern.Pattern_2] * len(names)],
            params_modifiers=[modifiers],
            has_separate_cfg=has_separate_cfg,
            check_forward_pattern=True,
            check_num_outputs=False,
        )
        parallelism = _build_parallelism_config(sp_group, tp_group)
        if parallelism is not None:
            _patch_cache_dit_similarity()
        _mark_transformer_parallelized(self.host, parallelism, sp_group, tp_group)
        try:
            cache_dit.enable_cache(self.adapter, parallelism_config=None)
            if parallelism is not None:
                manager = self.host._context_manager
                manager._sglang_sp_group = sp_group
                manager._sglang_tp_group = tp_group
                if sp_group is not None and tp_group is not None:
                    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                        get_dit_group,
                    )

                    manager._sglang_tp_sp_group = get_dit_group().device_group
            for expert, (transformer, original_forward) in enumerate(
                zip(transformers, self.original_forwards)
            ):
                runner = torch.nn.ModuleList([_ExpertBlocks(self.host, expert)])

                def wrap(original_forward, transformer, runner):
                    @functools.wraps(original_forward)
                    def forward(*args, **kwargs):
                        original_blocks = transformer.blocks
                        transformer.blocks = runner
                        try:
                            return original_forward(*args, **kwargs)
                        finally:
                            transformer.blocks = original_blocks

                    return forward

                transformer.forward = wrap(original_forward, transformer, runner)
        except Exception:
            self.disable()
            raise

    def refresh(self):
        if self.adapter is not None:
            self.log_stats()
            self.host.refresh()
        self.host.precisions = [None] * len(self.transformers)
        self.host.force_steps = [None] * len(self.transformers)
        self.host.counts = {name: [0, 0] for name in self.host.counts}

    def log_stats(self):
        for expert, name, start, end, _ in self.host.cached_groups:
            compute, reuse = self.host.counts[name]
            logger.info(
                "Wan F4 expert=%d layers=[%d,%d): compute=%d reuse=%d",
                expert,
                start,
                end,
                compute,
                reuse,
            )

    def disable(self):
        import cache_dit

        for transformer, original in zip(self.transformers, self.original_forwards):
            transformer.forward = original
        self.log_stats()
        if self.adapter is not None:
            cache_dit.disable_cache(self.adapter)
            self.adapter = None
