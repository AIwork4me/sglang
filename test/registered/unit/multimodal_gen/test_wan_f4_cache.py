"""Wan F4 cache lifecycle and precision transitions using real Cache-DiT."""

import argparse
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_T2V_A14B_Config
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CacheDitConfig,
    _patch_cache_dit_similarity,
    resolve_cache_dit_request_overrides,
)
from sglang.multimodal_gen.runtime.cache.wan_f4 import (
    WanF4GroupedCache,
    default_layer_groups,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp_utils import (
    default_mxfp4_policy,
    resolve_precision,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.wan import (
    WanProgressiveDenoisingStage,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _Block(torch.nn.Module):
    def __init__(self, index, *, dynamic=False, precision_sensitive=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.01))
        self.index = index
        self.calls = []
        self.dynamic = dynamic
        self.precision_sensitive = precision_sensitive
        self.attention_quant_description = {"blocks.5.ffn.weight": "W4A4_MXFP4"}
        self.attention_quant_scheme = None
        self.scheme = SimpleNamespace(
            is_dual_scale=False, quant_config=self.attention_quant_description
        )

    def forward(self, hidden_states, encoder_hidden_states):
        context = get_forward_context()
        self.calls.append(context.current_timestep)
        increment = self.weight
        if self.dynamic:
            increment = increment * (1 + context.current_timestep * 100)
        if self.precision_sensitive:
            precision = resolve_precision(
                self.scheme.quant_config,
                "w4a4_linear",
                "W4A4",
                default_mxfp4_policy(context.forward_batch.num_inference_steps),
            )
            if precision == "W4A8":
                increment = increment * 100
        return hidden_states + increment + encoder_hidden_states * 0


class _Transformer(torch.nn.Module):
    def __init__(self, *, dynamic=None, precision_sensitive=None):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                _Block(
                    index,
                    dynamic=index == dynamic,
                    precision_sensitive=index == precision_sensitive,
                )
                for index in range(40)
            ]
        )

    def forward(self, hidden_states, encoder_hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)
        return hidden_states


def _batch(
    quality="high",
    *,
    explicit=True,
    enable=None,
    overrides=None,
    precision=None,
    cfg=False,
    steps=20,
):
    params = SamplingParams(
        quality=quality,
        enable_cache_dit=enable,
        cache_dit_params=overrides,
        modelslim_precision=precision,
        num_inference_steps=steps,
    )
    params._explicit_fields = {"quality"} if explicit else set()
    return SimpleNamespace(
        sampling_params=params,
        num_inference_steps=steps,
        is_warmup=False,
        do_classifier_free_guidance=cfg,
        progressive_mode="fullres",
    )


def _run(transformer, batch, step, value=1.0):
    with set_forward_context(step, None, batch):
        return transformer(torch.full((1, 2, 2), value), torch.zeros(1, 2, 2))


def _stage():
    stage = WanProgressiveDenoisingStage.__new__(WanProgressiveDenoisingStage)
    stage.server_args = SimpleNamespace(
        pipeline_config=object.__new__(Wan2_2_T2V_A14B_Config),
        enable_breakable_cuda_graph=False,
        enable_torch_compile=False,
        enable_cfg_parallel=False,
    )
    stage.transformer = _Transformer()
    stage.transformer_2 = _Transformer()
    stage._wan_f4_grouped_cache = None
    stage._cache_dit_enabled = False
    stage._cache_dit_active_key = None
    stage._cache_dit_request_overrides = {}
    return stage


def _distributed_groups_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    controller = None
    try:
        _patch_cache_dit_similarity()
        transformer = _Transformer(dynamic=5 if rank else None)
        config = CacheDitConfig(
            num_inference_steps=20,
            Fn_compute_blocks=1,
            Bn_compute_blocks=1,
            max_warmup_steps=2,
            residual_diff_threshold=0.1,
            max_continuous_cached_steps=1,
        )
        controller = WanF4GroupedCache(
            [transformer],
            [config],
            [[{"start": 5, "end": 13}, {"start": 13, "end": 21}]],
            has_separate_cfg=False,
        )
        controller.host._context_manager._sglang_sp_group = dist.group.WORLD
        for step in range(8):
            _run(transformer, _batch(), step)
        # Rank 0's first group is locally reusable, but rank 1 vetoes it.
        assert len(transformer.blocks[6].calls) == 8
        assert len(transformer.blocks[14].calls) < 8
        counters = torch.tensor(
            [len(transformer.blocks[6].calls), len(transformer.blocks[14].calls)]
        )
        peers = [torch.empty_like(counters) for _ in range(2)]
        dist.all_gather(peers, counters)
        torch.testing.assert_close(peers[0], peers[1])
    finally:
        if controller is not None:
            controller.disable()
        dist.destroy_process_group()


class TestWanF4Cache(CustomTestCase):
    def test_quality_transition_removes_real_hooks_and_ignores_legacy_tuning(self):
        stage = _stage()
        self.addCleanup(stage._unmount_cache_dit)
        with (
            patch.dict(
                "os.environ",
                {
                    "SGLANG_CACHE_DIT_ENABLED": "true",
                    "SGLANG_CACHE_DIT_RDT": "0.4",
                    "SGLANG_CACHE_DIT_MC": "4",
                    "SGLANG_CACHE_DIT_TAYLORSEER": "true",
                },
            ),
            patch(
                "sglang.multimodal_gen.runtime.distributed.get_world_size",
                return_value=1,
            ),
        ):
            originals = [stage.transformer.blocks, stage.transformer_2.blocks]
            keys = list(stage.transformer.state_dict())
            stage._maybe_enable_cache_dit((10, 10), _batch())
            controller = stage._wan_f4_grouped_cache
            thresholds = [
                group[-1].residual_diff_threshold
                for group in controller.host.cached_groups
            ]
            self.assertEqual(set(thresholds), {0.08, 0.05})
            self.assertTrue(
                all(
                    not group[-1].enable_taylorseer
                    for group in controller.host.cached_groups
                )
            )
            _run(stage.transformer, _batch(), 3)
            self.assertIs(stage.transformer.blocks, originals[0])
            self.assertEqual(list(stage.transformer.state_dict()), keys)
            for quality in ("exact", "lossless"):
                stage._maybe_enable_cache_dit((10, 10), _batch(quality))
                self.assertFalse(stage._cache_dit_enabled)
                self.assertIsNone(stage._wan_f4_grouped_cache)
                torch.testing.assert_close(
                    _run(stage.transformer, _batch(quality), 3),
                    torch.full((1, 2, 2), 1.4),
                )
                self.assertFalse(hasattr(controller.host, "_context_manager"))
            stage._maybe_enable_cache_dit((10, 10), _batch(enable=False))
            self.assertFalse(stage._cache_dit_enabled)
            self.assertTrue(
                stage._cache_dit_requested_for_batch(_batch(explicit=False))
            )

    def test_groups_and_experts_decide_independently_and_refresh_buffers(self):
        primary, secondary = _Transformer(dynamic=13), _Transformer()
        configs = [
            CacheDitConfig(
                enabled=True,
                num_inference_steps=20,
                Fn_compute_blocks=1,
                Bn_compute_blocks=1,
                max_warmup_steps=2,
                residual_diff_threshold=0.1,
                max_continuous_cached_steps=1,
            )
        ] * 2
        groups = [
            [{"start": 5, "end": 13}, {"start": 13, "end": 21}],
            [{"start": 5, "end": 13}],
        ]
        controller = WanF4GroupedCache(
            [primary, secondary], configs, groups, has_separate_cfg=False
        )
        self.addCleanup(controller.disable)
        batch = _batch()
        for step in range(6):
            _run(primary, batch, step)
        self.assertLess(len(primary.blocks[6].calls), len(primary.blocks[14].calls))
        self.assertEqual(len(primary.blocks[14].calls), 6)
        self.assertEqual(secondary.blocks[6].calls, [])
        for step in range(6):
            _run(secondary, batch, step, value=100)
        controller.refresh()
        self.assertTrue(
            all(
                not controller.host._context_manager.get_context(name).buffers
                for name in controller.host._context_names
            )
        )
        torch.testing.assert_close(
            _run(secondary, batch, 0, value=2), torch.full((1, 2, 2), 2.4)
        )

    def test_precision_transitions_and_final_steps_compute_both_cfg_branches(self):
        for steps in (20, 40, 50):
            with self.subTest(steps=steps):
                transformer = _Transformer(precision_sensitive=6)
                config = CacheDitConfig(
                    enabled=True,
                    num_inference_steps=steps,
                    Fn_compute_blocks=1,
                    Bn_compute_blocks=1,
                    max_warmup_steps=0,
                    residual_diff_threshold=100,
                    max_continuous_cached_steps=1,
                )
                controller = WanF4GroupedCache(
                    [transformer],
                    [config],
                    [[{"start": 5, "end": 13}]],
                    has_separate_cfg=True,
                )
                try:
                    batch = _batch(cfg=True, steps=steps)
                    for step in range(steps):
                        for _ in range(2):
                            output = _run(transformer, batch, step)
                            expected = 2.39 if step < 2 or step >= steps - 2 else 1.4
                            torch.testing.assert_close(
                                output, torch.full((1, 2, 2), expected)
                            )
                    for step in (0, 1, 2, steps - 2, steps - 1):
                        self.assertEqual(transformer.blocks[6].calls.count(step), 2)
                    self.assertLess(len(transformer.blocks[6].calls), steps * 2)
                finally:
                    controller.disable()

    def test_failed_mount_and_forward_restore_original_modules(self):
        transformer = _Transformer()
        blocks, forward = transformer.blocks, transformer.forward
        with self.assertRaises(ValueError):
            WanF4GroupedCache(
                [transformer],
                [CacheDitConfig()],
                [[{"start": 0, "end": 8}]],
                has_separate_cfg=False,
            )
        self.assertIs(transformer.blocks, blocks)
        self.assertEqual(transformer.forward, forward)
        import cache_dit

        real_enable = cache_dit.enable_cache

        def fail_after_mount(*args, **kwargs):
            real_enable(*args, **kwargs)
            raise RuntimeError("mount failed")

        with (
            patch.object(cache_dit, "enable_cache", fail_after_mount),
            self.assertRaisesRegex(RuntimeError, "mount failed"),
        ):
            WanF4GroupedCache(
                [transformer],
                [CacheDitConfig(num_inference_steps=20, Bn_compute_blocks=1)],
                [default_layer_groups(40)],
                has_separate_cfg=False,
            )
        self.assertEqual(transformer.forward, forward)
        controller = WanF4GroupedCache(
            [transformer],
            [CacheDitConfig(num_inference_steps=20, Bn_compute_blocks=1)],
            [default_layer_groups(40)],
            has_separate_cfg=False,
        )
        self.addCleanup(controller.disable)
        with (
            patch.object(
                blocks[0], "forward", side_effect=RuntimeError("block failed")
            ),
            self.assertRaisesRegex(RuntimeError, "block failed"),
        ):
            _run(transformer, _batch(), 0)
        self.assertIs(transformer.blocks, blocks)

    def test_diagnostic_validation_cli_and_request_identity(self):
        parser = argparse.ArgumentParser()
        SamplingParams.add_cli_args(parser)
        raw = parser.parse_args(["--modelslim-precision", '{"attention":"FLOAT"}'])
        self.assertEqual(raw.modelslim_precision, {"attention": "FLOAT"})
        stage = _stage()
        self.addCleanup(stage._unmount_cache_dit)
        from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        scheduler.server_args = stage.server_args
        with patch.object(
            stage.server_args.pipeline_config,
            "supports_sequential_dit_inference",
            return_value=False,
        ):
            reference = scheduler._sampling_param_signature_items(
                _batch("lossless", explicit=False)
            )
            strict = scheduler._sampling_param_signature_items(_batch("lossless"))
            diagnostic = scheduler._sampling_param_signature_items(
                _batch("lossless", precision={"attention": "FLOAT"})
            )
        self.assertNotEqual(reference, strict)
        self.assertNotEqual(strict, diagnostic)
        with self.assertRaisesRegex(ValueError, "caching disabled"):
            stage._maybe_enable_cache_dit(
                (10, 10), _batch(precision={"linear": "W4A8"})
            )
        batch = _batch("exact", precision={"linear": "W4A8", "attention": "FLOAT"})
        stage._maybe_enable_cache_dit((10, 10), batch)
        with set_forward_context(5, None, batch):
            self.assertEqual(resolve_precision({}, "w4a4_linear", "W4A4"), "W4A8")
            self.assertEqual(resolve_precision({}, "fa", "MXFP4"), "FLOAT")
        stage.transformer.blocks[6].scheme.is_dual_scale = True
        with self.assertRaisesRegex(ValueError, "dual-scale"):
            stage._maybe_enable_cache_dit((10, 10), batch)
        for raw in ({"unknown": "FLOAT"}, {"linear": "FLOAT"}):
            with self.assertRaises(ValueError):
                SamplingParams(modelslim_precision=raw)
        for raw in (
            [{"start": 5, "end": 13}, {"start": 12, "end": 20}],
            [{"start": 5, "end": 5}],
        ):
            with self.assertRaises(ValueError):
                resolve_cache_dit_request_overrides({"layer_groups": raw})

    def test_checkpoint_policy_overrides_dynamic_boundary_defaults(self):
        from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp4_scheme import (
            ModelSlimMXFP4Scheme,
        )

        scheme = object.__new__(ModelSlimMXFP4Scheme)
        scheme.is_dual_scale = False
        scheme.quant_config = {}
        # Kernels are the device boundary; exercise the real offline scheme dispatcher.
        scheme.single_level_kernel = SimpleNamespace(apply=lambda layer, x, bias: x * 4)
        scheme.w4a8_kernel = SimpleNamespace(apply=lambda layer, x, bias: x * 8)
        for count in (20, 40, 50):
            batch = _batch(steps=count)
            for step in (0, 1, 2, count - 3, count - 2, count - 1):
                with set_forward_context(step, None, batch):
                    output = scheme.apply_weights(SimpleNamespace(), torch.ones(2))
                expected = 8 if step < 2 or step >= count - 2 else 4
                torch.testing.assert_close(output, torch.full((2,), float(expected)))
        batch = _batch(steps=20)
        for step, expected in ((0, "W4A8"), (2, "W4A4"), (18, "W4A8")):
            with set_forward_context(step, None, batch):
                self.assertEqual(
                    resolve_precision(
                        {}, "w4a4_linear", "W4A4", default_mxfp4_policy(20)
                    ),
                    expected,
                )
                self.assertEqual(
                    resolve_precision(
                        {"timestep_policy": {"w4a4_linear": {"default": "W4A4"}}},
                        "w4a4_linear",
                        "W4A4",
                        default_mxfp4_policy(20),
                    ),
                    "W4A4",
                )

    def test_distributed_similarity_uses_the_sequence_parallel_group(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_groups_worker,
                args=(f"{directory}/rendezvous",),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
