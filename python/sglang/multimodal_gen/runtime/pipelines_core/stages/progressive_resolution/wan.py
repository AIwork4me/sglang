# SPDX-License-Identifier: Apache-2.0
"""
Wan video progressive-resolution denoising stage.

Extends ProgressiveDenoisingStage for the Wan T2V video model:
  - Latent format: [B, C, T, H, W] (already spatial — no pack/unpack required)
  - Upsample: spatial H×W dims only; T (temporal frames) is fixed across all stages
  - No RoPE / freqs_cis update needed (Wan T2V uses no spatial positional embeddings
    that depend on H or W in the context)

Power-law spectrum constants fitted on VChitect dataset (9050 videos), spatial
spectrum P(ω) = A * |ω|^(-β):
  A    = 219.484718
  β    = 2.422687
"""

from __future__ import annotations

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_T2V_A14B_Config
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CACHE_DIT_REQUEST_KNOB_KEYS,
    CacheDitConfig,
    cache_dit_overrides_key,
    get_scm_mask,
    resolve_cache_dit_request_overrides,
)
from sglang.multimodal_gen.runtime.cache.wan_f4 import (
    WanF4GroupedCache,
    default_layer_groups,
    wan_f4_cache_mode,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
    is_progressive_resolution_mode,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Power-law spectrum constants for WAN 2.1 VAE
# Fitted on VChitect (9050 videos): P(ω) = A * |ω|^(-β)
WAN_SPECTRUM_A: float = 219.484718
WAN_SPECTRUM_BETA: float = 2.422687


class WanProgressiveDenoisingStage(ProgressiveDenoisingStage):
    """Wan T2V–specific progressive denoising stage.

    Differences from the FLUX progressive stage:
    - Wan latent is [B, C, T, H, W] — no patchify pack/unpack needed.
    - Progressive upsample grows only the spatial H×W plane; T stays fixed.
    - Wan T2V has no spatial RoPE freqs_cis that depends on H/W, so
      _on_resolution_change is a no-op.
    - Initial noise must carry the temporal dimension T_lat.
    """

    def __init__(self, transformer, scheduler, pipeline=None, vae=None) -> None:
        super().__init__(
            transformer,
            scheduler,
            pipeline=pipeline,
            vae=vae,
            spectrum_A=WAN_SPECTRUM_A,
            spectrum_beta=WAN_SPECTRUM_BETA,
        )
        self._wan_f4_grouped_cache = None

    def _is_wan_f4(self) -> bool:
        if not isinstance(self.server_args.pipeline_config, Wan2_2_T2V_A14B_Config):
            return False
        return any(
            isinstance(value, str) and value.startswith("W4A4_MXFP4")
            for transformer in (self.transformer, self.transformer_2)
            if transformer is not None
            for block in transformer.blocks
            for value in getattr(block, "attention_quant_description", {}).values()
        )

    def _cache_dit_requested_for_batch(self, batch: Req) -> bool:
        if self._is_wan_f4():
            return (
                wan_f4_cache_mode(batch.sampling_params, envs.SGLANG_CACHE_DIT_ENABLED)
                is not None
            )
        return super()._cache_dit_requested_for_batch(batch)

    def _unmount_cache_dit(self) -> None:
        grouped = getattr(self, "_wan_f4_grouped_cache", None)
        if grouped is None:
            super()._unmount_cache_dit()
            return
        grouped.disable()
        self._wan_f4_grouped_cache = None
        self._cache_dit_enabled = False
        self._cache_dit_active_key = None

    def _maybe_enable_cache_dit(self, num_inference_steps, batch: Req) -> None:
        params = batch.sampling_params
        diagnostics = {
            key: value
            for key, value in (params.modelslim_precision or {}).items()
            if value != "checkpoint"
        }
        scoped = self._is_wan_f4()
        if diagnostics and not scoped:
            raise ValueError(
                "modelslim_precision diagnostics require Wan2.2 T2V A14B ModelSlim F4"
            )
        if not scoped:
            return super()._maybe_enable_cache_dit(num_inference_steps, batch)
        mode = wan_f4_cache_mode(params, envs.SGLANG_CACHE_DIT_ENABLED)
        if diagnostics:
            if mode is not None or params.enable_teacache or params.enable_spectrum:
                raise ValueError(
                    "modelslim_precision diagnostics require all feature caching disabled"
                )
            if diagnostics.get("linear") == "W4A8" and any(
                getattr(getattr(module, "scheme", None), "is_dual_scale", False)
                for transformer in (self.transformer, self.transformer_2)
                if transformer is not None
                for module in transformer.modules()
            ):
                raise ValueError(
                    "W4A8 diagnostics do not support dual-scale MXFP4 linears"
                )
        if mode != "grouped":
            if self._wan_f4_grouped_cache is not None:
                self._unmount_cache_dit()
            return super()._maybe_enable_cache_dit(num_inference_steps, batch)
        if params.enable_teacache or params.enable_spectrum:
            raise ValueError(
                "Wan F4 grouped Cache-DiT cannot be combined with TeaCache or Spectrum"
            )
        if (
            self.server_args.enable_breakable_cuda_graph
            or self.server_args.enable_torch_compile
        ):
            raise ValueError(
                "Wan F4 grouped Cache-DiT currently requires eager execution"
            )
        if getattr(batch, "progressive_mode", "fullres") != "fullres":
            raise ValueError(
                "Wan F4 grouped Cache-DiT requires full-resolution denoising"
            )
        if batch.is_warmup:
            if self._cache_dit_enabled:
                self._unmount_cache_dit()
            return
        overrides = resolve_cache_dit_request_overrides(params.cache_dit_params)
        self._cache_dit_request_overrides = overrides
        primary_steps, secondary_steps = self._cache_dit_step_counts(
            num_inference_steps
        )
        steps = (
            [primary_steps]
            if self.transformer_2 is None
            else [primary_steps, secondary_steps]
        )
        transformers = (
            [self.transformer]
            if self.transformer_2 is None
            else [self.transformer, self.transformer_2]
        )
        high = (
            "quality" in getattr(params, "_explicit_fields", ())
            and params.quality == "high"
        )
        if high:
            configs = []
            for expert, count in enumerate(steps):
                knobs = {
                    key: value
                    for key, value in overrides.items()
                    if key in CACHE_DIT_REQUEST_KNOB_KEYS
                }
                if expert:
                    knobs.update(
                        {
                            key: value
                            for key, value in (overrides.get("secondary") or {}).items()
                            if key in CACHE_DIT_REQUEST_KNOB_KEYS
                        }
                    )
                configs.append(
                    CacheDitConfig(
                        enabled=True,
                        num_inference_steps=count,
                        **dict(
                            {
                                "Fn_compute_blocks": 1,
                                "Bn_compute_blocks": 1,
                                "max_warmup_steps": 6,
                                "residual_diff_threshold": 0.05 if expert else 0.08,
                                "max_continuous_cached_steps": 1,
                                "enable_taylorseer": False,
                                "enable_dmd": False,
                            },
                            **knobs,
                        ),
                        steps_computation_mask=get_scm_mask(
                            preset=overrides.get("scm_preset", "none"),
                            num_inference_steps=count,
                            compute_bins=overrides.get("scm_compute_bins"),
                            cache_bins=overrides.get("scm_cache_bins"),
                        ),
                        steps_computation_policy=overrides.get("scm_policy", "dynamic"),
                    )
                )
        else:
            _, policy, primary_mask, secondary_mask = self._cache_dit_scm_masks(
                primary_steps, secondary_steps
            )
            configs = [
                self._build_cache_dit_config(
                    count, mask, policy, secondary=bool(expert)
                )
                for expert, (count, mask) in enumerate(
                    zip(steps, [primary_mask, secondary_mask])
                )
            ]
        groups = []
        for expert, transformer in enumerate(transformers):
            raw = (overrides.get("secondary") or {}) if expert else overrides
            groups.append(
                raw.get(
                    "layer_groups",
                    overrides.get(
                        "layer_groups", default_layer_groups(len(transformer.blocks))
                    ),
                )
            )
        has_separate_cfg = (
            batch.do_classifier_free_guidance
            and not self.server_args.enable_cfg_parallel
        )
        key = (
            "wan_f4_grouped",
            high,
            tuple(steps),
            cache_dit_overrides_key(overrides),
            has_separate_cfg,
        )
        if self._cache_dit_enabled and key != self._cache_dit_active_key:
            self._unmount_cache_dit()
        if self._wan_f4_grouped_cache is not None:
            self._wan_f4_grouped_cache.refresh()
            return
        from sglang.multimodal_gen.runtime.distributed import (
            get_sp_group,
            get_tp_group,
            get_world_size,
        )

        sp_group = tp_group = None
        if get_world_size() > 1:
            sp, tp = get_sp_group(), get_tp_group()
            sp_group = sp.device_group if sp.world_size > 1 else None
            tp_group = tp.device_group if tp.world_size > 1 else None
        self._wan_f4_grouped_cache = WanF4GroupedCache(
            transformers,
            configs,
            groups,
            has_separate_cfg=has_separate_cfg,
            sp_group=sp_group,
            tp_group=tp_group,
        )
        self._cache_dit_enabled = True
        self._cache_dit_active_key = key
        logger.info(
            "Wan F4 quality=%s precision=%s cache=grouped",
            params.quality,
            params.modelslim_precision or "checkpoint",
        )

    # ------------------------------------------------------------------
    # Latent scale factor  (WanVAEArchConfig uses spatial_compression_ratio)
    # ------------------------------------------------------------------

    def _latent_scale_factor(self, server_args: ServerArgs) -> int:
        arch = server_args.pipeline_config.vae_config.arch_config
        return getattr(arch, "vae_scale_factor", None) or getattr(
            arch, "spatial_compression_ratio", 8
        )

    # ------------------------------------------------------------------
    # Resolution alignment  (Wan patch embedding requires even spatial dims)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Snap batch.height / batch.width to even multiples before progressive loop.

        Wan's patch embedding is Conv3d(stride=(1,2,2)), so each progressive
        stage latent must have even H and W.  At L levels of downsampling the
        initial latent is H_lat//(2^L) × W_lat//(2^L); if either is odd the
        patchification fails with a tensor size mismatch.

        Fix: align batch.height/width down to the nearest multiple of
        vae_scale_factor * 2^L * 2  (= vae_scale * align_unit) so that every
        stage latent is guaranteed even.  For 480p L=1 this is a no-op (60 is
        already divisible by 4).  For 720p L=1: 90→88 latent rows (704 px).
        """
        mode = getattr(batch, "progressive_mode", "fullres") or "fullres"
        if not is_progressive_resolution_mode(mode):
            try:
                return super().forward(batch, server_args)
            finally:
                if self._wan_f4_grouped_cache is not None:
                    self._wan_f4_grouped_cache.log_stats()

        levels = int(getattr(batch, "progressive_levels", 1))
        arch = server_args.pipeline_config.vae_config.arch_config
        vae_scale = getattr(arch, "vae_scale_factor", None) or getattr(
            arch, "spatial_compression_ratio", 8
        )
        # Each stage halves the spatial dims; Wan needs even dims at every stage.
        # Required: H_lat divisible by 2^L * patch_spatial (= 2^L * 2).
        align_pixels = vae_scale * (2**levels) * 2
        h_aligned = max((batch.height // align_pixels) * align_pixels, align_pixels)
        w_aligned = max((batch.width // align_pixels) * align_pixels, align_pixels)

        if h_aligned != batch.height or w_aligned != batch.width:
            logger.info(
                "WanProgressiveDenoisingStage: aligning resolution %dx%d → %dx%d "
                "so all progressive stage latents have even spatial dims (patch=2, L=%d)",
                batch.height,
                batch.width,
                h_aligned,
                w_aligned,
                levels,
            )
            batch.height = h_aligned
            batch.width = w_aligned

        return super().forward(batch, server_args)

    # ------------------------------------------------------------------
    # Initial noise  (must include the temporal dim T_lat)
    # ------------------------------------------------------------------

    def _generate_initial_noise(
        self,
        batch: Req,
        server_args: ServerArgs,
        h_lat: int,
        w_lat: int,
        seed,
    ) -> torch.Tensor:
        """Generate low-res initial noise [1, C, T_lat, h_lat, w_lat].

        The base-class version generates 4-D [1, C, H, W] noise and uses
        in_channels // 4 for the channel count (FLUX patchify convention).
        Wan operates directly on the 5-D latent, so we override to:
          - Use z_dim (= 16) as the correct latent channel count.
          - Preserve T_lat from the original full-res latent in batch.latents,
            since progressive upsample only grows spatial H×W.
        """
        device = get_local_torch_device()
        C = server_args.pipeline_config.vae_config.arch_config.z_dim
        # batch.latents still holds the full-res latent from LatentPreparationStage
        # at this call site, so shape[2] gives the fixed T_lat.
        T_lat = batch.latents.shape[2]
        dtype = server_args.pipeline_config.get_latent_dtype(
            batch.prompt_embeds[0].dtype if batch.prompt_embeds else torch.bfloat16
        )
        noise = randn_tensor(
            (self._initial_noise_batch_size(batch), C, T_lat, h_lat, w_lat),
            generator=self._get_initial_noise_generator(batch, seed, device),
            device=device,
            dtype=dtype,
        )
        return noise
