# SPDX-License-Identifier: Apache-2.0
"""HIP host import guards must defer failure to call time, not import time.

sgl_kernel publishes CUDA-only wheels (no ROCm build; the PyPI manylinux
wheel requires ``libnvrtc.so.13``) and aiter ships only for Instinct CDNA
targets (gfx94x/gfx95x). On other HIP devices — e.g. Ryzen AI APU gfx1151
with a ``--no-deps`` ROCm install — the previous unconditional imports in
``fused_moe.py`` and the quark schemes made ``import sglang`` itself fail
before any model loaded.

These tests simulate exactly that host (HIP + both packages unavailable)
regardless of the machine they run on, and assert the two required
properties of the guards:

1. importing the affected modules still succeeds, and
2. the unavailable operators are bound to ``None`` so that *using* them
   fails loudly (``TypeError``) instead of silently producing wrong
   results.

With the wheels present (CUDA CI, CDNA aiter hosts) the guards are
no-ops: the ``try`` branch binds the real functions exactly as before.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

FUSED_MOE = "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
QUARK_MXFP4 = "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4"


@pytest.fixture
def simulated_hip_without_cuda_wheels(monkeypatch):
    """Make both target modules reload as if HIP without sgl_kernel/aiter."""
    monkeypatch.setattr("sglang.srt.utils.is_hip", lambda: True, raising=False)
    # ``None`` entries make ``from sgl_kernel import ...``/``import aiter``
    # raise ImportError on any interpreter, installed or not.
    monkeypatch.setitem(sys.modules, "sgl_kernel", None)
    monkeypatch.setitem(sys.modules, "aiter", None)
    yield
    # monkeypatch restores is_hip and sys.modules; re-import cleanly so the
    # real bindings (or the real absence) apply to any later test.
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm
    import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 as qs

    importlib.reload(fm)
    importlib.reload(qs)


def test_fused_moe_imports_on_hip_without_sgl_kernel(simulated_hip_without_cuda_wheels):
    fused_moe = importlib.import_module(FUSED_MOE)
    importlib.reload(fused_moe)
    assert fused_moe.gelu_and_mul is None
    assert fused_moe.gelu_tanh_and_mul is None
    assert fused_moe.silu_and_mul is None


def test_quark_mxfp4_imports_on_hip_without_aiter(simulated_hip_without_cuda_wheels):
    quark = importlib.import_module(QUARK_MXFP4)
    importlib.reload(quark)
    assert quark._gemm_a4w4_orig is None
    assert quark.shuffle_weight is None
    assert quark._dynamic_mxfp4_quant_orig is None


def test_unavailable_ops_fail_explicitly(simulated_hip_without_cuda_wheels):
    fused_moe = importlib.import_module(FUSED_MOE)
    importlib.reload(fused_moe)
    with pytest.raises(TypeError):
        fused_moe.silu_and_mul(None, None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
