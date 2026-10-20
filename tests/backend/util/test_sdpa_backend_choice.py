"""`sdpa_backend_choice`: the dispatcher's own answer to "which kernel would run this exact call?".

These run against torch's real CPU chooser, whose one fused kernel is flash. That chooser ignores the
priority order, so the order's part in the answer is pinned on CUDA, in the Krea-2 grouped-query tests
and the working-memory probe tests.
"""

import pytest
import torch
from torch.nn.attention import SDPBackend

from invokeai.backend.util.attention import sdpa_backend_choice, sdpa_runs_fused
from invokeai.backend.util.sdpa_scope import sdpa_policy

HQ, HKV, SEQ, HEAD_DIM = 8, 2, 32, 64


def _qkv(kv_heads: int = HQ, device: str = "cpu", dtype: torch.dtype = torch.float32):
    q = torch.randn(1, HQ, SEQ, HEAD_DIM, device=device, dtype=dtype)
    kv = torch.randn(1, kv_heads, SEQ, HEAD_DIM, device=device, dtype=dtype)
    return q, kv, kv


def test_a_plain_call_gets_the_fused_kernel_under_the_default_selection():
    assert sdpa_backend_choice(*_qkv()) == SDPBackend.FLASH_ATTENTION
    assert sdpa_runs_fused(*_qkv()) is True


def test_a_window_in_force_is_part_of_the_answer_and_nothing_is_remembered():
    with sdpa_policy([SDPBackend.MATH]):
        assert sdpa_backend_choice(*_qkv()) == SDPBackend.MATH
        assert sdpa_runs_fused(*_qkv()) is False
    assert sdpa_backend_choice(*_qkv()) == SDPBackend.FLASH_ATTENTION


def test_enable_gqa_is_part_of_the_question():
    """Mismatched head counts reach a fused kernel only as grouped-query attention; asked without
    the flag, torch answers math for the same tensors."""
    q, k, v = _qkv(HKV)
    assert sdpa_backend_choice(q, k, v, enable_gqa=True) == SDPBackend.FLASH_ATTENTION
    assert sdpa_backend_choice(q, k, v, enable_gqa=False) == SDPBackend.MATH


def test_with_math_enabled_the_query_is_silent(capfd):
    """It runs on every Krea-2 attention call and inside every guarded ROCm one, so it must not print."""
    q, k, v = _qkv(HKV)
    sdpa_backend_choice(*_qkv())
    sdpa_backend_choice(q, k, v, enable_gqa=False)
    assert capfd.readouterr().err == ""


def test_with_math_disabled_a_call_no_kernel_takes_answers_none_without_torchs_noise(capfd):
    """Under an exclusive window torch's own query, finding nothing, re-runs every check in debug mode
    -- C++ warnings straight to stderr, which no Python filter reaches -- and raises. Asked once per
    attention call, that was thousands of lines a generation."""
    q, k, v = _qkv(HKV)
    with sdpa_policy([SDPBackend.FLASH_ATTENTION]):
        with pytest.raises(RuntimeError):
            torch.ops.aten._fused_sdp_choice(q, k, v, None, 0.0, False)
        assert capfd.readouterr().err != "", "pins why the helper must not ask torch directly here"

        assert sdpa_backend_choice(q, k, v) is None
        assert sdpa_runs_fused(q, k, v) is False
    assert capfd.readouterr().err == ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the per-kernel checks exist for CUDA tensors only")
class TestWithMathDisabledOnCuda:
    def test_the_first_permitted_kernel_that_takes_the_call_is_named(self):
        q, k, v = _qkv(device="cuda", dtype=torch.bfloat16)
        params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, False, False)
        if not torch.backends.cuda.can_use_efficient_attention(params):
            pytest.skip("this build has no memory-efficient kernel for the shape")
        with sdpa_policy([SDPBackend.EFFICIENT_ATTENTION]):
            assert sdpa_backend_choice(q, k, v) == SDPBackend.EFFICIENT_ATTENTION

    def test_grouped_query_attention_no_permitted_kernel_takes_is_none_and_silent(self, capfd):
        """The `INVOKE_KREA2_SDPA_BACKEND=efficient` case: memory-efficient refuses grouped-query attention."""
        q, k, v = _qkv(HKV, device="cuda", dtype=torch.bfloat16)
        with sdpa_policy([SDPBackend.EFFICIENT_ATTENTION]):
            assert sdpa_backend_choice(q, k, v, enable_gqa=True) is None
        assert capfd.readouterr().err == ""
