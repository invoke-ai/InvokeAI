"""The automatic fallback from a cuDNN attention call that fails at run time.

torch checks cuDNN's eligibility statically, so a call can still fail inside it; the fallback demotes
cuDNN behind math in the process order and retries. A real cuDNN failure cannot be produced on CPU, so
the kernel is a double that raises, and the dispatcher's choice is forced to cuDNN for the failing
case only. The priority order and the enable flags are torch's real ones; the orders expected of them
come from `sdpa_orders`, written out per layout of torch's own order.
"""

import logging
import threading

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

import invokeai.backend.util.attention as attention
from invokeai.backend.util.attention import install_cudnn_sdpa_fallback
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.sdpa_scope import install_default_sdpa_priority, sdpa_policy

LOGGER = InvokeAILogger.get_logger(attention.__name__).name
# torch's own function, whatever the tests rebind: the grouped-query retry is checked against it.
TORCH_SDPA = F.scaled_dot_product_attention


class _Kernel:
    """Stands in for torch's SDPA: raises the scripted errors in call order, then returns a tensor
    that says which call produced it."""

    def __init__(self, *errors: BaseException | None) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(
        self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False
    ):
        self.calls += 1
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error
        return torch.full_like(query, float(self.calls))


def _priority() -> list[SDPBackend]:
    return [SDPBackend(backend) for backend in torch._C._get_sdp_priority_order()]


def _dispatcher_names(monkeypatch, backend: SDPBackend) -> None:
    monkeypatch.setattr(torch.ops.aten, "_fused_sdp_choice", lambda *args, **kwargs: int(backend))


@pytest.fixture
def install(monkeypatch, preserve_sdpa_priority_order, sdpa_orders):
    """Install the fallback over `kernel` on a build that looks like CUDA, under the process order."""
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(attention, "_cudnn_demotions", 0)
    torch._C._set_sdp_priority_order([int(backend) for backend in sdpa_orders.torch_own])
    install_default_sdpa_priority()

    def _install(kernel: _Kernel) -> None:
        monkeypatch.setattr(F, "scaled_dot_product_attention", kernel)
        install_cudnn_sdpa_fallback()

    return _install


def _qkv() -> torch.Tensor:
    return torch.randn(1, 2, 8, 16)


def test_a_cudnn_failure_demotes_cudnn_warns_once_and_returns_the_retry(install, monkeypatch, caplog, sdpa_orders):
    kernel = _Kernel(RuntimeError("cuDNN Frontend error: no valid execution plan"))
    install(kernel)
    _dispatcher_names(monkeypatch, SDPBackend.CUDNN_ATTENTION)
    q = _qkv()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = F.scaled_dot_product_attention(q, q, q)
        F.scaled_dot_product_attention(q, q, q)

    assert torch.equal(out, torch.full_like(q, 2.0)), "the retry's result"
    assert _priority() == sdpa_orders.torch_own
    assert len(caplog.records) == 1
    assert "no valid execution plan" in caplog.records[0].message
    assert kernel.calls == 3


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"),
        pytest.param(
            getattr(torch, "AcceleratorError", RuntimeError)("CUDA error: an illegal memory access was encountered"),
            marks=pytest.mark.skipif(not hasattr(torch, "AcceleratorError"), reason="this torch has no such class"),
        ),
    ],
    ids=["allocator", "cudnn workspace", "sticky cuda error"],
)
def test_an_error_that_is_not_cudnns_propagates_untouched(install, monkeypatch, error, sdpa_orders):
    """Out of memory is the model cache's business, and another kernel would not make room. A sticky CUDA
    error from earlier work surfaces at whatever launch comes next and poisons the context: demoting cuDNN
    would announce a recovery the retry cannot make, and the report it asks for would blame cuDNN."""
    kernel = _Kernel(error)
    install(kernel)
    _dispatcher_names(monkeypatch, SDPBackend.CUDNN_ATTENTION)
    q = _qkv()

    with pytest.raises(type(error)) as raised:
        F.scaled_dot_product_attention(q, q, q)

    assert raised.value is error
    assert _priority() == sdpa_orders.installed
    assert kernel.calls == 1


def test_a_failure_on_another_kernel_propagates_untouched(install, monkeypatch, sdpa_orders):
    error = RuntimeError("memory-efficient attention failed")
    kernel = _Kernel(error)
    install(kernel)
    _dispatcher_names(monkeypatch, SDPBackend.EFFICIENT_ATTENTION)
    q = _qkv()

    with pytest.raises(RuntimeError) as raised:
        F.scaled_dot_product_attention(q, q, q)

    assert raised.value is error
    assert _priority() == sdpa_orders.installed
    assert kernel.calls == 1


def test_cudnn_asked_for_exclusively_propagates_untouched(install, monkeypatch, sdpa_orders):
    """`INVOKE_KREA2_SDPA_BACKEND=cudnn`: math is disabled, and the user wanted to see cuDNN fail.

    The choice is forced to cuDNN here too -- on CPU tensors the math-disabled branch would otherwise
    answer None by itself -- so that disabled math is the only thing that can make it propagate.
    """
    error = RuntimeError("cuDNN Frontend error")
    kernel = _Kernel(error)
    install(kernel)
    monkeypatch.setattr(attention, "sdpa_backend_choice", lambda *args, **kwargs: SDPBackend.CUDNN_ATTENTION)
    q = _qkv()

    with sdpa_policy([SDPBackend.CUDNN_ATTENTION]), pytest.raises(RuntimeError) as raised:
        F.scaled_dot_product_attention(q, q, q)

    assert raised.value is error
    assert _priority() == sdpa_orders.installed


def test_a_retry_that_fails_too_propagates(install, monkeypatch, sdpa_orders):
    retry_error = RuntimeError("memory-efficient attention failed as well")
    kernel = _Kernel(RuntimeError("cuDNN Frontend error"), retry_error)
    install(kernel)
    _dispatcher_names(monkeypatch, SDPBackend.CUDNN_ATTENTION)
    q = _qkv()

    with pytest.raises(RuntimeError) as raised:
        F.scaled_dot_product_attention(q, q, q)

    assert raised.value is retry_error
    assert _priority() == sdpa_orders.torch_own
    assert kernel.calls == 2


def test_two_workers_failing_at_once_both_recover_and_warn_once(install, monkeypatch, caplog, sdpa_orders):
    """Both calls are inside the kernel before either fails. The second to reach the fallback finds
    cuDNN already demoted -- the dispatcher no longer names it for the call -- and must retry anyway."""
    both_inside = threading.Barrier(2, timeout=5)
    failed: set[int] = set()

    def kernel(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        if threading.get_ident() not in failed:
            failed.add(threading.get_ident())
            both_inside.wait()
            raise RuntimeError("cuDNN Frontend error")
        return query.clone()

    install(kernel)
    _dispatcher_names(monkeypatch, SDPBackend.CUDNN_ATTENTION)
    results: list[torch.Tensor] = []

    def worker() -> None:
        q = _qkv()
        results.append(F.scaled_dot_product_attention(q, q, q))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

    assert len(results) == 2
    assert len(caplog.records) == 1
    assert _priority() == sdpa_orders.torch_own


def test_installing_twice_wraps_once(install, monkeypatch):
    kernel = _Kernel(RuntimeError("cuDNN Frontend error"), RuntimeError("still failing"))
    install(kernel)
    wrapped = F.scaled_dot_product_attention
    install_cudnn_sdpa_fallback()
    assert F.scaled_dot_product_attention is wrapped

    _dispatcher_names(monkeypatch, SDPBackend.CUDNN_ATTENTION)
    q = _qkv()
    with pytest.raises(RuntimeError, match="still failing"):
        F.scaled_dot_product_attention(q, q, q)
    assert kernel.calls == 2, "one retry, not one per layer"


def test_nothing_is_installed_where_there_is_no_cudnn_attention(install, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    kernel = _Kernel()
    install(kernel)
    assert F.scaled_dot_product_attention is kernel


HQ, HKV = 8, 2


def _grouped_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(1, HQ, 8, 16), torch.randn(1, HKV, 8, 16), torch.randn(1, HKV, 8, 16)


def test_a_grouped_query_call_is_retried_with_expanded_heads(install, monkeypatch):
    """Krea-2 hands SDPA unexpanded K/V only because cuDNN took them. With cuDNN demoted no fused kernel
    takes grouped-query attention, so the unchanged call would build the full score matrix on math --
    gigabytes per call, priced nowhere. The retry must not be that call."""
    seen: list[tuple[int, bool]] = []

    def kernel(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        seen.append((key.shape[1], enable_gqa))
        if len(seen) == 1:
            raise RuntimeError("cuDNN Frontend error")
        return query.clone()

    install(kernel)
    _dispatcher_names(monkeypatch, SDPBackend.CUDNN_ATTENTION)
    q, k, v = _grouped_qkv()

    F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    assert seen == [(HKV, True), (HQ, False)], "first as asked, then expanded without enable_gqa"


def test_the_expanded_retry_is_the_same_attention(install, monkeypatch):
    """Against torch's own grouped-query result, so the expansion's axis and head order are proven rather
    than assumed: query head h attends to K/V head h // (HQ // HKV)."""
    failed = []

    def kernel(*args, **kwargs):
        if not failed:
            failed.append(True)
            raise RuntimeError("cuDNN Frontend error")
        return TORCH_SDPA(*args, **kwargs)

    install(kernel)
    _dispatcher_names(monkeypatch, SDPBackend.CUDNN_ATTENTION)
    q, k, v = _grouped_qkv()

    retried = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    torch.testing.assert_close(retried, TORCH_SDPA(q, k, v, enable_gqa=True))
