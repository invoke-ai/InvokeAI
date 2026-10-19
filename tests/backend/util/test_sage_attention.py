"""SageAttention routing for `F.scaled_dot_product_attention` (`backend.util.sage_attention`).

The wrapper must hand exactly the eligible calls made inside a `sage_attention_scope` to the SageAttention kernel for
the call's GPU, pass every other call to torch with its arguments untouched, keep one thread's scope from leaking into
another, check each GPU's first result against torch, fall back for good on a device whose kernel failed, and install
only where SageAttention can run at all.

CPU tensors stand in for the CUDA device and a fake `sageattention` module for the kernels, so none of this establishes
how the real kernels behave; `TestOnNvidiaHardware` does, on a GPU with SageAttention installed (`-m slow`).
"""

import contextlib
import functools
import importlib.util
import logging
import threading
from collections import Counter
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import invokeai.backend.util.sage_attention as sage
from invokeai.backend.util.sage_attention import install_sage_attention, sage_attention_scope

# torch's own function, captured before any test installs the wrapper.
TORCH_SDPA = F.scaled_dot_product_attention
VERSION = "2.2.0+cu130torch2.10.0andhigher.post6"
KERNELS = (
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda_sm90",
)
FIXTURE = Path(__file__).parent / "data" / "qwen_image_first_block_outlier_head.safetensors"


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """Attention in float32 over [B, H, S, D], K/V heads repeated for GQA."""
    q, k, v = q.float(), k.float(), v.float()
    if k.shape[1] != q.shape[1]:
        k, v = (t.repeat_interleave(q.shape[1] // t.shape[1], dim=1) for t in (k, v))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v


class FakeSageAttention:
    """Stands in for the `sageattention` module. Its kernels take the real kernels' named parameters, record each
    call and return the exact result -- or `result`, when a test sets one."""

    def __init__(self, without: tuple[str, ...] = ()) -> None:
        self.calls: list[dict] = []
        self.error: Exception | None = None
        self.result: torch.Tensor | None = None
        for name in KERNELS:
            if name not in without:
                setattr(self, name, self._kernel(name))

    def _kernel(self, name: str):
        def kernel(
            q, k, v, tensor_layout="HND", is_causal=False, qk_quant_gran="per_thread", sm_scale=None,
            pv_accum_dtype="fp32", smooth_k=True, smooth_v=False, return_lse=False, **kwargs,
        ):  # fmt: skip
            self.calls.append(
                {"kernel": name, "layout": tensor_layout, "causal": is_causal, "sm_scale": sm_scale, "k": k.shape,
                 "qk_quant_gran": qk_quant_gran, "pv_accum_dtype": pv_accum_dtype, "smooth_k": smooth_k}
            )  # fmt: skip
            if self.error is not None:
                raise self.error
            if self.result is not None:
                return self.result
            return _reference(q, k, v, sm_scale).to(q.dtype)

        kernel.__name__ = name
        return kernel


def _q(*shape: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return torch.randn(*shape).to(dtype)


# Long enough for SageAttention by every rule, small enough to be cheap on the CPU.
QUERY_LEN = sage._MIN_QUERY_LEN
KEY_LEN = sage._MIN_KEY_LEN


def _cuda_build(
    monkeypatch, capabilities: dict[int, tuple[int, int]], module: FakeSageAttention | None = None
) -> tuple[FakeSageAttention, list[dict]]:
    """Make this build look like CUDA 13 with the given devices and the fake module the installed SageAttention, with
    every device's first-call check already passed. Returns the module and the argument tuples of every call that
    reached torch's function (computed in float32: no CPU half-precision GEMM)."""
    module = module or FakeSageAttention()
    torch_calls: list[dict] = []

    def torch_spy(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *args, scale=None, enable_gqa=False, **kwargs
    ):
        torch_calls.append(
            {"mask": attn_mask, "dropout": dropout_p, "causal": is_causal, "scale": scale, "gqa": enable_gqa,
             "args": args, "kwargs": kwargs, "q": query.shape}
        )  # fmt: skip
        mask = attn_mask if attn_mask is None or attn_mask.dtype == torch.bool else attn_mask.float()
        out = TORCH_SDPA(
            query.float(), key.float(), value.float(), mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa
        )
        return out.to(query.dtype)

    monkeypatch.setattr(F, "scaled_dot_product_attention", torch_spy)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capabilities[index or 0])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: "Fake GPU")
    monkeypatch.setattr(sage, "_import_sageattention", lambda: (VERSION, module))
    monkeypatch.setattr(sage, "_cuda_index", lambda q, k, v: 0)
    monkeypatch.setattr(sage, "_disabled_devices", set())
    monkeypatch.setattr(sage, "_validated", _all_variants(capabilities))
    monkeypatch.setattr(sage, "_cache_failures", Counter())
    monkeypatch.setattr(sage, "_announced_devices", set())
    monkeypatch.setattr(torch.cuda, "device", lambda index: contextlib.nullcontext())
    return module, torch_calls


def _all_variants(devices) -> set:
    """Every kernel variant of these devices, as if each had passed its first-call check."""
    return {
        (index, dtype, head_dim) for index in devices for dtype in sage._HALF_PRECISION for head_dim in sage._HEAD_DIMS
    }


@pytest.fixture
def installed(monkeypatch):
    """The wrapper installed on a build that looks like CUDA 13 with one sm_89 device."""
    module, torch_calls = _cuda_build(monkeypatch, {0: (8, 9)})
    spy = F.scaled_dot_product_attention
    install_sage_attention()
    assert F.scaled_dot_product_attention is not spy
    return module, torch_calls


class TestKernelTable:
    @pytest.mark.parametrize(
        ("capability", "cuda", "kernel", "options"),
        [
            ((8, 0), (13, 0), "sageattn_qk_int8_pv_fp16_cuda", {"pv_accum_dtype": "fp32"}),
            # Upstream's `sageattn` picks the Triton kernel here; the Windows build moved sm86 to CUDA unexplained.
            ((8, 6), (13, 0), "sageattn_qk_int8_pv_fp16_triton", {}),
            ((8, 9), (13, 0), "sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32+fp16"}),
            ((8, 9), (12, 6), "sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32+fp32"}),
            ((9, 0), (13, 0), "sageattn_qk_int8_pv_fp8_cuda_sm90", {"pv_accum_dtype": "fp32+fp32"}),
            ((12, 0), (13, 0), "sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32+fp16", "qk_quant_gran": "per_warp"}),
            ((12, 0), (12, 6), "sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32", "qk_quant_gran": "per_warp"}),
        ],
        ids=["ampere-a100", "ampere-rtx30", "ada", "ada-cuda12.6", "hopper", "blackwell", "blackwell-cuda12.6"],
    )  # fmt: skip
    def test_each_generation_gets_its_kernel_with_smoothing_off(self, capability, cuda, kernel, options):
        """K smoothing broke the first block of Qwen-Image and Krea-2 (cosine 0.18 and 0.53); FP16 accumulation inside
        the FP8 kernels needs CUDA 12.8. See `_kernel`."""
        module = FakeSageAttention()
        q = _q(1, 2, 8, 64)

        sage._kernel(module, capability, cuda)(q, q, q, tensor_layout="HND", is_causal=False, sm_scale=None)

        assert module.calls[0]["kernel"] == kernel
        assert {name: module.calls[0][name] for name in (*options, "smooth_k")} == options | {"smooth_k": False}

    @pytest.mark.parametrize(
        "capability", [(7, 5), (8, 7), (10, 3), (11, 0)], ids=["turing", "jetson-orin", "sm103", "sm110"]
    )
    def test_gpus_sageattention_has_no_kernel_for_get_none(self, capability):
        assert sage._kernel(FakeSageAttention(), capability, (13, 0)) is None


class TestRouting:
    def test_an_eligible_call_in_scope_runs_on_the_kernel_in_hnd_layout(self, installed):
        module, torch_calls = installed
        q, k, v = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64), _q(1, 2, KEY_LEN, 64)

        with sage_attention_scope():
            out = F.scaled_dot_product_attention(q, k, v)

        assert torch_calls == []
        call = module.calls[0]
        assert (call["layout"], call["causal"], call["sm_scale"], call["k"]) == ("HND", False, 64**-0.5, k.shape)
        torch.testing.assert_close(out.float(), _reference(q, k, v), atol=2e-3, rtol=2e-3)

    def test_an_explicit_scale_reaches_the_kernel(self, installed):
        module, _ = installed
        q, k = _q(1, 2, QUERY_LEN, 128), _q(1, 2, KEY_LEN, 128)

        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k, scale=0.3)

        assert module.calls[0]["sm_scale"] == 0.3

    def test_outside_a_scope_every_call_reaches_torch_untouched(self, installed):
        """Text encoders and VAEs run outside any scope, so they must keep torch's exact semantics."""
        module, torch_calls = installed
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)

        F.scaled_dot_product_attention(q, k, k, None, 0.0, False, scale=0.5)

        assert module.calls == []
        assert torch_calls == [
            {"mask": None, "dropout": 0.0, "causal": False, "scale": 0.5, "gqa": False, "args": (), "kwargs": {},
             "q": q.shape}
        ]  # fmt: skip

    @pytest.mark.parametrize(
        ("make", "reason"),
        [
            (lambda: ((_q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)),
                      {"attn_mask": torch.ones(QUERY_LEN, KEY_LEN, dtype=torch.bool)}), "mask"),
            (lambda: ((_q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)),
                      {"attn_mask": torch.zeros(1, 1, 1, KEY_LEN, dtype=torch.float16)}), "mask"),
            (lambda: ((_q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)), {"dropout_p": 0.1}), "dropout"),
            (lambda: ((_q(1, 2, QUERY_LEN, 64), _q(1, 2, QUERY_LEN, 64)), {"is_causal": True}), "causal"),
            (lambda: ((_q(1, 2, QUERY_LEN, 64, dtype=torch.float32), _q(1, 2, KEY_LEN, 64, dtype=torch.float32)), {}),
             "dtype"),
            (lambda: ((_q(1, 2, QUERY_LEN, 40), _q(1, 2, KEY_LEN, 40)), {}), "head_dim"),
            (lambda: ((_q(1, 2, QUERY_LEN, 80), _q(1, 2, KEY_LEN, 80)), {}), "head_dim"),
            (lambda: ((_q(1, 2, QUERY_LEN, 256), _q(1, 2, KEY_LEN, 256)), {}), "head_dim"),
            (lambda: ((_q(2, QUERY_LEN, 64), _q(2, KEY_LEN, 64)), {}), "shape"),
            (lambda: ((_q(2, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)), {}), "shape"),
            (lambda: ((_q(1, 2, QUERY_LEN - 1, 64), _q(1, 2, KEY_LEN, 64)), {}), "short"),
            (lambda: ((_q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN - 1, 64)), {}), "short"),
            (lambda: ((_q(1, 2, QUERY_LEN, 128)[..., ::2], _q(1, 2, KEY_LEN, 64)), {}), "layout"),
        ],
        ids=["bool-mask", "additive-mask", "dropout", "causal", "fp32", "hd40", "hd80", "hd256", "3d",
             "broadcast-batch", "short-query", "short-key", "strided-last-dim"],
    )  # fmt: skip
    def test_ineligible_calls_reach_torch_with_their_arguments(self, installed, caplog, make, reason):
        module, torch_calls = installed
        (q, k), kwargs = make()
        caplog.set_level(logging.DEBUG, logger=sage.__name__)

        with sage_attention_scope():
            out = F.scaled_dot_product_attention(q, k, k, **kwargs)

        assert module.calls == []
        assert len(torch_calls) == 1
        assert torch_calls[0]["mask"] is kwargs.get("attn_mask")
        assert torch_calls[0]["dropout"] == kwargs.get("dropout_p", 0.0)
        assert torch_calls[0]["causal"] == kwargs.get("is_causal", False)
        assert torch_calls[0]["gqa"] == kwargs.get("enable_gqa", False)
        assert out.dtype == q.dtype
        assert any(f"on PyTorch SDPA: {reason}=1" in r.message for r in caplog.records)

    @pytest.mark.parametrize(("kv_heads", "enable_gqa"), [(2, False), (3, True)], ids=["gqa-flag-off", "indivisible"])
    def test_mismatched_heads_get_torchs_error_not_a_kernel_answer(self, installed, kv_heads, enable_gqa):
        module, torch_calls = installed
        q, k = _q(1, 8, QUERY_LEN, 64), _q(1, kv_heads, KEY_LEN, 64)

        with sage_attention_scope(), pytest.raises(RuntimeError):
            F.scaled_dot_product_attention(q, k, k, enable_gqa=enable_gqa)

        assert module.calls == [] and len(torch_calls) == 1

    def test_an_input_that_requires_grad_under_grad_mode_reaches_torch(self, installed):
        module, torch_calls = installed
        q = _q(1, 2, QUERY_LEN, 64).requires_grad_()
        k = _q(1, 2, KEY_LEN, 64)

        with torch.enable_grad(), sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k)

        assert module.calls == [] and len(torch_calls) == 1

    def test_a_call_under_cuda_autocast_reaches_torch(self, installed, monkeypatch):
        """Autocast would hand SageAttention whatever dtype it picks; torch's SDPA applies autocast's rules itself."""
        module, torch_calls = installed
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda device_type=None: device_type == "cuda")
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)

        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k)

        assert module.calls == [] and len(torch_calls) == 1

    @pytest.mark.filterwarnings("ignore:The PyTorch API of nested tensors")
    def test_a_nested_query_is_not_eligible(self):
        """Nested (jagged) tensors report four dimensions but have no fixed sequence length to quantize over."""
        q = torch.nested.nested_tensor([_q(2, QUERY_LEN, 64), _q(2, QUERY_LEN + 1, 64)])
        k = _q(2, 2, KEY_LEN, 64)
        assert sage._ineligible(q, k, k, None, 0.0, False, False, False) == "shape"

    def test_an_unknown_keyword_reaches_torch_rather_than_being_dropped(self, installed):
        module, torch_calls = installed
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)

        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k, some_future_flag=True)

        assert module.calls == [] and torch_calls[0]["kwargs"] == {"some_future_flag": True}

    def test_grouped_query_attention_reaches_the_kernel_unexpanded(self, installed):
        module, torch_calls = installed
        q, k, v = _q(1, 8, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64), _q(1, 2, KEY_LEN, 64)

        with sage_attention_scope():
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        assert torch_calls == [] and module.calls[0]["k"] == k.shape
        torch.testing.assert_close(out.float(), _reference(q, k, v), atol=2e-3, rtol=2e-3)


class TestFirstCallCheck:
    def test_a_devices_first_result_is_compared_with_torch_once(self, monkeypatch):
        module, torch_calls = _cuda_build(monkeypatch, {0: (8, 9)})
        monkeypatch.setattr(sage, "_validated", set())
        install_sage_attention()
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)

        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k)
            F.scaled_dot_product_attention(q, k, k)

        assert len(module.calls) == 2, "both calls ran on the kernel"
        assert len(torch_calls) == 1, "only the first was also computed by torch, for the comparison"

    def test_the_check_pairs_a_gqa_query_group_with_its_own_kv_head(self, monkeypatch):
        """Query heads 0-3 attend to K/V head 0; comparing them with head 1 instead would retire a correct kernel."""
        module, torch_calls = _cuda_build(monkeypatch, {0: (8, 9)})
        monkeypatch.setattr(sage, "_validated", set())
        install_sage_attention()
        q, k, v = _q(1, 8, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64), _q(1, 2, KEY_LEN, 64)
        v[:, 1] += 3.0  # a K/V head unlike head 0, so a wrong pairing cannot pass

        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        assert len(module.calls) == 2, "the check passed, so the device stays on the kernel"
        assert len(torch_calls) == 1 and torch_calls[0]["gqa"] and torch_calls[0]["q"] == (1, 4, 1024, 64)

    def test_each_precision_and_head_size_is_checked_on_its_first_call(self, monkeypatch):
        """They select different compiled kernels; one passing says nothing about the others."""
        module, torch_calls = _cuda_build(monkeypatch, {0: (8, 9)})
        monkeypatch.setattr(sage, "_validated", set())
        install_sage_attention()
        hd64, hd128 = _q(1, 2, QUERY_LEN, 64), _q(1, 2, QUERY_LEN, 128)
        bf16 = _q(1, 2, QUERY_LEN, 64, dtype=torch.bfloat16)

        with sage_attention_scope():
            for q in (hd64, hd64, hd128, hd128, bf16, bf16):
                F.scaled_dot_product_attention(q, q, q)

        assert len(module.calls) == 6
        assert len(torch_calls) == 3, "one comparison per variant"

    @pytest.mark.parametrize("garbage", [torch.nan, 50.0], ids=["non-finite", "wrong"])
    def test_a_wrong_first_result_retires_the_device_and_returns_torchs(self, monkeypatch, caplog, garbage):
        """A kernel the build lacks for this GPU returns uninitialized memory instead of raising."""
        module, torch_calls = _cuda_build(monkeypatch, {0: (8, 9)})
        monkeypatch.setattr(sage, "_validated", set())
        install_sage_attention()
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)
        module.result = torch.full((1, 2, QUERY_LEN, 64), garbage, dtype=torch.float16)
        caplog.set_level(logging.WARNING, logger=sage.__name__)

        with sage_attention_scope():
            out = F.scaled_dot_product_attention(q, k, k)
            F.scaled_dot_product_attention(q, k, k)

        torch.testing.assert_close(out.float(), _reference(q, k, k), atol=2e-3, rtol=2e-3)
        assert len(module.calls) == 1, "a retired device must not be tried again"
        assert any("first result differs from PyTorch SDPA" in r.message for r in caplog.records)


class TestFailures:
    def test_a_kernel_failure_falls_back_warns_once_and_retires_only_that_device(self, monkeypatch, caplog):
        module, _ = _cuda_build(monkeypatch, {0: (8, 9), 1: (8, 9)})
        install_sage_attention()
        device = {"index": 0}
        monkeypatch.setattr(sage, "_cuda_index", lambda q, k, v: device["index"])
        module.error = RuntimeError("kernel not compiled for this architecture")
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)
        caplog.set_level(logging.WARNING, logger=sage.__name__)

        with sage_attention_scope():
            first = F.scaled_dot_product_attention(q, k, k)
            F.scaled_dot_product_attention(q, k, k)

        assert len(module.calls) == 1, "a retired device must not be tried again"
        torch.testing.assert_close(first.float(), _reference(q, k, k), atol=2e-3, rtol=2e-3)
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and "cuda:0" in warnings[0]

        module.error = None
        device["index"] = 1
        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k)
        assert len(module.calls) == 2, "another device is unaffected"

    @pytest.mark.parametrize(
        "error",
        [
            torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.50 GiB"),
            RuntimeError("Triton Error [CUDA]: out of memory"),
        ],
        ids=["torch-oom", "triton-oom"],
    )
    def test_running_out_of_memory_moves_the_rest_of_the_scope_to_torch_only(self, installed, caplog, error):
        """SDPA needs less memory. Retrying every call would flush the allocator's cache each time; retiring the
        device would cost every later, smaller generation SageAttention."""
        module, torch_calls = installed
        module.error = error
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)
        caplog.set_level(logging.INFO, logger=sage.__name__)

        with sage_attention_scope():
            out = F.scaled_dot_product_attention(q, k, k)
            module.error = None
            F.scaled_dot_product_attention(q, k, k)
        assert len(module.calls) == 1 and len(torch_calls) == 2, "after the OOM, this scope stayed on torch"
        torch.testing.assert_close(out.float(), _reference(q, k, k), atol=2e-3, rtol=2e-3)

        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k)
        assert len(module.calls) == 2, "the next scope tries SageAttention again"
        assert any("ran out of memory" in r.message for r in caplog.records if r.levelno == logging.INFO)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_a_triton_cache_failure_falls_back_for_that_call_only(self, installed, caplog):
        """On Windows two GPUs compiling the same kernel can race for its cache file; that says nothing about it."""
        module, torch_calls = installed
        module.error = PermissionError("[WinError 5] Access is denied: 'triton\\cache\\tmp.pid_123'")
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)
        caplog.set_level(logging.WARNING, logger=sage.__name__)

        with sage_attention_scope():
            out = F.scaled_dot_product_attention(q, k, k)
            module.error = None
            F.scaled_dot_product_attention(q, k, k)

        torch.testing.assert_close(out.float(), _reference(q, k, k), atol=2e-3, rtol=2e-3)
        assert len(module.calls) == 2 and len(torch_calls) == 1
        assert not caplog.records

    def test_only_cache_failures_in_a_row_retire_the_device(self, installed, caplog):
        """A cache that can never be written would recompile on every call; one that failed now and then is fine."""
        module, torch_calls = installed
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)
        caplog.set_level(logging.WARNING, logger=sage.__name__)

        def call(error):
            module.error = error
            with sage_attention_scope():
                F.scaled_dot_product_attention(q, k, k)

        for error in [PermissionError("denied")] * (sage._MAX_CACHE_FAILURES - 1) + [None]:
            call(error)
        for _ in range(sage._MAX_CACHE_FAILURES - 1):
            call(PermissionError("denied"))
        assert not caplog.records and 0 not in sage._disabled_devices, "the success in between reset the count"

        call(PermissionError("denied"))
        call(None)
        assert 0 in sage._disabled_devices
        assert len(module.calls) == 2 * sage._MAX_CACHE_FAILURES, "the last call no longer reached the kernel"
        assert any("cuda:0" in r.message for r in caplog.records if r.levelno == logging.WARNING)

    def test_the_scope_closes_when_its_block_raises(self, installed):
        module, torch_calls = installed
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)

        with pytest.raises(ValueError), sage_attention_scope():
            raise ValueError("generation failed")

        F.scaled_dot_product_attention(q, k, k)
        assert module.calls == [] and len(torch_calls) == 1


class TestScopeIsPerThread:
    def test_only_the_thread_inside_the_scope_reaches_the_kernel(self, installed):
        """One generation session per GPU runs concurrently; a scope must not leak into the other session."""
        module, torch_calls = installed
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)
        both_inside = threading.Barrier(2)
        errors: list[BaseException] = []

        def scoped() -> None:
            try:
                with sage_attention_scope():
                    both_inside.wait(timeout=30)
                    F.scaled_dot_product_attention(q, k, k)
                    both_inside.wait(timeout=30)
            except BaseException as e:  # noqa: BLE001 - re-raised in the main thread
                errors.append(e)

        def unscoped() -> None:
            try:
                both_inside.wait(timeout=30)
                F.scaled_dot_product_attention(q, k, k)
                both_inside.wait(timeout=30)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=scoped), threading.Thread(target=unscoped)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == []
        assert len(module.calls) == 1 and len(torch_calls) == 1


class TestDeviceRule:
    @staticmethod
    def _on(index: int) -> SimpleNamespace:
        return SimpleNamespace(device=torch.device("cuda", index))

    def test_tensors_on_one_cuda_device_give_its_index(self):
        assert sage._cuda_index(self._on(1), self._on(1), self._on(1)) == 1

    def test_mixed_devices_and_cpu_do_not(self):
        assert sage._cuda_index(self._on(0), self._on(1), self._on(0)) is None
        cpu = SimpleNamespace(device=torch.device("cpu"))
        assert sage._cuda_index(cpu, cpu, cpu) is None

    def test_the_kernel_runs_with_the_tensors_device_current(self, monkeypatch):
        """SageAttention launches on the thread's current device. A single-device install with `device: cuda:1` never
        makes that device current, so the wrapper does, for the call only."""
        module, _ = _cuda_build(monkeypatch, {0: (8, 9), 1: (8, 9)})
        events: list[str] = []
        kernel = module.sageattn_qk_int8_pv_fp8_cuda

        @functools.wraps(kernel)
        def recording_kernel(*args, **kwargs):
            events.append("kernel")
            return kernel(*args, **kwargs)

        @contextlib.contextmanager
        def recording_device(index):
            events.append(f"enter cuda:{index}")
            yield
            events.append("exit")

        module.sageattn_qk_int8_pv_fp8_cuda = recording_kernel
        monkeypatch.setattr(torch.cuda, "device", recording_device)
        monkeypatch.setattr(sage, "_cuda_index", lambda q, k, v: 1)
        install_sage_attention()
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)

        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k)

        assert events == ["enter cuda:1", "kernel", "exit"]


def _raise(error: Exception):
    def raiser():
        raise error

    return raiser


class TestInstall:
    def test_install_is_idempotent(self, monkeypatch):
        _cuda_build(monkeypatch, {0: (8, 9)})
        spy = F.scaled_dot_product_attention
        install_sage_attention()
        wrapped = F.scaled_dot_product_attention
        install_sage_attention()
        assert F.scaled_dot_product_attention is wrapped
        assert wrapped.__wrapped__ is spy

    @pytest.mark.parametrize(
        ("patch", "message"),
        [
            (lambda m: m.setattr(torch.version, "hip", "7.2"), "NVIDIA GPU with CUDA"),
            (lambda m: m.setattr(torch.cuda, "is_available", lambda: False), "NVIDIA GPU with CUDA"),
            (lambda m: m.setattr(torch.cuda, "get_device_capability", lambda index=None: (7, 5)), "8.0 or newer"),
            (lambda m: m.setattr(sage, "_import_sageattention", _raise(ModuleNotFoundError("x", name="sageattention"))),
             "`sageattention` package is not installed"),
            (lambda m: m.setattr(sage, "_import_sageattention", _raise(ModuleNotFoundError("x", name="triton.language"))),
             "triton-windows"),
            (lambda m: m.setattr(sage, "_import_sageattention", _raise(PackageNotFoundError("sageattention"))),
             "no package metadata"),
            (lambda m: m.setattr(sage, "_import_sageattention", _raise(OSError("DLL load failed"))), "DLL load failed"),
            (lambda m: m.setattr(sage, "_import_sageattention", lambda: ("1.0.6", FakeSageAttention())),
             "SageAttention 1"),
            (lambda m: m.setattr(sage, "_import_sageattention",
                                 lambda: (VERSION, FakeSageAttention(without=("sageattn_qk_int8_pv_fp8_cuda",)))),
             "sageattn_qk_int8_pv_fp8_cuda"),
        ],
        ids=["rocm", "no-cuda", "turing", "no-package", "no-triton", "no-metadata", "broken-import", "sageattention-1",
             "missing-kernel"],
    )  # fmt: skip
    def test_install_keeps_torchs_function_and_says_why(self, monkeypatch, caplog, patch, message):
        _cuda_build(monkeypatch, {0: (8, 9)})
        spy = F.scaled_dot_product_attention
        patch(monkeypatch)
        caplog.set_level(logging.WARNING, logger=sage.__name__)

        install_sage_attention()

        assert F.scaled_dot_product_attention is spy
        assert any(message in r.message for r in caplog.records), [r.message for r in caplog.records]

    def test_a_kernel_that_would_swallow_smooth_k_is_refused(self, monkeypatch, caplog):
        """Every real kernel ends in **kwargs, so a renamed `smooth_k` would be ignored and smoothing come back."""
        module = FakeSageAttention()

        def renamed(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, pv_accum_dtype="fp32", **kwargs):
            raise AssertionError("must not be called")

        renamed.__name__ = "sageattn_qk_int8_pv_fp8_cuda"
        module.sageattn_qk_int8_pv_fp8_cuda = renamed
        _cuda_build(monkeypatch, {0: (8, 9)}, module)
        spy = F.scaled_dot_product_attention
        caplog.set_level(logging.WARNING, logger=sage.__name__)

        install_sage_attention()

        assert F.scaled_dot_product_attention is spy
        assert any("does not take smooth_k" in r.message for r in caplog.records)

    @pytest.mark.parametrize(("version", "installs"), [("2.2", True), ("3.0", True), ("2.1.1", False)])
    def test_versions_are_compared_by_their_numbers(self, monkeypatch, version, installs):
        _cuda_build(monkeypatch, {0: (8, 9)})
        monkeypatch.setattr(sage, "_import_sageattention", lambda: (version, FakeSageAttention()))

        install_sage_attention()

        assert getattr(F.scaled_dot_product_attention, sage._SENTINEL, False) is installs

    def test_a_device_without_a_kernel_keeps_torch_while_others_serve(self, monkeypatch, caplog):
        module, _ = _cuda_build(monkeypatch, {0: (8, 9), 1: (7, 5)})
        caplog.set_level(logging.INFO, logger=sage.__name__)
        install_sage_attention()
        device = {"index": 1}
        monkeypatch.setattr(sage, "_cuda_index", lambda q, k, v: device["index"])
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)

        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k)
            device["index"] = 0
            F.scaled_dot_product_attention(q, k, k)

        assert len(module.calls) == 1, "only the sm_89 device runs SageAttention"
        assert any("no SageAttention kernel for cuda:1" in r.message for r in caplog.records)

    @pytest.mark.parametrize("backend", ["sage", "auto"])
    def test_apply_monkeypatches_installs_sageattention_only_when_configured(self, monkeypatch, backend):
        from invokeai.app.util.startup_utils import apply_monkeypatches

        _cuda_build(monkeypatch, {0: (8, 9)})
        monkeypatch.setattr(torch.cuda, "empty_cache", torch.cuda.empty_cache)

        apply_monkeypatches(backend)

        assert getattr(F.scaled_dot_product_attention, sage._SENTINEL, False) is (backend == "sage")


class TestLogging:
    def test_first_use_on_a_device_is_announced_once_with_its_kernel(self, installed, caplog):
        q, k = _q(1, 2, QUERY_LEN, 64), _q(1, 2, KEY_LEN, 64)
        caplog.set_level(logging.INFO, logger=sage.__name__)

        with sage_attention_scope():
            F.scaled_dot_product_attention(q, k, k)
            F.scaled_dot_product_attention(q, k, k)

        announcements = [r.message for r in caplog.records if "serves diffusion-model attention" in r.message]
        assert announcements == [
            f"SageAttention {VERSION} serves diffusion-model attention on cuda:0 "
            "(Fake GPU, sm_89, sageattn_qk_int8_pv_fp8_cuda)."
        ]


_HAVE_SAGE = importlib.util.find_spec("sageattention") is not None


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available() or not _HAVE_SAGE, reason="needs an NVIDIA GPU and sageattention")
class TestOnNvidiaHardware:
    """Real kernels: needs CUDA, `sageattention` >= 2.2 and compute capability 8.0+, and about 10 GB of free VRAM for
    the float32 references. Measured on an RTX 4090."""

    @pytest.fixture(params=["own", "sm86-triton"])
    def kernel(self, request, monkeypatch):
        """The kernel this GPU gets, or the Triton kernel an sm86 card gets: it runs on any Ampere or newer GPU, so this
        one stands in for an sm86 card that is not here."""
        import sageattention

        capability = torch.cuda.get_device_capability(0)
        if capability < (8, 0):
            pytest.skip("SageAttention 2 needs compute capability 8.0 or newer")
        if request.param == "sm86-triton":
            pick = sage._kernel
            monkeypatch.setattr(sage, "_kernel", lambda module, _capability, cuda: pick(module, (8, 6), cuda))
        monkeypatch.setattr(F, "scaled_dot_product_attention", TORCH_SDPA)
        monkeypatch.setattr(sage, "_disabled_devices", set())
        monkeypatch.setattr(sage, "_validated", set())
        monkeypatch.setattr(sage, "_cache_failures", Counter())
        monkeypatch.setattr(sage, "_announced_devices", set())
        install_sage_attention()
        if not getattr(F.scaled_dot_product_attention, sage._SENTINEL, False):
            pytest.skip("the installed sageattention is not one InvokeAI uses (see the startup warning)")
        cuda = sage._parse_version(torch.version.cuda)
        assert cuda is not None
        return sage._kernel(sageattention, capability, cuda[:2])

    def test_every_kernel_takes_the_arguments_it_is_given(self, kernel, monkeypatch):
        import sageattention

        monkeypatch.undo()  # the real table, not the fixture's stand-in
        for capability in ((8, 0), (8, 6), (8, 9), (9, 0), (12, 0)):
            assert sage._unaccepted_arguments(sage._kernel(sageattention, capability, (13, 0))) == []

    @pytest.mark.parametrize(
        ("shape_q", "shape_kv", "dtype", "gqa"),
        [
            ((1, 24, 4608, 128), (1, 24, 4608, 128), torch.bfloat16, False),  # FLUX.1 1024^2
            ((2, 10, 4096, 64), (2, 10, 4096, 64), torch.float16, False),  # SDXL 1024^2, level 1
            ((1, 48, 4608, 128), (1, 12, 4608, 128), torch.bfloat16, True),  # Krea-2 GQA
            # Lengths that are no multiple of a tile, and cross-attention to a shorter key.
            ((1, 24, 4429, 128), (1, 24, 845, 128), torch.bfloat16, False),
        ],
        ids=["flux", "sdxl", "krea2-gqa", "partial-tile-cross"],
    )
    def test_output_stays_close_to_a_float32_reference(self, kernel, shape_q, shape_kv, dtype, gqa):
        g = torch.Generator(device="cuda").manual_seed(0)

        def make(b, h, s, d):  # real activations arrive as [B, S, H, D] storage viewed as [B, H, S, D]
            return torch.randn(b, s, h, d, device="cuda", generator=g).to(dtype).transpose(1, 2)

        q, k, v = make(*shape_q), make(*shape_kv), make(*shape_kv)
        with sage_attention_scope():
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=gqa)

        # SDPA would pass the bounds below too; the kernel must have served the call.
        assert torch.equal(out, kernel(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=shape_q[-1] ** -0.5))
        ref = _reference(q, k, v)
        assert torch.isfinite(out).all()
        cos = F.cosine_similarity(out.float().flatten(2), ref.flatten(2), dim=-1)
        # Measured on random inputs: worst head 0.9991-0.9993; SDPA in bf16 reaches 0.99999.
        assert cos.min() > 0.998
        assert (out.float() - ref).norm() / ref.norm() < 0.06

    def test_the_outlier_channel_of_qwen_images_first_block_stays_accurate(self, kernel):
        """Real activations (see the fixture's metadata). With K smoothing on, as `sageattn` runs it, the relative
        error here is 1.62; with it off, 0.10."""
        from safetensors.torch import load_file

        t = {name: tensor.cuda() for name, tensor in load_file(FIXTURE).items()}

        out = kernel(t["q"], t["k"], t["v"], tensor_layout="HND", is_causal=False, sm_scale=128**-0.5)

        ref = _reference(t["q"], t["k"], t["v"])
        assert (out.float() - ref).norm() / ref.norm() < 0.2

    def test_in_scope_it_is_the_kernel_exactly_and_outside_it_is_torch_exactly(self, kernel):
        q, k, v = (torch.randn(1, 4608, 24, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(3))
        mask = torch.zeros(1, 1, 4608, 4608, device="cuda", dtype=torch.bfloat16)
        direct = kernel(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=128**-0.5)

        with sage_attention_scope():
            assert torch.equal(F.scaled_dot_product_attention(q, k, v), direct)
            masked = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            assert torch.equal(masked, TORCH_SDPA(q, k, v, attn_mask=mask))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                assert torch.equal(F.scaled_dot_product_attention(q, k, v), TORCH_SDPA(q, k, v))
        assert torch.equal(F.scaled_dot_product_attention(q, k, v), TORCH_SDPA(q, k, v))

    def test_extra_memory_per_call_stays_within_the_measured_bound(self, kernel):
        """Including the first call, whose check against PyTorch must not cost a full-size reference."""
        q, k, v = (torch.randn(1, 4608, 24, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for _ in range(3))
        kernel(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=128**-0.5)  # load the kernels unmeasured
        # Quantized copies of Q, K and V at ~1 byte per element; measured 1.7x that on an RTX 4090.
        bound = 2.5 * (q.numel() + k.numel() + v.numel())
        with sage_attention_scope():
            for call in ("first, checked", "later"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                out = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                extra = torch.cuda.max_memory_allocated() - before - out.numel() * out.element_size()
                assert extra <= bound, call
                del out
