"""The report for `attention_backend: cudnn | flash | efficient | math`.

A pinned kernel is ranked first and never exclusive, so attention it cannot serve runs on the next kernel without a word
from torch. The report logs that once per node type, kernel and reason. These run real `F.scaled_dot_product_attention`
calls against torch's real dispatcher and priority order; the CUDA-only reasons need a GPU.
"""

import logging
import subprocess
import sys
from typing import get_args

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

import invokeai.backend.util.attention as attention
from invokeai.app.services.config.config_default import ATTENTION_BACKEND
from invokeai.backend.util.attention import (
    attention_caller,
    install_cudnn_sdpa_fallback,
    install_pinned_sdpa_kernel_report,
    install_rocm_sdpa_guard,
    sdpa_backend_choice,
    sdpa_score_matrix_bytes,
)
from invokeai.backend.util.sdpa_scope import install_default_sdpa_priority, sdpa_policy

LOGGER = attention.__name__
# torch's own function, whatever the tests rebind.
TORCH_SDPA = F.scaled_dot_product_attention
CPU_RANKING = "on cpu PyTorch picks the kernel itself, without the ranking"
# The kernel each `attention_backend` value pins, written out rather than read from `SDPA_KERNELS`.
PINNED_BY_VALUE = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "math": SDPBackend.MATH,
}


@pytest.fixture(autouse=True)
def _forget_reports():
    """Reports are once per process; a leftover one would silently pass the next test."""

    def forget():
        attention._pinned_reports.clear()
        attention._pinned_reports_dropped = False
        attention._plain_call_served.clear()

    forget()
    yield
    forget()


@pytest.fixture
def pin(monkeypatch, caplog, preserve_sdpa_priority_order):
    """Pin a kernel the way startup does: its order, then the report over torch's own function. What installing
    logs is the startup tests' business, so it is dropped here."""
    monkeypatch.setattr(F, "scaled_dot_product_attention", TORCH_SDPA)

    def _pin(kernel: SDPBackend) -> None:
        assert install_default_sdpa_priority(kernel)
        install_pinned_sdpa_kernel_report(kernel)
        caplog.clear()

    return _pin


def _reports(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]


def _priority() -> list[SDPBackend]:
    return [SDPBackend(backend) for backend in torch._C._get_sdp_priority_order()]


def _set_priority(order: list[SDPBackend]) -> None:
    torch._C._set_sdp_priority_order([int(backend) for backend in order])


# Run in a fresh interpreter. After each step of pinning a kernel at startup it records whether CUDA is in use: torch's
# own flag, or a primary context on any device, which C++ code can create without setting the flag. Prints the steps
# after which it was.
_PINNED_STARTUP_CUDA_PROBE = """
import torch

has_primary_context = getattr(torch._C, "_cuda_hasPrimaryContext", lambda index: False)


def cuda_in_use():
    contexts = [index for index in range(torch.cuda.device_count()) if has_primary_context(index)]
    return torch.cuda.is_initialized() or bool(contexts)


steps = [("import torch", cuda_in_use())]
from invokeai.backend.util.sdpa_scope import SDPA_KERNELS, install_default_sdpa_priority

steps.append(("import invokeai.backend.util.sdpa_scope", cuda_in_use()))
from invokeai.backend.util.attention import install_pinned_sdpa_kernel_report

steps.append(("import invokeai.backend.util.attention", cuda_in_use()))
torch_sdpa = torch.nn.functional.scaled_dot_product_attention
for name, kernel in SDPA_KERNELS.items():
    install_default_sdpa_priority(kernel)
    steps.append((f"{name}: the order", cuda_in_use()))
    # A fresh report each time: installing over an installed one is a no-op.
    torch.nn.functional.scaled_dot_product_attention = torch_sdpa
    install_pinned_sdpa_kernel_report(kernel)
    steps.append((f"{name}: the report and the build check", cuda_in_use()))
print([step for step, in_use in steps if in_use])
"""


def _qkv(*shape: int, dtype: torch.dtype = torch.float32, device: str = "cpu") -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(*shape, dtype=dtype, device=device)


class TestOnCpu:
    def test_each_node_type_and_serving_kernel_is_reported_once(self, pin, caplog):
        pin(SDPBackend.CUDNN_ATTENTION)
        plain = _qkv(1, 8, 32, 64)
        # 3-D inputs: CPU flash refuses them, so math serves the call.
        three_d = _qkv(8, 32, 64)

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            with attention_caller("flux_denoise"):
                out = F.scaled_dot_product_attention(plain, plain, plain)
                F.scaled_dot_product_attention(plain, plain, plain)
                F.scaled_dot_product_attention(three_d, three_d, three_d)
            with attention_caller("z_image_text_encoder"):
                F.scaled_dot_product_attention(plain, plain, plain)
                F.scaled_dot_product_attention(plain, plain, plain)

        assert _reports(caplog) == [
            f"attention_backend 'cudnn' is not possible for attention in flux_denoise: {CPU_RANKING}; "
            "it runs on FLASH_ATTENTION.",
            f"attention_backend 'cudnn' is not possible for attention in flux_denoise: {CPU_RANKING}; it runs on MATH.",
            f"attention_backend 'cudnn' is not possible for attention in z_image_text_encoder: {CPU_RANKING}; "
            "it runs on FLASH_ATTENTION.",
        ]
        assert torch.equal(out, TORCH_SDPA(plain, plain, plain)), "the call itself runs unchanged"

    def test_attention_outside_a_node_says_so(self, pin, caplog):
        pin(SDPBackend.EFFICIENT_ATTENTION)
        q = _qkv(1, 2, 8, 16)

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            F.scaled_dot_product_attention(q, q, q)

        assert _reports(caplog) == [
            f"attention_backend 'efficient' is not possible for attention outside a node: {CPU_RANKING}; "
            "it runs on FLASH_ATTENTION."
        ]

    def test_a_call_the_pinned_kernel_serves_is_not_reported(self, pin, caplog):
        """CPU flash serves a plain call, so pinning flash there has nothing to report."""
        pin(SDPBackend.FLASH_ATTENTION)
        q = _qkv(1, 2, 8, 16)

        with caplog.at_level(logging.WARNING, logger=LOGGER), attention_caller("flux_denoise"):
            F.scaled_dot_product_attention(q, q, q)

        assert _reports(caplog) == []

    def test_reports_stop_at_the_cap_and_say_so_once(self, pin, caplog, monkeypatch):
        monkeypatch.setattr(attention, "_MAX_PINNED_REPORTS", 2)
        pin(SDPBackend.CUDNN_ATTENTION)
        q = _qkv(1, 2, 8, 16)

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            for caller in ("a", "b", "c", "d"):
                with attention_caller(caller):
                    F.scaled_dot_product_attention(q, q, q)

        reports = _reports(caplog)
        assert [report.split(":")[0] for report in reports] == [
            "attention_backend 'cudnn' is not possible for attention in a",
            "attention_backend 'cudnn' is not possible for attention in b",
            "attention_backend 'cudnn'",
        ]
        assert reports[2].endswith("further ones are not logged.")
        assert len(attention._pinned_reports) == 2

    def test_a_failing_report_never_fails_the_call(self, pin, caplog, monkeypatch):
        pin(SDPBackend.CUDNN_ATTENTION)
        attention._warn_pinned_report_failed.cache_clear()
        monkeypatch.setattr(attention, "_pinned_kernel_bypass", lambda *args: 1 / 0)
        q = _qkv(1, 2, 8, 16)

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = F.scaled_dot_product_attention(q, q, q)
            F.scaled_dot_product_attention(q, q, q)

        assert torch.equal(out, TORCH_SDPA(q, q, q))
        assert [r.exc_info is not None for r in caplog.records if r.levelno == logging.WARNING] == [True], "said once"
        attention._warn_pinned_report_failed.cache_clear()

    def test_a_device_the_dispatcher_cannot_answer_for_is_not_asked(self, pin, caplog, monkeypatch):
        """On MPS the dispatcher query raises, at 79us a call. CPU stands in for such a device here."""
        pin(SDPBackend.CUDNN_ATTENTION)
        monkeypatch.setattr(attention, "_DISPATCHER_DEVICE_TYPES", frozenset({"cuda", "xpu"}))
        asked: list[str] = []
        monkeypatch.setattr(attention, "sdpa_backend_choice", lambda query, *args, **kwargs: asked.append("asked"))
        q = _qkv(1, 2, 8, 16)

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            out = F.scaled_dot_product_attention(q, q, q)

        assert asked == []
        assert _reports(caplog) == []
        assert torch.equal(out, TORCH_SDPA(q, q, q))

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Silicon")
    def test_attention_on_mps_is_not_asked_about(self, pin, caplog, monkeypatch):
        pin(SDPBackend.CUDNN_ATTENTION)
        asked: list[str] = []
        real = attention.sdpa_backend_choice
        monkeypatch.setattr(
            attention, "sdpa_backend_choice", lambda *args, **kwargs: asked.append("asked") or real(*args, **kwargs)
        )
        q = _qkv(1, 2, 8, 16, device="mps")

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            F.scaled_dot_product_attention(q, q, q)

        assert asked == []
        assert _reports(caplog) == []

    def test_installing_twice_wraps_once(self, pin):
        pin(SDPBackend.MATH)
        wrapped = F.scaled_dot_product_attention
        install_pinned_sdpa_kernel_report(SDPBackend.MATH)
        assert F.scaled_dot_product_attention is wrapped
        assert wrapped.__wrapped__ is TORCH_SDPA


class TestAtStartup:
    @pytest.fixture
    def startup(self, monkeypatch, preserve_sdpa_priority_order):
        from invokeai.app.util.startup_utils import apply_monkeypatches

        # Pin the entry points apply_monkeypatches rebinds so monkeypatch restores them afterwards.
        monkeypatch.setattr(F, "scaled_dot_product_attention", TORCH_SDPA)
        monkeypatch.setattr(torch.cuda, "empty_cache", torch.cuda.empty_cache)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", torch.cuda.is_current_stream_capturing)
        return apply_monkeypatches

    def test_every_kernel_value_of_the_setting_is_covered_here(self):
        """The config accepts these values and startup maps them to kernels; a value it does not map would quietly
        run as `auto`. The next test pins each one."""
        assert set(get_args(ATTENTION_BACKEND)) == {"auto", "sage", *PINNED_BY_VALUE}

    @pytest.mark.parametrize("value", sorted(PINNED_BY_VALUE))
    def test_a_pinned_kernel_installs_its_order_and_the_report(self, startup, sdpa_orders, value):
        _set_priority(sdpa_orders.torch_own)

        startup(value)

        assert _priority() == sdpa_orders.pinned[PINNED_BY_VALUE[value]]
        assert getattr(F.scaled_dot_product_attention, attention._PINNED_REPORT_SENTINEL, False)

    def test_the_startup_line_names_the_order_with_each_kernel_once(self, startup, caplog):
        """torch 2.7's own order repeats MATH in its fifth slot, which the four ranked slots leave in place. Set
        here on any torch, so the line meets the repeat wherever the suite runs."""
        _set_priority(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.MATH,
            ]
        )

        with caplog.at_level(logging.INFO, logger=LOGGER):
            startup("efficient")

        assert (
            "attention_backend 'efficient': PyTorch SDPA kernels ranked EFFICIENT_ATTENTION > FLASH_ATTENTION > "
            "CUDNN_ATTENTION > MATH. Attention the memory-efficient kernel cannot serve runs on the next one"
            in caplog.text
        )

    @pytest.mark.parametrize("backend", ["auto", "sage"])
    def test_no_pinned_kernel_installs_no_report(self, startup, backend, sdpa_orders):
        _set_priority(sdpa_orders.torch_own)

        startup(backend)

        assert not getattr(F.scaled_dot_product_attention, attention._PINNED_REPORT_SENTINEL, False)
        assert _priority() == sdpa_orders.installed

    def test_a_torch_that_cannot_rank_installs_no_report(self, startup, monkeypatch):
        """The report reads 'not first' as 'not possible'; without the ranking every call would be reported."""
        with monkeypatch.context() as patch:
            patch.delattr(torch._C, "_set_sdp_priority_order")
            startup("math")

        assert not getattr(F.scaled_dot_product_attention, attention._PINNED_REPORT_SENTINEL, False)

    @pytest.mark.parametrize(
        ("backend", "hip", "missing"),
        [
            ("flash", None, "this PyTorch build has no flash-attention kernel for GPUs"),
            ("cudnn", "7.2", "ROCm builds of PyTorch have no cuDNN attention kernel"),
        ],
    )
    def test_a_build_known_to_lack_the_kernel_is_warned_about(
        self, startup, monkeypatch, caplog, backend, hip, missing
    ):
        monkeypatch.setattr(torch.version, "cuda", None if hip else "13.0")
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: False)

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            startup(backend)

        assert _reports(caplog) == [
            f"attention_backend '{backend}': {missing}, so GPU attention runs on the next kernel in that order. "
            "`attention_backend: auto` is the default."
        ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="only a machine with a CUDA device can create a context")
    def test_pinning_a_kernel_at_startup_creates_no_cuda_context(self):
        """A context parks VRAM for the process lifetime on a server that may stay idle. Checked for what pinning adds
        to startup -- the modules holding it, the order, the report and the build check -- and not for the whole of
        `apply_monkeypatches`, whose first step, importing `invokeai.backend.util.hotfixes`, initializes CUDA by itself.
        In a fresh interpreter, because whether CUDA is initialized is process state an earlier test may have
        changed."""
        result = subprocess.run(
            [sys.executable, "-c", _PINNED_STARTUP_CUDA_PROBE], capture_output=True, text=True, timeout=300
        )

        assert result.returncode == 0, result.stderr
        initialized_by = result.stdout.strip().splitlines()[-1]
        assert initialized_by == "[]", f"CUDA was initialized by: {initialized_by}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA reasons need a CUDA or ROCm GPU")
class TestOnAGpu:
    """Measured on an RTX 4090 under Windows; any CUDA or ROCm GPU answers them."""

    @staticmethod
    def _first_accepting(params, kernels: list[SDPBackend]) -> SDPBackend:
        checks = {
            SDPBackend.FLASH_ATTENTION: torch.backends.cuda.can_use_flash_attention,
            SDPBackend.CUDNN_ATTENTION: torch.backends.cuda.can_use_cudnn_attention,
            SDPBackend.EFFICIENT_ATTENTION: torch.backends.cuda.can_use_efficient_attention,
        }
        return next((kernel for kernel in kernels if checks[kernel](params)), SDPBackend.MATH)

    def test_grouped_query_attention_pinned_to_efficient_names_the_reason_and_the_kernel(self, pin, caplog, capfd):
        """A grouped-query call that reaches SDPA unexpanded: memory-efficient attention refuses it, and the next
        kernel takes it. (Krea-2 expands its heads for a pinned kernel that needs that; see its processor.)"""
        q = _qkv(1, 8, 256, 64, dtype=torch.bfloat16, device="cuda")
        kv = _qkv(1, 2, 256, 64, dtype=torch.bfloat16, device="cuda")
        params = torch.backends.cuda.SDPAParams(q, kv, kv, None, 0.0, False, True)
        if torch.backends.cuda.can_use_efficient_attention(params):
            pytest.skip("this build's memory-efficient kernel serves grouped-query attention")
        served = self._first_accepting(params, [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION])
        pin(SDPBackend.EFFICIENT_ATTENTION)
        capfd.readouterr()

        with caplog.at_level(logging.WARNING, logger=LOGGER), attention_caller("grouped_query_node"):
            F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
            first = capfd.readouterr().err
            F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
            second = capfd.readouterr().err

        assert _reports(caplog) == [
            "attention_backend 'efficient' is not possible for attention in grouped_query_node: the memory-efficient "
            "kernel cannot serve this call (bfloat16, head_dim 64, no mask, 8 query / 2 key-value heads, enable_gqa); "
            f"it runs on {served.name}. PyTorch's own check follows on stderr."
        ]
        assert first != "", "torch's debug check printed its reason"
        assert second == "", "and only for the first call of its kind"

    @pytest.mark.skipif(
        torch.backends.cuda.is_flash_attention_available(), reason="this build has a flash-attention kernel"
    )
    def test_flash_pinned_on_a_build_without_it_says_so(self, pin, caplog):
        q = _qkv(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, False, False)
        served = self._first_accepting(params, [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
        pin(SDPBackend.FLASH_ATTENTION)

        with caplog.at_level(logging.WARNING, logger=LOGGER), attention_caller("flux_denoise"):
            F.scaled_dot_product_attention(q, q, q)

        assert _reports(caplog) == [
            f"attention_backend 'flash' is not possible for attention in flux_denoise: this PyTorch build or GPU "
            f"({q.device}) has no flash kernel; it runs on {served.name}."
        ]

    def test_a_pinned_kernel_an_exclusive_window_switched_off_says_so(self, pin, caplog):
        """`INVOKE_KREA2_SDPA_BACKEND=efficient` with cuDNN pinned: the window wins, and the report blames it."""
        q = _qkv(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        if not torch.backends.cuda.can_use_efficient_attention(
            torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, False, False)
        ):
            pytest.skip("this GPU has no memory-efficient kernel for the call")
        pin(SDPBackend.CUDNN_ATTENTION)

        with caplog.at_level(logging.WARNING, logger=LOGGER), attention_caller("krea2_denoise"):
            with sdpa_policy([SDPBackend.EFFICIENT_ATTENTION]):
                F.scaled_dot_product_attention(q, q, q)

        assert _reports(caplog) == [
            "attention_backend 'cudnn' is not possible for attention in krea2_denoise: the cuDNN kernel is switched "
            "off for this call (by an exclusive window, such as INVOKE_KREA2_SDPA_BACKEND's); it runs on "
            "EFFICIENT_ATTENTION."
        ]

    def test_pinned_math_is_priced_in_working_memory(self, preserve_sdpa_priority_order):
        """The estimators ask the dispatcher, so the score matrix math builds is reserved without being told."""
        device = torch.device("cuda", torch.cuda.current_device())
        args = {"device": device, "dtype": torch.float16, "num_heads": 24, "head_dim": 128, "seq_len": 4096}
        install_default_sdpa_priority()
        if sdpa_score_matrix_bytes(**args) != 0:
            pytest.skip("no fused kernel serves the plain call on this GPU, so math is priced either way")

        install_default_sdpa_priority(SDPBackend.MATH)

        assert sdpa_score_matrix_bytes(**args) == 24 * 4096 * 4096 * 17


class _FailsOnceInCudnn:
    """Stands in for torch's SDPA: the first call fails the way a cuDNN plan does at run time, later ones run torch's
    own function."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False
    ):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("cuDNN Frontend error: no valid execution plan")
        return TORCH_SDPA(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)


class TestWithTheOtherSdpaWrappers:
    """The report sits outside the ROCm guard and inside the cuDNN fallback, as `apply_monkeypatches` installs them.
    Each order is visible only in what gets reported, so a reordered startup fails here."""

    @pytest.mark.skipif(
        torch.version.cuda is None or torch.version.hip is not None or not torch.cuda.is_available(),
        reason="the cuDNN fallback exists on CUDA builds, and the reason needs a GPU",
    )
    def test_the_retry_after_a_cudnn_failure_is_reported_with_the_demotion(
        self, monkeypatch, caplog, preserve_sdpa_priority_order
    ):
        """Inside the fallback, the report sees the retry, which runs on the next kernel. The first call's failure
        is a double; the order, the demotion and the dispatcher are real. Measured on an RTX 4090 under Windows."""
        q = _qkv(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        if not torch.backends.cuda.can_use_cudnn_attention(
            torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, False, False)
        ):
            pytest.skip("this GPU has no cuDNN kernel for the call")
        kernel = _FailsOnceInCudnn()
        monkeypatch.setattr(F, "scaled_dot_product_attention", kernel)
        monkeypatch.setattr(attention, "_cudnn_demotions", 0)
        install_default_sdpa_priority(SDPBackend.CUDNN_ATTENTION)
        install_pinned_sdpa_kernel_report(SDPBackend.CUDNN_ATTENTION)
        install_cudnn_sdpa_fallback()
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger=LOGGER), attention_caller("flux_denoise"):
            F.scaled_dot_product_attention(q, q, q)
        served = sdpa_backend_choice(q, q, q)

        reports = _reports(caplog)
        assert kernel.calls == 2, "the call and its retry"
        assert len(reports) == 2
        assert reports[0].startswith("cuDNN attention failed on")
        assert reports[1] == (
            "attention_backend 'cudnn' is not possible for attention in flux_denoise: cuDNN attention failed earlier "
            f"in this run and now ranks behind math; it runs on {served.name}."
        )

    @pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(),
        reason="the wide-head guard exists on ROCm builds, and the reason needs a GPU",
    )
    def test_a_wide_head_the_rocm_guard_sends_to_math_says_so(self, monkeypatch, caplog, preserve_sdpa_priority_order):
        """Outside the guard, the report sees the heads it sends to math whatever the dispatcher would pick. To be
        run on an RX 9060 XT."""
        monkeypatch.setattr(F, "scaled_dot_product_attention", TORCH_SDPA)
        monkeypatch.setattr(attention, "ROCM_FUSED_SDPA_MAX_HEAD_DIM", 256)
        install_rocm_sdpa_guard()
        install_default_sdpa_priority(SDPBackend.EFFICIENT_ATTENTION)
        install_pinned_sdpa_kernel_report(SDPBackend.EFFICIENT_ATTENTION)
        caplog.clear()
        q = _qkv(1, 1, 64, 512, dtype=torch.float16, device="cuda")

        with caplog.at_level(logging.WARNING, logger=LOGGER), attention_caller("flux_vae_decode"):
            F.scaled_dot_product_attention(q, q, q)

        assert _reports(caplog) == [
            "attention_backend 'efficient' is not possible for attention in flux_vae_decode: on ROCm, heads wider "
            "than 256 run on math (INVOKE_ROCM_FUSED_SDPA_MAX_HEAD_DIM sets the width); it runs on MATH."
        ]
