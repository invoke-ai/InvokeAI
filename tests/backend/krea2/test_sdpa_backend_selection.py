"""Which SDPA backend selection the Krea-2 attention processors run under, and the opt-in override.

By default Krea-2 opens no SDPA window: its attention runs under InvokeAI's process-wide priority order
(`util.sdpa_scope`), like every other model's. The override exists for benchmarking and support: a
named backend is pinned exclusively, so a run that completes proves which kernel served it, and
`default` names the process-wide order so a benchmarking run can state which policy it used.
"""

import contextlib
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.models.transformers.transformer_krea2 import Krea2Attention
from torch.nn.attention import SDPBackend, sdpa_kernel

import invokeai.backend.krea2.attention as krea2_attention
from invokeai.backend.krea2.attention import (
    KREA2_SDPA_BACKEND_ENV_VAR,
    Krea2MemoryEfficientAttnProcessor,
    Krea2RegionalPromptingState,
    build_krea2_attention_processors,
    resolve_krea2_sdpa_backends,
)


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    """Decide the override here, never inherit it from the shell.

    The docs tell users -- and benchmarking runs -- to export this variable, so the suite is run in
    exactly the shell where it is set. A test that asserts what the default is has to own that
    default; one that wants an override sets it explicitly.
    """
    monkeypatch.delenv(KREA2_SDPA_BACKEND_ENV_VAR, raising=False)


class TestTheDefault:
    def test_unset_pins_no_backend_and_records_no_override(self):
        # `override is None` is what keeps the benchmark instrumentation, and its synchronize()
        # calls, entirely off the default path -- and what lets SageAttention serve Krea-2.
        choice = resolve_krea2_sdpa_backends(raw_override=None)
        assert choice.exclusive is None
        assert choice.override is None

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_a_blank_value_is_the_default_not_an_error(self, blank):
        choice = resolve_krea2_sdpa_backends(raw_override=blank)
        assert choice.exclusive is None
        assert choice.override is None

    def test_default_names_the_process_order_and_pins_nothing(self):
        """The value a benchmarking run exports to measure the shipped behaviour. It must run the
        same kernels as unset -- no window -- while still counting as an override."""
        choice = resolve_krea2_sdpa_backends(raw_override="default")
        assert choice.exclusive is None
        assert choice.override == "default"


class TestTheOverride:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("cudnn", SDPBackend.CUDNN_ATTENTION),
            ("efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("flash", SDPBackend.FLASH_ATTENTION),
            ("math", SDPBackend.MATH),
        ],
    )
    def test_a_named_backend_is_pinned_exclusively(self, value, expected):
        """The point of the exclusive modes: a run that completes proves that kernel was used,
        because an unavailable backend raises visibly instead of degrading to math unnoticed."""
        choice = resolve_krea2_sdpa_backends(raw_override=value)
        assert choice.exclusive is expected
        assert choice.override == value

    @pytest.mark.parametrize("value", ["  CUDNN  ", "Default", "EFFICIENT"])
    def test_values_are_stripped_and_lowercased(self, value):
        assert resolve_krea2_sdpa_backends(raw_override=value).override == value.strip().lower()

    def test_an_unknown_value_raises_and_names_the_valid_ones(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_krea2_sdpa_backends(raw_override="cudnn-attention")
        message = str(excinfo.value)
        assert KREA2_SDPA_BACKEND_ENV_VAR in message
        for valid in ("cudnn", "efficient", "flash", "math", "default"):
            assert valid in message

    def test_the_environment_is_read_when_no_value_is_passed(self, monkeypatch):
        monkeypatch.setenv(KREA2_SDPA_BACKEND_ENV_VAR, "math")
        assert resolve_krea2_sdpa_backends().exclusive is SDPBackend.MATH
        monkeypatch.delenv(KREA2_SDPA_BACKEND_ENV_VAR)
        assert resolve_krea2_sdpa_backends().override is None


class TestProcessorsCarryTheChoice:
    def test_every_processor_gets_the_same_resolved_choice(self, monkeypatch):
        """Resolved once per generation and handed down -- not re-read per attention call, of which
        there are dozens per step."""
        monkeypatch.setenv(KREA2_SDPA_BACKEND_ENV_VAR, "cudnn")
        transformer = MagicMock()
        transformer.attn_processors = {
            "transformer_blocks.0.attn.processor": object(),
            "transformer_blocks.1.attn.processor": object(),
            "single_transformer_blocks.0.attn.processor": object(),
        }

        processors = build_krea2_attention_processors(transformer, Krea2RegionalPromptingState())

        choices = {id(p.sdpa_backends) for p in processors.values()}
        assert len(choices) == 1
        assert next(iter(processors.values())).sdpa_backends.exclusive is SDPBackend.CUDNN_ATTENTION

    def test_a_standalone_processor_resolves_for_itself(self, monkeypatch):
        # Constructed directly in tests and by custom code; it must not depend on the builder.
        monkeypatch.setenv(KREA2_SDPA_BACKEND_ENV_VAR, "math")
        assert Krea2MemoryEfficientAttnProcessor().sdpa_backends.exclusive is SDPBackend.MATH


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise the SDPA dispatcher")
class TestAnExclusiveOverrideHasNoFallback:
    def test_the_call_raises_where_the_pinned_backend_cannot_serve_it(self):
        """The likely explanation of the sm_86 report: that measurement was taken with
        `sdpa_kernel([backend])` -- exclusive, no fallback -- which is exactly the mode that raises
        `No available kernel` when the backend is unusable. The process-wide order skips it instead
        (`test_sdpa_scope`)."""
        # fp32 is refused by the fused kernels, which is a portable way to make cuDNN unusable on a
        # card where it otherwise works -- i.e. to stand in for the sm_86 report.
        t = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.float32)
        params = torch.backends.cuda.SDPAParams(t, t, t, None, 0.0, False, False)
        if torch.backends.cuda.can_use_cudnn_attention(params):
            pytest.skip("This build serves fp32 with cuDNN, so it cannot stand in for an unusable backend")

        choice = resolve_krea2_sdpa_backends(raw_override="cudnn")
        with pytest.raises(RuntimeError, match="No available kernel"):
            with sdpa_kernel([choice.exclusive]):
                torch.nn.functional.scaled_dot_product_attention(t, t, t)


class TestTheDispatchSite:
    """That the resolved choice actually reaches the SDPA window -- or that no window is opened.

    Everything above asserts what `resolve_krea2_sdpa_backends` *returns*. None of it survives the
    processor ignoring that value, and spying on the window is the only place that is observable.
    """

    def _run_with_spy(self, monkeypatch, processor):
        windows: list[list[SDPBackend]] = []

        def spy(backends):
            windows.append(list(backends))
            return contextlib.nullcontext()

        monkeypatch.setattr(krea2_attention, "sdpa_policy", spy)

        attn = Krea2Attention(hidden_size=256, num_heads=8, num_kv_heads=2, eps=1e-5).eval()
        attn.set_processor(processor)
        with torch.no_grad():
            out = attn(torch.randn(1, 24, attn.hidden_size), attention_mask=None, image_rotary_emb=None)
        assert out.shape == (1, 24, attn.hidden_size)
        return windows

    @pytest.mark.parametrize("raw", [None, "default"])
    def test_the_default_opens_no_window(self, monkeypatch, raw):
        """A window here would replace the process-wide order for the block -- and, being process-global,
        for every other session's attention while it is open."""
        processor = Krea2MemoryEfficientAttnProcessor(sdpa_backends=resolve_krea2_sdpa_backends(raw_override=raw))
        assert self._run_with_spy(monkeypatch, processor) == []

    def test_an_exclusive_override_pins_its_backend(self, monkeypatch):
        exclusive = resolve_krea2_sdpa_backends(raw_override="math")
        windows = self._run_with_spy(monkeypatch, Krea2MemoryEfficientAttnProcessor(sdpa_backends=exclusive))

        assert windows == [[SDPBackend.MATH]]
