"""Which SDPA backends the Krea-2 attention processors run under, and the opt-in override.

The ranking is a measurement, not a preference. On the real Krea-2 attention shape, per call: on a
Windows 4090 (no flash in the build) cuDNN runs at 3.72ms against efficient's 5.92ms; on a Linux
30-series card flash runs at 19.74ms, cuDNN at 21.27ms, efficient at 31.45ms. So flash leads where it
exists and cuDNN takes the blocks it cannot serve -- the masked ones, and every block on a build
without flash.

These tests pin what makes that reachable at all: the order, and `set_priority=True`. Without the
latter the list only *permits* backends and torch picks by its own order, so the ranking would be a
silent no-op.
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

    The PR tells users -- and its own A/B workflow -- to export this variable, so the suite is run
    in exactly the shell where it is set. Reading it made 11 of these 24 tests fail there, including
    the fallback test the design rests on. A test that asserts what the default ranking is has to
    own that default; one that wants an override sets it explicitly.
    """
    monkeypatch.delenv(KREA2_SDPA_BACKEND_ENV_VAR, raising=False)


class TestTheDefaultRanking:
    def test_flash_leads_and_cudnn_follows(self):
        """Flash is the fastest kernel where the build has it -- and is already what runs today,
        because torch's own order puts it above efficient. Ranking cuDNN over it would be a small
        regression on every flash-capable build."""
        backends = resolve_krea2_sdpa_backends(raw_override=None).backends
        assert backends[0] is SDPBackend.FLASH_ATTENTION
        assert backends[1] is SDPBackend.CUDNN_ATTENTION

    def test_cudnn_outranks_efficient(self):
        """This is where the win actually comes from: flash refuses the additive mask the
        regional-prompting blocks pass, so on those blocks it is skipped and cuDNN takes over --
        1.6x-2.0x over efficient. On a build without flash, that is every block."""
        backends = resolve_krea2_sdpa_backends(raw_override=None).backends
        assert backends.index(SDPBackend.CUDNN_ATTENTION) < backends.index(SDPBackend.EFFICIENT_ATTENTION)

    def test_math_is_last(self):
        # The unfused fallback: correct everywhere, and ~35x slower at ~35x the memory.
        assert resolve_krea2_sdpa_backends(raw_override=None).backends[-1] is SDPBackend.MATH

    def test_priority_is_set(self):
        """Load-bearing: without set_priority the list only *permits* backends and torch picks by
        its own order, in which cuDNN ranks last and would never be chosen. The ranking would be a
        no-op."""
        assert resolve_krea2_sdpa_backends(raw_override=None).set_priority is True

    def test_the_fallbacks_are_all_still_there(self):
        # A ranked list, never an exclusive backend: an unavailable entry is skipped by the
        # dispatcher, so the list degrades to today's behaviour on its own.
        assert set(resolve_krea2_sdpa_backends(raw_override=None).backends) == {
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.MATH,
        }

    def test_flash_is_kept_even_where_a_probe_would_call_it_dead(self):
        """A sibling plan proposed dropping FLASH where a probe shows it absent. That would be wrong
        twice over: on ROCm flash is available and cuDNN is not, and on Linux CUDA it is the fastest
        kernel of the four. A dead entry in a ranked list costs nothing; a missing one costs a
        platform."""
        assert SDPBackend.FLASH_ATTENTION in resolve_krea2_sdpa_backends(raw_override=None).backends

    def test_no_override_is_recorded_by_default(self):
        # `override is None` is what keeps the benchmark instrumentation, and its synchronize()
        # calls, entirely off the default path.
        assert resolve_krea2_sdpa_backends(raw_override=None).override is None

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_a_blank_value_is_the_default_not_an_error(self, blank):
        assert resolve_krea2_sdpa_backends(raw_override=blank).override is None


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
    def test_a_named_backend_is_exclusive_with_no_fallback(self, value, expected):
        """The point of the exclusive modes: a run that completes proves that kernel was used,
        because an unavailable backend raises visibly instead of degrading to math unnoticed."""
        choice = resolve_krea2_sdpa_backends(raw_override=value)
        assert choice.backends == (expected,)
        assert choice.set_priority is False
        assert choice.override == value

    def test_default_names_the_shipped_list_without_changing_it(self):
        """The value a benchmarking run exports to say "no override"; it must be the shipped list.

        It used to be spelled `priority-cudnn`, which asserted an ordering it did not produce -- the
        list is flash-first, because flash wins wherever a build has it."""
        explicit = resolve_krea2_sdpa_backends(raw_override="default")
        default = resolve_krea2_sdpa_backends(raw_override=None)
        assert explicit.backends == default.backends
        assert explicit.set_priority == default.set_priority
        assert explicit.override == "default"
        assert explicit.backends[0] is SDPBackend.FLASH_ATTENTION

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
        assert resolve_krea2_sdpa_backends().backends == (SDPBackend.MATH,)
        monkeypatch.delenv(KREA2_SDPA_BACKEND_ENV_VAR)
        assert resolve_krea2_sdpa_backends().override is None


class TestProcessorsCarryTheChoice:
    def test_every_processor_gets_the_same_resolved_choice(self):
        """Resolved once per generation and handed down -- not re-read per attention call, of which
        there are dozens per step."""
        transformer = MagicMock()
        transformer.attn_processors = {
            "transformer_blocks.0.attn.processor": object(),
            "transformer_blocks.1.attn.processor": object(),
            "single_transformer_blocks.0.attn.processor": object(),
        }

        processors = build_krea2_attention_processors(transformer, Krea2RegionalPromptingState())

        choices = {id(p.sdpa_backends) for p in processors.values()}
        assert len(choices) == 1
        assert next(iter(processors.values())).sdpa_backends.backends[0] is SDPBackend.FLASH_ATTENTION

    def test_a_standalone_processor_resolves_for_itself(self):
        # Constructed directly in tests and by custom code; it must not depend on the builder.
        assert Krea2MemoryEfficientAttnProcessor().sdpa_backends.backends[0] is SDPBackend.FLASH_ATTENTION


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise the SDPA dispatcher")
class TestTheFallbackIsReal:
    """The whole design rests on one dispatcher property: an unusable backend in a *ranked* list is
    skipped, while an *exclusively* selected one raises.

    This is what makes cuDNN-first safe on hardware where cuDNN cannot serve the call, and it is
    worth pinning: if a future torch made a ranked list raise instead, the ranking would turn every
    such device from "today's behaviour" into a failed generation, silently as far as our tests go.
    """

    @staticmethod
    def _unservable_by_cudnn():
        # fp32 is refused by the fused kernels, which is a portable way to make cuDNN unusable on a
        # card where it otherwise works -- i.e. to stand in for the sm_86 report.
        t = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.float32)
        params = torch.backends.cuda.SDPAParams(t, t, t, None, 0.0, False, False)
        if torch.backends.cuda.can_use_cudnn_attention(params):
            pytest.skip("This build serves fp32 with cuDNN, so it cannot stand in for an unusable backend")
        return t

    def test_a_ranked_list_completes_where_cudnn_cannot_serve(self):
        t = self._unservable_by_cudnn()
        choice = resolve_krea2_sdpa_backends(raw_override=None)
        with sdpa_kernel(list(choice.backends), set_priority=choice.set_priority):
            out = torch.nn.functional.scaled_dot_product_attention(t, t, t)
        assert torch.isfinite(out).all()

    def test_the_same_call_raises_when_cudnn_is_selected_exclusively(self):
        """The counterpart, and the likely explanation of the sm_86 report: that measurement was
        taken with `sdpa_kernel([backend])` -- exclusive, no fallback -- which is exactly the mode
        that raises `No available kernel` when the backend is unusable."""
        t = self._unservable_by_cudnn()
        choice = resolve_krea2_sdpa_backends(raw_override="cudnn")
        with pytest.raises(RuntimeError, match="No available kernel"):
            with sdpa_kernel(list(choice.backends), set_priority=choice.set_priority):
                torch.nn.functional.scaled_dot_product_attention(t, t, t)


class TestTheDispatchSite:
    """That the resolved choice actually reaches the SDPA window.

    Everything above asserts what `resolve_krea2_sdpa_backends` *returns*. None of it survives the
    processor ignoring that value: drop `set_priority=` from the call, or read the module constant
    instead of `self.sdpa_backends`, and Krea-2 falls back to torch's own order on every block --
    the entire measured win -- with all 31 tests still green.

    Spying on the context manager is the only place the effect is observable: torch exposes no way
    to read back the priority order a window installed.
    """

    def _run_with_spy(self, monkeypatch, processor):
        seen: dict[str, object] = {}

        def spy(backends, set_priority=False):
            seen["backends"] = list(backends)
            seen["set_priority"] = set_priority
            return contextlib.nullcontext()

        monkeypatch.setattr(krea2_attention, "sdpa_policy", spy)

        attn = Krea2Attention(hidden_size=256, num_heads=8, num_kv_heads=2, eps=1e-5).eval()
        attn.set_processor(processor)
        with torch.no_grad():
            attn(torch.randn(1, 24, attn.hidden_size), attention_mask=None, image_rotary_emb=None)
        return seen

    def test_the_default_ranking_reaches_the_dispatcher_with_priority_on(self, monkeypatch):
        seen = self._run_with_spy(monkeypatch, Krea2MemoryEfficientAttnProcessor())

        assert seen["backends"] == list(resolve_krea2_sdpa_backends(raw_override=None).backends)
        # Without this the list only *permits* backends and torch picks by its own order, which is
        # what the ranking exists to override.
        assert seen["set_priority"] is True

    def test_the_processors_own_choice_is_used_not_the_module_default(self, monkeypatch):
        """An exclusive override differs from the shipped list in both fields, so a processor that
        reads the module constant instead of its own attribute fails here on either one."""
        exclusive = resolve_krea2_sdpa_backends(raw_override="math")
        seen = self._run_with_spy(monkeypatch, Krea2MemoryEfficientAttnProcessor(sdpa_backends=exclusive))

        assert seen["backends"] == [SDPBackend.MATH]
        assert seen["set_priority"] is False
