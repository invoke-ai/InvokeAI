"""The ROCm head-dim guard for `F.scaled_dot_product_attention`.

The fused ROCm kernels are wrong above head_dim 256 (see `ROCM_FUSED_SDPA_MAX_HEAD_DIM`). The guard
must send exactly those calls to the math kernel, leave every other call on the original function
with its arguments intact, install once, and make the working-memory estimator budget the score
matrix the math kernel materializes.
"""

import pytest
import torch
import torch.nn.functional as F

import invokeai.backend.util.attention as attention
from invokeai.backend.util.attention import (
    ROCM_FUSED_SDPA_MAX_HEAD_DIM,
    install_rocm_sdpa_head_dim_guard,
    rocm_sdpa_uses_math_kernel,
    sdpa_score_matrix_bytes,
)

WIDE = ROCM_FUSED_SDPA_MAX_HEAD_DIM + 1
NARROW = ROCM_FUSED_SDPA_MAX_HEAD_DIM


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1) @ v


class TestPredicate:
    """The one rule the guard and the estimators share, tested against real device objects."""

    def test_only_wide_heads_on_a_rocm_cuda_device(self, monkeypatch):
        monkeypatch.setattr(attention, "_IS_ROCM", True)
        assert rocm_sdpa_uses_math_kernel("cuda", WIDE)
        assert not rocm_sdpa_uses_math_kernel("cuda", NARROW), "the threshold itself is trusted"
        assert not rocm_sdpa_uses_math_kernel("cpu", WIDE)
        assert not rocm_sdpa_uses_math_kernel("mps", WIDE)

    def test_never_off_rocm(self, monkeypatch):
        monkeypatch.setattr(attention, "_IS_ROCM", False)
        assert not rocm_sdpa_uses_math_kernel("cuda", WIDE)

    def test_threshold_override_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(attention._ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV, "100000")
        assert attention._read_rocm_fused_sdpa_max_head_dim() == 100000
        monkeypatch.setenv(attention._ROCM_FUSED_SDPA_MAX_HEAD_DIM_ENV, "wide")
        assert attention._read_rocm_fused_sdpa_max_head_dim() == attention._ROCM_FUSED_SDPA_MAX_HEAD_DIM_DEFAULT


def _install_as_if_rocm(monkeypatch) -> list[tuple]:
    """Install the guard on a CPU build, treating CPU tensors as the ROCm device, and record the
    full argument tuple of every call that reaches the original function."""
    calls: list[tuple] = []
    original = F.scaled_dot_product_attention

    def spy(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        calls.append((query.shape[-1], attn_mask, dropout_p, is_causal, scale, enable_gqa))
        return original(query, key, value, attn_mask, dropout_p, is_causal, scale=scale, enable_gqa=enable_gqa)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    monkeypatch.setattr(attention, "_IS_ROCM", True)
    # CPU tensors stand in for the ROCm device; the real predicate is covered by TestPredicate.
    monkeypatch.setattr(
        attention, "rocm_sdpa_uses_math_kernel", lambda device_type, head_dim: head_dim > ROCM_FUSED_SDPA_MAX_HEAD_DIM
    )
    install_rocm_sdpa_head_dim_guard()
    return calls


class TestGuard:
    def test_wide_heads_take_the_math_kernel_and_match_the_reference(self, monkeypatch):
        calls = _install_as_if_rocm(monkeypatch)
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 1, 64, ROCM_FUSED_SDPA_MAX_HEAD_DIM * 2) for _ in range(3))

        out = F.scaled_dot_product_attention(q, k, v)

        assert calls == [], "a wide head must not reach the fused entry point"
        torch.testing.assert_close(out, _reference(q, k, v), atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"attn_mask": torch.rand(16, 16) > 0.3, "scale": 0.5, "enable_gqa": True},
            {"attn_mask": torch.randn(1, 1, 16, 16), "enable_gqa": True},
            {"attn_mask": torch.randn(16, 16).to(torch.bfloat16), "enable_gqa": True},
            {"is_causal": True, "enable_gqa": True},
        ],
        ids=["bool-mask+scale", "float-broadcast-mask", "bf16-mask", "causal"],
    )
    def test_wide_heads_keep_torchs_semantics_for_masks_scale_causal_and_gqa(self, monkeypatch, kwargs):
        _install_as_if_rocm(monkeypatch)
        original = F.scaled_dot_product_attention.__wrapped__
        torch.manual_seed(1)
        q = torch.randn(1, 4, 16, WIDE)
        k, v = (torch.randn(1, 2, 16, WIDE) for _ in range(2))
        mask = kwargs.get("attn_mask")
        if mask is not None and mask.dtype == torch.bfloat16:
            q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v, **kwargs)

        # torch's own (CPU, unguarded) semantics are the oracle.
        expected = original(q, k, v, **kwargs)
        torch.testing.assert_close(out, expected, atol=1e-2 if out.dtype == torch.bfloat16 else 1e-5, rtol=1e-2)

    def test_narrow_heads_pass_through_with_every_argument(self, monkeypatch):
        calls = _install_as_if_rocm(monkeypatch)
        torch.manual_seed(2)
        q, k, v = (torch.randn(1, 2, 8, NARROW) for _ in range(3))
        mask = torch.ones(8, 8, dtype=torch.bool)

        F.scaled_dot_product_attention(q, k, v, mask, 0.0, True, scale=0.25)

        assert calls == [(NARROW, mask, 0.0, True, 0.25, False)]

    def test_wide_heads_refuse_arguments_the_math_call_cannot_forward(self, monkeypatch):
        _install_as_if_rocm(monkeypatch)
        q = torch.randn(1, 1, 8, WIDE)
        with pytest.raises(TypeError, match="does not forward"):
            F.scaled_dot_product_attention(q, q, q, some_future_flag=True)

    def test_install_is_idempotent_and_inert_off_rocm(self, monkeypatch):
        _install_as_if_rocm(monkeypatch)
        guarded = F.scaled_dot_product_attention
        install_rocm_sdpa_head_dim_guard()
        assert F.scaled_dot_product_attention is guarded, "a second install must not wrap the wrapper"

        monkeypatch.setattr(F, "scaled_dot_product_attention", guarded.__wrapped__)
        monkeypatch.setattr(attention, "_IS_ROCM", False)
        install_rocm_sdpa_head_dim_guard()
        assert F.scaled_dot_product_attention is guarded.__wrapped__, "no guard on a non-ROCm build"

    def test_install_backs_off_when_the_math_op_rejects_the_probe(self, monkeypatch, caplog):
        """A torch whose aten op changed must fail at startup, visibly, not inside a generation."""
        original = F.scaled_dot_product_attention
        monkeypatch.setattr(attention, "_IS_ROCM", True)
        monkeypatch.setattr(attention, "_math_sdpa", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no such op")))
        caplog.set_level("WARNING", logger=attention.__name__)

        install_rocm_sdpa_head_dim_guard()

        assert F.scaled_dot_product_attention is original
        assert any("NOT installed" in r.message for r in caplog.records)

    def test_apply_monkeypatches_installs_the_guard(self, monkeypatch):
        from invokeai.app.util.startup_utils import apply_monkeypatches

        original = F.scaled_dot_product_attention
        # Pin both entry points apply_monkeypatches rebinds so monkeypatch restores them afterwards.
        monkeypatch.setattr(F, "scaled_dot_product_attention", original)
        monkeypatch.setattr(torch.cuda, "empty_cache", torch.cuda.empty_cache)
        monkeypatch.setattr(attention, "_IS_ROCM", True)

        apply_monkeypatches()

        assert getattr(F.scaled_dot_product_attention, attention._ROCM_SDPA_GUARD_SENTINEL, False)


class TestEstimator:
    def test_wide_heads_on_rocm_reserve_the_score_matrix_without_asking_torch(self, monkeypatch):
        monkeypatch.setattr(attention, "_IS_ROCM", True)
        probed: list[int] = []

        def fused_choice(q, *args, **kwargs):
            probed.append(q.shape[-1])
            return next(iter(attention._FUSED_SDP_CHOICES))

        monkeypatch.setattr(torch.ops.aten, "_fused_sdp_choice", fused_choice)
        # On a CPU-only build the probe cannot even allocate on "cuda" and would report the
        # materializing path for the wrong reason; make it run so the test discriminates everywhere.
        real_empty = torch.empty
        monkeypatch.setattr(torch, "empty", lambda *a, device=None, **kw: real_empty(*a, **kw))
        device = torch.device("cuda", 0)

        wide = sdpa_score_matrix_bytes(device=device, dtype=torch.bfloat16, num_heads=1, head_dim=WIDE, seq_len=64)
        assert wide == 64 * 64 * attention.SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert probed == []

        narrow = sdpa_score_matrix_bytes(device=device, dtype=torch.bfloat16, num_heads=1, head_dim=NARROW, seq_len=64)
        assert narrow == 0, "a narrow head still asks torch, and a fused answer means no score matrix"
        assert probed == [NARROW]

    def test_wide_heads_off_rocm_still_ask_torch(self, monkeypatch):
        monkeypatch.setattr(attention, "_IS_ROCM", False)
        monkeypatch.setattr(
            torch.ops.aten, "_fused_sdp_choice", lambda q, *a, **k: next(iter(attention._FUSED_SDP_CHOICES))
        )
        assert (
            sdpa_score_matrix_bytes(
                device=torch.device("cpu"), dtype=torch.float32, num_heads=1, head_dim=WIDE, seq_len=64
            )
            == 0
        )


needs_rocm = pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="needs a ROCm GPU")


@pytest.mark.slow
@needs_rocm
class TestOnRocmHardware:
    """The upstream defect and the guard, on the real device. `slow`: a dev-machine lane, not CI."""

    SHAPE = (1, 1, 16384, 512)  # the FLUX.2 VAE mid-block at a 1024px decode

    def _inputs(self):
        torch.manual_seed(0)
        return (torch.randn(self.SHAPE, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    def test_a_fused_kernel_is_still_wrong_for_the_wide_head(self):
        """Pins the reason the guard exists. When this passes on a new torch, the guard can go."""
        from torch.nn.attention import SDPBackend

        from invokeai.backend.util.sdpa_scope import sdpa_policy

        q, k, v = self._inputs()
        reference = attention._math_sdpa(q, k, v, None, 0.0, False, None, False).float()
        # The unguarded entry point, whether or not the guard is installed in this process.
        unguarded = getattr(F.scaled_dot_product_attention, "__wrapped__", F.scaled_dot_product_attention)
        wrong = 0
        for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
            with sdpa_policy([backend]):
                try:
                    out = unguarded(q, k, v).float()
                except RuntimeError:
                    continue  # the backend refused the shape: not wrong, just absent
            err = (out - reference).abs().max().item() / reference.abs().max().item()
            wrong += int(not torch.isfinite(out).all() or err > 0.05)
        assert wrong > 0, "every fused kernel now agrees with math: the ROCm head-dim guard may be unnecessary"

    def test_the_guard_is_exact_under_allocation_churn(self):
        import random

        install_rocm_sdpa_head_dim_guard()
        q, k, v = self._inputs()
        reference = attention._math_sdpa(q, k, v, None, 0.0, False, None, False)
        rnd = random.Random(0)
        for _ in range(20):
            junk = [
                torch.empty(rnd.randint(1, 400) * 2**20, device="cuda", dtype=torch.uint8)
                for _ in range(rnd.randint(1, 4))
            ]
            out = F.scaled_dot_product_attention(q, k, v)
            del junk
            if rnd.random() < 0.5:
                torch.cuda.empty_cache()
            assert torch.equal(out, reference)


class TestClassicVaeEstimators:
    """The SD/SDXL and FLUX.1 VAE estimators price the mid-block score matrix the guard forces."""

    @pytest.fixture(autouse=True)
    def _rocm_cuda(self, monkeypatch):
        import invokeai.backend.util.vae_working_memory as vwm

        monkeypatch.setattr(attention, "_IS_ROCM", True)
        monkeypatch.setattr(vwm.TorchDevice, "choose_torch_device", classmethod(lambda cls: torch.device("cuda", 0)))

    @staticmethod
    def _vae(spec):
        from unittest.mock import MagicMock

        vae = MagicMock(spec=spec)
        vae.parameters.side_effect = lambda: iter([torch.zeros(1, dtype=torch.bfloat16)])
        return vae

    @staticmethod
    def _score_bytes(px: int) -> int:
        tokens = (px // 8) ** 2
        return tokens * tokens * attention.SDPA_MATH_BYTES_PER_SCORE_ELEMENT

    def test_sdxl_untiled_decode_reserves_the_score_matrix_when_it_dominates(self, monkeypatch):
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_sd15_sdxl

        latents = torch.zeros(1, 4, 2048 // 8, 2048 // 8)
        linear = 2048 * 2048 * 2 * 2200
        with_guard = estimate_vae_working_memory_sd15_sdxl(
            "decode", latents, self._vae(AutoencoderKL), None, fp32=False
        )
        assert with_guard == self._score_bytes(2048) > linear

        monkeypatch.setattr(attention, "_IS_ROCM", False)
        monkeypatch.setattr(
            torch.ops.aten, "_fused_sdp_choice", lambda q, *a, **k: next(iter(attention._FUSED_SDP_CHOICES))
        )
        real_empty = torch.empty
        monkeypatch.setattr(torch, "empty", lambda *a, device=None, **kw: real_empty(*a, **kw))
        assert (
            estimate_vae_working_memory_sd15_sdxl("decode", latents, self._vae(AutoencoderKL), None, fp32=False)
            == linear
        )

    @pytest.mark.parametrize("estimator", ["estimate_vae_working_memory_sd3", "estimate_vae_working_memory_cogview4"])
    def test_the_other_autoencoder_kl_estimators_price_it_too(self, estimator):
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        import invokeai.backend.util.vae_working_memory as vwm

        latents = torch.zeros(1, 16, 2048 // 8, 2048 // 8)
        assert getattr(vwm, estimator)("decode", latents, self._vae(AutoencoderKL)) == self._score_bytes(2048)

    def test_a_tiny_autoencoder_has_no_attention_to_price(self):
        from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny

        from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_sd15_sdxl

        latents = torch.zeros(1, 4, 2048 // 8, 2048 // 8)
        assert estimate_vae_working_memory_sd15_sdxl(
            "decode", latents, self._vae(AutoencoderTiny), None, fp32=False
        ) == (2048 * 2048 * 2 * 2200)

    def test_flux1_decode_prices_the_score_matrix_by_tile_when_tiled(self):
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

        from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux

        latents = torch.zeros(1, 16, 2048 // 8, 2048 // 8)
        untiled = estimate_vae_working_memory_flux("decode", latents, self._vae(AutoencoderKL))
        assert untiled == self._score_bytes(2048)

        tiled = estimate_vae_working_memory_flux("decode", latents, self._vae(AutoencoderKL), tile_size=1024)
        assert tiled < self._score_bytes(2048)
        assert tiled >= self._score_bytes(1024), "a 1024px tile still materializes its own 16384-token score matrix"
