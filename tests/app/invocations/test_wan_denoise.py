"""CPU-only integration tests for ``WanDenoiseInvocation``.

These tests substitute a synthetic transformer (no weights) for the real
``WanTransformer3DModel`` so the denoise loop's shape-handling, scheduler
integration, CFG branch, and step-callback wiring can be exercised on a CPU
runner. End-to-end tests against real Wan checkpoints are gated behind
``INVOKEAI_HEAVY_TESTS=1`` and require a working CUDA install.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from invokeai.app.invocations.fields import WanConditioningField, WanRefImageConditioningField
from invokeai.app.invocations.model import WanTransformerField
from invokeai.app.invocations.wan_denoise import WanDenoiseInvocation
from invokeai.backend.model_manager.taxonomy import WanVariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    WanConditioningInfo,
)


class _ZeroTransformer(nn.Module):
    """Stand-in for ``WanTransformer3DModel``.

    Returns ``torch.zeros_like(hidden_states)`` so the flow-matching scheduler
    treats every step as a no-op velocity. After N steps the latents equal the
    initial noise — a useful invariant for shape correctness.

    ``label`` lets dual-expert tests record which expert was invoked.
    """

    def __init__(self, label: str = "single") -> None:
        super().__init__()
        self.dtype = torch.float32
        self.label = label
        self.calls: list[tuple[int, ...]] = []
        self.timesteps_seen: list[float] = []

    def forward(  # noqa: D401 — match diffusers signature
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs=None,
        return_dict: bool = True,
    ):
        # Record the call so assertions can verify shape contracts.
        self.calls.append(
            (
                tuple(hidden_states.shape),
                tuple(timestep.shape),
                tuple(encoder_hidden_states.shape),
            )
        )
        # Record the timestep (t.expand(B) → take first element).
        self.timesteps_seen.append(float(timestep.flatten()[0].item()))
        # Real Wan I2V transformer has in_channels=36 (16 noise + 20 ref-image
        # condition) but out_channels=16. T2V is 16/16 and TI2V-5B is 48/48 —
        # both have matching in/out. Mirror that by only collapsing the I2V
        # input width back to 16 channels.
        out_shape = list(hidden_states.shape)
        if out_shape[1] == 36:
            out_shape[1] = 16
        out = torch.zeros(out_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        if return_dict:
            return type("Out", (), {"sample": out})
        return (out,)


@contextmanager
def _model_on_device_ctx(model: nn.Module):
    yield (None, model)


def _make_loaded_model(model: nn.Module) -> MagicMock:
    """Mock ``LoadedModel`` exposing only the methods the denoise loop touches."""
    loaded = MagicMock()
    loaded.model_on_device = lambda: _model_on_device_ctx(model)
    return loaded


def _build_context(
    transformer: nn.Module,
    *,
    variant: WanVariantType,
    model_root: Path,
    pos_cond: WanConditioningInfo,
    neg_cond: WanConditioningInfo | None,
    transformer_low: nn.Module | None = None,
) -> MagicMock:
    """Build a MagicMock InvocationContext sufficient for ``_run_diffusion``.

    When ``transformer_low`` is provided, ``context.models.load`` routes the
    request based on the ``ModelIdentifierField.submodel_type`` so dual-expert
    code paths see two distinct loaded models.
    """
    config = MagicMock()
    config.variant = variant
    config.format = "diffusers"

    context = MagicMock()
    context.models.get_config.return_value = config
    context.models.get_absolute_path.return_value = model_root

    def _load(model_id) -> MagicMock:
        submodel_type = getattr(model_id, "submodel_type", None)
        if transformer_low is not None and str(submodel_type) == "SubModelType.Transformer2":
            return _make_loaded_model(transformer_low)
        return _make_loaded_model(transformer)

    context.models.load.side_effect = _load

    def _load_conditioning(name: str) -> ConditioningFieldData:
        if name == "pos":
            return ConditioningFieldData(conditionings=[pos_cond])
        if name == "neg" and neg_cond is not None:
            return ConditioningFieldData(conditionings=[neg_cond])
        raise KeyError(name)

    context.conditioning.load.side_effect = _load_conditioning
    context.util.signal_progress = MagicMock()
    context.util.sd_step_callback = MagicMock()
    context.logger = MagicMock()
    return context


def _make_conditioning(seq_len: int = 226, hidden: int = 4096) -> WanConditioningInfo:
    return WanConditioningInfo(
        prompt_embeds=torch.zeros(seq_len, hidden),
        prompt_attention_mask=None,
    )


def _make_invocation(
    transformer_field: WanTransformerField,
    pos_field: WanConditioningField,
    neg_field: WanConditioningField | None,
    *,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    guidance_scale_low_noise: float | None = None,
) -> WanDenoiseInvocation:
    return WanDenoiseInvocation(
        id="test",
        transformer=transformer_field,
        positive_conditioning=pos_field,
        negative_conditioning=neg_field,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        guidance_scale_low_noise=guidance_scale_low_noise,
        seed=42,
    )


@pytest.fixture
def fake_model_root():
    """A directory layout the denoise helpers can read.

    No ``scheduler/`` subfolder, so the scheduler falls back to defaults — that
    keeps the test self-contained.
    """
    with TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Pin TorchDevice to CPU + float32 for deterministic, GPU-free tests."""
    from invokeai.backend.util.devices import TorchDevice

    monkeypatch.setattr(TorchDevice, "choose_torch_device", classmethod(lambda cls: torch.device("cpu")))
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", classmethod(lambda cls, device=None: torch.float32))


def _wan_transformer_field(*, dual: bool = False, boundary_ratio: float = 0.875) -> WanTransformerField:
    """Build a WanTransformerField. With ``dual=True`` a low-noise expert slot
    is also populated so the denoise loop exercises the MoE swap path."""
    base_id = {
        "key": "wan-test",
        "name": "wan-test",
        "base": "wan",
        "type": "main",
        "hash": "h",
    }
    field_kwargs: dict = {
        "transformer": {**base_id, "submodel_type": "transformer"},
        "boundary_ratio": boundary_ratio,
    }
    if dual:
        field_kwargs["transformer_low_noise"] = {**base_id, "submodel_type": "transformer_2"}
    return WanTransformerField(**field_kwargs)


class TestWanDenoiseShapes:
    """Verify the denoise loop runs end-to-end on CPU for both variants."""

    @pytest.mark.parametrize(
        "variant,latent_channels,scale,height,width",
        [
            (WanVariantType.T2V_A14B, 16, 8, 64, 64),
            (WanVariantType.TI2V_5B, 48, 16, 64, 64),
        ],
    )
    def test_run_diffusion_returns_4d_finite(
        self, variant, latent_channels, scale, height, width, fake_model_root
    ) -> None:
        transformer = _ZeroTransformer()
        pos = _make_conditioning()
        ctx = _build_context(
            transformer,
            variant=variant,
            model_root=fake_model_root,
            pos_cond=pos,
            neg_cond=None,
        )

        inv = _make_invocation(
            transformer_field=_wan_transformer_field(),
            pos_field=WanConditioningField(conditioning_name="pos"),
            neg_field=None,
            width=width,
            height=height,
            steps=4,
            guidance_scale=1.0,  # disables CFG, so neg conditioning isn't required
        )

        latents = inv._run_diffusion(ctx)

        # Output is 4D [B, C, H/scale, W/scale] — temporal dim squeezed.
        assert latents.ndim == 4
        assert latents.shape == (1, latent_channels, height // scale, width // scale)
        assert torch.isfinite(latents).all()

        # Transformer should have been called exactly steps times.
        assert len(transformer.calls) == 4
        # Hidden states are 5D with T=1.
        h_shape, t_shape, ctx_shape = transformer.calls[0]
        assert h_shape == (1, latent_channels, 1, height // scale, width // scale)
        assert t_shape == (1,)
        assert ctx_shape == (1, 226, 4096)

        # Step callback invoked once per step.
        assert ctx.util.sd_step_callback.call_count == 4

    def test_cfg_doubles_transformer_calls(self, fake_model_root) -> None:
        """With cfg_scale != 1.0 and a negative prompt, each step runs the model twice."""
        transformer = _ZeroTransformer()
        pos = _make_conditioning()
        neg = _make_conditioning()
        ctx = _build_context(
            transformer,
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            pos_cond=pos,
            neg_cond=neg,
        )

        inv = _make_invocation(
            transformer_field=_wan_transformer_field(),
            pos_field=WanConditioningField(conditioning_name="pos"),
            neg_field=WanConditioningField(conditioning_name="neg"),
            width=64,
            height=64,
            steps=3,
            guidance_scale=4.0,
        )

        inv._run_diffusion(ctx)
        # 3 steps × 2 (cond + uncond) = 6 forward calls.
        assert len(transformer.calls) == 6

    def test_zero_velocity_preserves_initial_noise(self, fake_model_root) -> None:
        """A zero-output transformer means the flow-match step never updates latents."""
        transformer = _ZeroTransformer()
        pos = _make_conditioning()
        ctx = _build_context(
            transformer,
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            pos_cond=pos,
            neg_cond=None,
        )

        inv = _make_invocation(
            transformer_field=_wan_transformer_field(),
            pos_field=WanConditioningField(conditioning_name="pos"),
            neg_field=None,
            width=64,
            height=64,
            steps=4,
            guidance_scale=1.0,
        )

        latents = inv._run_diffusion(ctx)

        # Reproduce the same noise the loop would have generated and compare.
        from invokeai.backend.wan.sampling_utils import make_noise

        expected = make_noise(
            batch_size=1,
            latent_channels=16,
            height=64,
            width=64,
            spatial_scale_factor=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=42,
        ).squeeze(2)

        assert torch.allclose(latents, expected, atol=1e-5)


class TestWanDenoiseDualExpert:
    """Verify the A14B dual-expert MoE swap behaves correctly."""

    def test_swap_fires_at_boundary(self, fake_model_root) -> None:
        """High expert handles t >= boundary_timestep, low expert handles t < boundary_timestep."""
        high = _ZeroTransformer(label="high")
        low = _ZeroTransformer(label="low")
        pos = _make_conditioning()
        ctx = _build_context(
            high,
            transformer_low=low,
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            pos_cond=pos,
            neg_cond=None,
        )

        # boundary_ratio=0.5 → boundary_timestep=500 (default num_train_timesteps=1000).
        inv = _make_invocation(
            transformer_field=_wan_transformer_field(dual=True, boundary_ratio=0.5),
            pos_field=WanConditioningField(conditioning_name="pos"),
            neg_field=None,
            width=64,
            height=64,
            steps=10,
            guidance_scale=1.0,
        )

        inv._run_diffusion(ctx)

        # Both experts called.
        assert len(high.timesteps_seen) > 0, "high-noise expert never invoked"
        assert len(low.timesteps_seen) > 0, "low-noise expert never invoked"

        # Every high-noise timestep is >= 500; every low-noise timestep is < 500.
        for t in high.timesteps_seen:
            assert t >= 500.0, f"high-noise expert saw t={t}, should be >= 500"
        for t in low.timesteps_seen:
            assert t < 500.0, f"low-noise expert saw t={t}, should be < 500"

        # Total steps adds up.
        assert len(high.timesteps_seen) + len(low.timesteps_seen) == 10

    def test_no_swap_when_boundary_skipped(self, fake_model_root) -> None:
        """boundary_ratio=0.0 → boundary_timestep=0 → all timesteps go to high-noise expert."""
        high = _ZeroTransformer(label="high")
        low = _ZeroTransformer(label="low")
        pos = _make_conditioning()
        ctx = _build_context(
            high,
            transformer_low=low,
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            pos_cond=pos,
            neg_cond=None,
        )

        inv = _make_invocation(
            transformer_field=_wan_transformer_field(dual=True, boundary_ratio=0.0),
            pos_field=WanConditioningField(conditioning_name="pos"),
            neg_field=None,
            width=64,
            height=64,
            steps=4,
            guidance_scale=1.0,
        )

        inv._run_diffusion(ctx)

        # boundary_timestep=0 → t >= 0 always → high-noise expert handles every step.
        assert len(high.timesteps_seen) == 4
        assert len(low.timesteps_seen) == 0

    def test_full_low_noise_when_boundary_at_max(self, fake_model_root) -> None:
        """boundary_ratio=1.0 → boundary_timestep=1000 → every step goes to the low-noise expert.

        The UniPC flow schedule's first timestep is slightly below 1000 (flow sigmas
        don't start exactly at 1.0), so with the boundary at the maximum no timestep
        satisfies ``t >= 1000`` and the high-noise expert is never invoked.
        """
        high = _ZeroTransformer(label="high")
        low = _ZeroTransformer(label="low")
        pos = _make_conditioning()
        ctx = _build_context(
            high,
            transformer_low=low,
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            pos_cond=pos,
            neg_cond=None,
        )

        inv = _make_invocation(
            transformer_field=_wan_transformer_field(dual=True, boundary_ratio=1.0),
            pos_field=WanConditioningField(conditioning_name="pos"),
            neg_field=None,
            width=64,
            height=64,
            steps=4,
            guidance_scale=1.0,
        )

        inv._run_diffusion(ctx)

        # All steps are < 1000 → low expert handles the whole schedule.
        assert len(high.timesteps_seen) == 0
        assert len(low.timesteps_seen) == 4

    def test_cfg_with_dual_experts_doubles_calls_per_step(self, fake_model_root) -> None:
        """With negative conditioning + cfg_scale != 1, every step runs the active expert twice."""
        high = _ZeroTransformer(label="high")
        low = _ZeroTransformer(label="low")
        pos = _make_conditioning()
        neg = _make_conditioning()
        ctx = _build_context(
            high,
            transformer_low=low,
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            pos_cond=pos,
            neg_cond=neg,
        )

        inv = _make_invocation(
            transformer_field=_wan_transformer_field(dual=True, boundary_ratio=0.5),
            pos_field=WanConditioningField(conditioning_name="pos"),
            neg_field=WanConditioningField(conditioning_name="neg"),
            width=64,
            height=64,
            steps=6,
            guidance_scale=4.0,
            guidance_scale_low_noise=2.0,  # Field accepted by the invocation; effect is implicit.
        )

        inv._run_diffusion(ctx)

        # Total transformer invocations: 6 steps × 2 (cond + uncond) = 12, split across experts.
        total = len(high.timesteps_seen) + len(low.timesteps_seen)
        assert total == 12

        # Each unique timestep appears twice (cond + uncond) on the same expert.
        from collections import Counter

        high_counts = Counter(high.timesteps_seen)
        low_counts = Counter(low.timesteps_seen)
        assert all(v == 2 for v in high_counts.values()), high_counts
        assert all(v == 2 for v in low_counts.values()), low_counts

        # And the swap actually happened — both experts saw work.
        assert len(high_counts) > 0 and len(low_counts) > 0


@pytest.mark.skipif(
    os.environ.get("INVOKEAI_HEAVY_TESTS") != "1",
    reason="End-to-end test requires real Wan weights and CUDA; opt in with INVOKEAI_HEAVY_TESTS=1",
)
class TestWanDenoiseHeavy:
    """Placeholder for a real-weights smoke test once CUDA is available."""

    def test_real_ti2v_5b_runs(self) -> None:
        pytest.skip("Heavy test stub — implement once a TI2V-5B checkpoint is installable.")


class TestWanDenoiseRefImage:
    """Phase 7: VAE-latent reference-image conditioning for I2V-A14B.

    The denoise loop must concatenate the 20-channel condition tensor to the
    16-channel noise latents at every transformer call, producing 36-channel
    input. Variant gate must fast-fail when ref_image is wired to a non-I2V
    transformer."""

    def _build_ctx_with_condition(
        self,
        transformer: _ZeroTransformer,
        variant: WanVariantType,
        model_root: Path,
        condition_tensor: torch.Tensor | None,
    ) -> MagicMock:
        ctx = _build_context(
            transformer,
            variant=variant,
            model_root=model_root,
            pos_cond=_make_conditioning(),
            neg_cond=None,
        )
        if condition_tensor is not None:
            ctx.tensors.load.return_value = condition_tensor
        return ctx

    def _make_inv_with_ref(
        self,
        ref_field: "WanRefImageConditioningField | None",
        *,
        width: int = 64,
        height: int = 64,
    ) -> WanDenoiseInvocation:
        return WanDenoiseInvocation(
            id="test",
            transformer=_wan_transformer_field(dual=True),
            positive_conditioning=WanConditioningField(conditioning_name="pos"),
            negative_conditioning=None,
            ref_image=ref_field,
            width=width,
            height=height,
            steps=3,
            guidance_scale=1.0,
            seed=42,
        )

    def test_ref_image_concatenated_to_36_channels(self, fake_model_root: Path) -> None:
        """I2V_A14B + ref_image → transformer sees [B, 36, T, H/8, W/8]."""
        transformer = _ZeroTransformer()
        # Build the 20-channel condition tensor the encoder would have saved:
        # 4-ch first-frame mask + 16-ch VAE-encoded image latents.
        # At 64x64 → 8x8 latent spatial dims.
        condition = torch.zeros(1, 20, 1, 8, 8)
        ctx = self._build_ctx_with_condition(transformer, WanVariantType.I2V_A14B, fake_model_root, condition)

        ref_field = WanRefImageConditioningField(condition_tensor_name="condition", width=64, height=64)
        inv = self._make_inv_with_ref(ref_field)
        inv._run_diffusion(ctx)

        assert len(transformer.calls) == 3
        # Every call's hidden_states must have 36 channels (16 noise + 20 condition).
        for h_shape, *_ in transformer.calls:
            assert h_shape == (1, 36, 1, 8, 8), f"expected 36-channel input, got {h_shape}"

    def test_no_ref_image_keeps_16_channels(self, fake_model_root: Path) -> None:
        """Without ref_image → transformer sees [B, 16, T, H/8, W/8] as before."""
        transformer = _ZeroTransformer()
        ctx = self._build_ctx_with_condition(
            transformer, WanVariantType.I2V_A14B, fake_model_root, condition_tensor=None
        )

        inv = self._make_inv_with_ref(ref_field=None)
        inv._run_diffusion(ctx)

        for h_shape, *_ in transformer.calls:
            assert h_shape == (1, 16, 1, 8, 8), f"expected unchanged 16-channel input, got {h_shape}"

    def test_variant_gate_rejects_ref_image_on_t2v(self, fake_model_root: Path) -> None:
        """T2V_A14B + ref_image must raise — fast-fail before doing any work."""
        transformer = _ZeroTransformer()
        condition = torch.zeros(1, 20, 1, 8, 8)
        ctx = self._build_ctx_with_condition(transformer, WanVariantType.T2V_A14B, fake_model_root, condition)

        ref_field = WanRefImageConditioningField(condition_tensor_name="condition", width=64, height=64)
        inv = self._make_inv_with_ref(ref_field)
        with pytest.raises(ValueError, match="only supported by the Wan 2.2 I2V variant"):
            inv._run_diffusion(ctx)

    def test_variant_gate_rejects_ref_image_on_ti2v(self, fake_model_root: Path) -> None:
        """TI2V-5B + ref_image must raise — TI2V uses a different image path."""
        transformer = _ZeroTransformer()
        condition = torch.zeros(1, 20, 1, 8, 8)
        ctx = self._build_ctx_with_condition(transformer, WanVariantType.TI2V_5B, fake_model_root, condition)

        ref_field = WanRefImageConditioningField(condition_tensor_name="condition", width=64, height=64)
        inv = self._make_inv_with_ref(ref_field)
        with pytest.raises(ValueError, match="only supported by the Wan 2.2 I2V variant"):
            inv._run_diffusion(ctx)

    def test_dim_mismatch_raises(self, fake_model_root: Path) -> None:
        """If the encoder's width/height differ from denoise's, fail clearly."""
        transformer = _ZeroTransformer()
        condition = torch.zeros(1, 20, 1, 8, 8)
        ctx = self._build_ctx_with_condition(transformer, WanVariantType.I2V_A14B, fake_model_root, condition)

        ref_field = WanRefImageConditioningField(condition_tensor_name="condition", width=512, height=512)
        inv = self._make_inv_with_ref(ref_field, width=64, height=64)
        with pytest.raises(ValueError, match="must match denoise dimensions"):
            inv._run_diffusion(ctx)


class TestWanDenoiseInpaint:
    """Phase 8: ``denoise_mask`` (inpaint) wiring via ``RectifiedFlowInpaintExtension``.

    User-side mask convention (matches Anima / Flux): 1.0 = preserve,
    0.0 = regenerate. After ``_prep_inpaint_mask`` inverts, the extension
    sees: 0.0 = preserve, 1.0 = regenerate.

    With the synthetic zero-output transformer, the scheduler step is a
    no-op (noise_pred=0 → latents unchanged). The init latents are placed
    into the preserved regions at every step via the extension's merge
    function; the regenerated regions stay as the original noise tensor
    because the model never updates them.
    """

    def _build_inpaint_context(
        self,
        transformer: _ZeroTransformer,
        variant: WanVariantType,
        model_root: Path,
        init_latents: torch.Tensor,
        mask: torch.Tensor,
    ) -> MagicMock:
        ctx = _build_context(
            transformer,
            variant=variant,
            model_root=model_root,
            pos_cond=_make_conditioning(),
            neg_cond=None,
        )

        # tensors.load needs to return different tensors for the init-latents
        # and the mask, dispatched by the name field.
        def _load_tensor(name: str) -> torch.Tensor:
            if name == "init":
                return init_latents
            if name == "mask":
                return mask
            raise KeyError(name)

        ctx.tensors.load.side_effect = _load_tensor
        return ctx

    def test_preserved_region_matches_init_exactly(self, fake_model_root: Path) -> None:
        from invokeai.app.invocations.fields import DenoiseMaskField, LatentsField

        transformer = _ZeroTransformer()
        # 64x64 image -> 8x8 latents at scale 8 (T2V-A14B family).
        # Init latents: fixed value 0.5 so the preserved region is detectable.
        init_latents = torch.full((1, 16, 8, 8), 0.5)
        # Mask: 8x8 spatial mask, half-1 (preserve left), half-0 (regenerate right).
        # User-side convention: 1 = preserve, 0 = regenerate.
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :, :4] = 1.0  # left half preserved

        ctx = self._build_inpaint_context(
            transformer,
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            init_latents=init_latents,
            mask=mask,
        )

        inv = WanDenoiseInvocation(
            id="test",
            transformer=_wan_transformer_field(),
            positive_conditioning=WanConditioningField(conditioning_name="pos"),
            negative_conditioning=None,
            latents=LatentsField(latents_name="init"),
            denoise_mask=DenoiseMaskField(mask_name="mask", masked_latents_name=None, gradient=False),
            width=64,
            height=64,
            steps=4,
            guidance_scale=1.0,
            denoising_start=0.0,
            denoising_end=1.0,
            seed=42,
        )

        out = inv._run_diffusion(ctx)  # [B, C, H_lat, W_lat]
        assert out.shape == (1, 16, 8, 8)

        # Preserved (left) half: must exactly match the init latents at t_prev=0
        # (final step's merge produces noised_init = noise*0 + 1*init = init).
        assert torch.allclose(out[..., :, :4], torch.full_like(out[..., :, :4], 0.5)), (
            "Preserved region must equal init latents at the end of denoise"
        )

        # Regenerated (right) half: model never changed anything (zero transformer)
        # so this region stays equal to the original noise, NOT to init.
        # Assert it's *not* equal to init — concrete proof the regions are
        # being handled separately.
        assert not torch.allclose(out[..., :, 4:], torch.full_like(out[..., :, 4:], 0.5)), (
            "Regenerated region should NOT equal init — extension must route it through the model path"
        )

    def test_inpaint_requires_init_latents(self, fake_model_root: Path) -> None:
        """Providing a mask without init latents must raise — there's nothing
        to merge back into the preserved regions."""
        from invokeai.app.invocations.fields import DenoiseMaskField

        transformer = _ZeroTransformer()
        mask = torch.ones(1, 1, 8, 8)
        ctx = self._build_inpaint_context(
            transformer,
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            init_latents=torch.zeros(1, 16, 8, 8),  # unused
            mask=mask,
        )

        inv = WanDenoiseInvocation(
            id="test",
            transformer=_wan_transformer_field(),
            positive_conditioning=WanConditioningField(conditioning_name="pos"),
            negative_conditioning=None,
            latents=None,  # missing — error
            denoise_mask=DenoiseMaskField(mask_name="mask", masked_latents_name=None, gradient=False),
            width=64,
            height=64,
            steps=2,
            guidance_scale=1.0,
            seed=42,
        )

        with pytest.raises(ValueError, match="img2img inpainting"):
            inv._run_diffusion(ctx)

    def test_no_mask_path_is_unchanged(self, fake_model_root: Path) -> None:
        """Without a denoise_mask, the loop behaves as before — sanity check
        that adding the inpaint extension didn't introduce a regression on
        the non-inpaint codepath."""
        from invokeai.app.invocations.fields import LatentsField

        transformer = _ZeroTransformer()
        init_latents = torch.full((1, 16, 8, 8), 0.3)
        ctx = self._build_inpaint_context(
            transformer,
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            init_latents=init_latents,
            mask=torch.zeros(1, 1, 8, 8),  # unused — no mask wired
        )

        inv = WanDenoiseInvocation(
            id="test",
            transformer=_wan_transformer_field(),
            positive_conditioning=WanConditioningField(conditioning_name="pos"),
            negative_conditioning=None,
            latents=LatentsField(latents_name="init"),
            denoise_mask=None,  # no mask
            width=64,
            height=64,
            steps=4,
            guidance_scale=1.0,
            denoising_start=0.5,  # img2img-style partial denoise
            denoising_end=1.0,
            seed=42,
        )

        out = inv._run_diffusion(ctx)
        assert out.shape == (1, 16, 8, 8)
        assert torch.isfinite(out).all()


class TestLoRAExpertWiring:
    def test_high_only_loras_do_not_leak_to_low_expert(self, fake_model_root) -> None:
        """A LoRA routed only to the primary (high-noise) list must not be applied to
        the low-noise expert (PR #9163 review): the old ``loras_low_noise or loras``
        fallback silently reused the primary list for the low expert, so an
        expert-tagged LoRA (e.g. a Lightning high-noise distill) was misapplied and
        high-only targeting was impossible."""
        from unittest.mock import patch

        from invokeai.app.invocations.model import LoRAField
        from invokeai.app.invocations.wan_denoise import _ExpertSwapper  # noqa: F401 (documents the patch target)

        captured: dict = {}

        class _RecordingSwapper:
            HIGH = "high"
            LOW = "low"

            def __init__(self, **kwargs) -> None:
                captured.update(kwargs)
                self._model = _ZeroTransformer(label="either")

            def get(self, label: str) -> nn.Module:
                return self._model

            def close(self) -> None:
                pass

        field = _wan_transformer_field(dual=True)
        field.loras.append(
            LoRAField(
                lora={"key": "l", "hash": "h", "name": "lightning-high", "base": "wan", "type": "lora"},
                weight=1.0,
            )
        )
        assert field.loras_low_noise == []

        ctx = _build_context(
            _ZeroTransformer(label="high"),
            transformer_low=_ZeroTransformer(label="low"),
            variant=WanVariantType.T2V_A14B,
            model_root=fake_model_root,
            pos_cond=_make_conditioning(),
            neg_cond=None,
        )
        inv = _make_invocation(
            transformer_field=field,
            pos_field=WanConditioningField(conditioning_name="pos"),
            neg_field=None,
            width=64,
            height=64,
            steps=2,
            guidance_scale=1.0,
        )
        with patch("invokeai.app.invocations.wan_denoise._ExpertSwapper", _RecordingSwapper):
            inv._run_diffusion(ctx)

        assert captured["high_lora_factory"] is not None
        assert captured["low_lora_factory"] is None


class TestDefaultSchedulerForVariant:
    """``_default_scheduler_for_variant`` returns the right class + config when no
    on-disk ``scheduler/`` directory exists (the standalone GGUF / single-file case).
    """

    def test_ti2v_5b_returns_unipc_with_flow_config(self) -> None:
        from diffusers import UniPCMultistepScheduler

        from invokeai.app.invocations.wan_denoise import _default_scheduler_for_variant

        s = _default_scheduler_for_variant(WanVariantType.TI2V_5B)
        assert isinstance(s, UniPCMultistepScheduler)
        # The combination below is what makes this a "Wan flow" UniPC rather than a
        # generic UniPC schedule — wrong values here drift on TI2V-5B samples.
        assert s.config.flow_shift == 5.0
        assert s.config.prediction_type == "flow_prediction"
        assert s.config.use_flow_sigmas is True
        assert s.config.solver_type == "bh2"

    def test_a14b_variants_return_unipc_with_flow_shift_3(self) -> None:
        """Both A14B reference repos (Wan-AI/Wan2.2-{T2V,I2V}-A14B-Diffusers) ship
        UniPCMultistepScheduler with flow_shift=3.0 — NOT FlowMatchEuler (PR #9163
        review): an unshifted Euler fallback silently degrades every A14B GGUF render
        and skews how many steps land above the MoE expert boundary."""
        from diffusers import UniPCMultistepScheduler

        from invokeai.app.invocations.wan_denoise import _default_scheduler_for_variant

        for v in (WanVariantType.T2V_A14B, WanVariantType.I2V_A14B):
            s = _default_scheduler_for_variant(v)
            assert isinstance(s, UniPCMultistepScheduler)
            assert s.config.flow_shift == 3.0
            assert s.config.prediction_type == "flow_prediction"
            assert s.config.use_flow_sigmas is True
            assert s.config.solver_type == "bh2"
