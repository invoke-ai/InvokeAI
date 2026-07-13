import pytest
import torch

from invokeai.app.invocations.fields import DenoiseMaskField
from invokeai.app.invocations.krea2_denoise import KREA2_LATENT_CHANNELS, Krea2DenoiseInvocation


@pytest.mark.parametrize(("denoising_start", "denoising_end"), [(0.75, 0.25), (0.5, 0.5)])
def test_validate_inputs_rejects_empty_or_inverted_denoising_range(
    denoising_start: float, denoising_end: float
) -> None:
    invocation = Krea2DenoiseInvocation.model_construct(
        denoising_start=denoising_start, denoising_end=denoising_end, denoise_mask=None
    )

    with pytest.raises(ValueError, match="denoising_start must be less than denoising_end"):
        invocation._validate_inputs()


def test_validate_inputs_rejects_denoise_mask_without_latents() -> None:
    invocation = Krea2DenoiseInvocation.model_construct(
        denoising_start=0.0,
        denoising_end=1.0,
        denoise_mask=DenoiseMaskField(mask_name="mask"),
        latents=None,
    )

    with pytest.raises(ValueError, match="Initial latents are required when a denoise mask is provided"):
        invocation._validate_inputs()


def test_validate_inputs_accepts_a_valid_configuration() -> None:
    invocation = Krea2DenoiseInvocation.model_construct(
        denoising_start=0.0, denoising_end=1.0, denoise_mask=None, latents=None
    )
    # A full-range denoise with no mask is valid and must not raise.
    invocation._validate_inputs()


class TestPrepareCfgScale:
    def test_scalar_is_broadcast_to_the_step_count(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct(cfg_scale=4.5)
        assert invocation._prepare_cfg_scale(8) == [4.5] * 8

    def test_list_of_matching_length_is_returned_unchanged(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct(cfg_scale=[4.0, 3.0, 2.0])
        assert invocation._prepare_cfg_scale(3) == [4.0, 3.0, 2.0]

    def test_list_of_wrong_length_raises(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct(cfg_scale=[4.0, 3.0, 2.0])
        with pytest.raises(ValueError, match="cfg_scale list has 3 values but the model is configured for 8 steps"):
            invocation._prepare_cfg_scale(8)

    def test_invalid_type_raises(self) -> None:
        invocation = Krea2DenoiseInvocation.model_construct(cfg_scale="nonsense")
        with pytest.raises(ValueError, match="Invalid CFG scale type"):
            invocation._prepare_cfg_scale(8)


def test_cfg_scale_list_is_built_against_full_step_count_then_clipped() -> None:
    # Regression: a per-step CFG list carries one value per *configured* step. img2img/inpaint clips the
    # sampling schedule (denoising_start/denoising_end), which shrinks the number of active steps. The list
    # must be prepared against the full step count (``total_sigmas``) and *then* sliced to the active
    # window — preparing it against the already-reduced count would reject the user's full-length list.
    steps = 8
    invocation = Krea2DenoiseInvocation.model_construct(cfg_scale=[float(i) for i in range(steps)])

    full_cfg_scale = invocation._prepare_cfg_scale(steps)  # built against the FULL step count -> ok
    assert len(full_cfg_scale) == steps

    # Slicing to the active window [start_idx:end_idx] (as _run_diffusion does) keeps the right per-step values.
    start_idx, end_idx = 2, 6
    assert full_cfg_scale[start_idx:end_idx] == [2.0, 3.0, 4.0, 5.0]

    # Preparing against the reduced (clipped) count instead would have raised — the bug this guards against.
    with pytest.raises(ValueError):
        invocation._prepare_cfg_scale(end_idx - start_idx)


def test_get_noise_is_deterministic_and_correctly_shaped() -> None:
    invocation = Krea2DenoiseInvocation.model_construct()
    device = torch.device("cpu")

    noise_a = invocation._get_noise(height=64, width=128, dtype=torch.float32, device=device, seed=42)
    noise_b = invocation._get_noise(height=64, width=128, dtype=torch.float32, device=device, seed=42)
    noise_other = invocation._get_noise(height=64, width=128, dtype=torch.float32, device=device, seed=43)

    # Shape is (1, latent_channels, H // 8, W // 8).
    assert noise_a.shape == (1, KREA2_LATENT_CHANNELS, 8, 16)
    # Same seed -> identical noise (reproducibility); different seed -> different noise.
    assert torch.equal(noise_a, noise_b)
    assert not torch.equal(noise_a, noise_other)
