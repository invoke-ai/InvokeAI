"""The MOSS-Image denoise loop against the reference pipeline's sampling loop, on a tiny float32 transformer."""

import pytest
import torch

from invokeai.backend.moss_image.denoise import denoise
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageScheduler, MOSSImageTransformer
from invokeai.backend.moss_image.sampling import get_moss_sigmas
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState

NUM_STEPS = 4
LATENT_SHAPE = (1, 128, 4, 6)


@pytest.fixture(scope="module")
def transformer() -> MOSSImageTransformer:
    """Three blocks of the real architecture with its default (non-zero) initialization, so every block, the
    modulation and the output head shape the prediction."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MOSSImageTransformer(
            dim=64, ffn_dim=128, num_heads=2, num_layers=3, text_dim=16, freq_dim=16, latent_channels=128
        )
    return model.eval()


@pytest.fixture(scope="module")
def inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return {
        "latents": torch.randn(LATENT_SHAPE, generator=generator),
        # Different lengths tell the two branches apart in the call log below.
        "positive": torch.randn(1, 5, 16, generator=generator),
        "negative": torch.randn(1, 3, 16, generator=generator),
    }


def _unpatchify(x: torch.Tensor) -> torch.Tensor:
    """The reference pipeline's unpatchify before vae.decode."""
    b, c, h, w = x.shape
    return x.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)


def _reference_loop(
    transformer: MOSSImageTransformer,
    latents: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    cfg_scale: float,
    uncond_skip_layers: list[int],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """MOSSImagePipeline.__call__'s loop for unpadded embeddings and default SLG settings, plus each step's x0."""
    scheduler = MOSSImageScheduler()
    scheduler.set_timesteps(NUM_STEPS)
    x0s = []
    for index, timestep in enumerate(scheduler.timesteps):
        prediction = transformer(latents, timestep, positive).sample.float()
        if cfg_scale != 1.0:
            skip_kwargs = {"skip_layers": uncond_skip_layers} if uncond_skip_layers else {}
            uncond = transformer(latents, timestep, negative, **skip_kwargs).sample.float()
            prediction = uncond + cfg_scale * (prediction - uncond)
        x0s.append(_unpatchify(latents - scheduler.sigmas[index] * prediction))
        latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
    return latents, x0s


def _denoise(
    transformer: MOSSImageTransformer,
    inputs: dict[str, torch.Tensor],
    cfg_scale: float,
    uncond_skip_layers: list[int],
    sigmas: list[float] | None = None,
    inpaint_extension: RectifiedFlowInpaintExtension | None = None,
) -> tuple[torch.Tensor, list[PipelineIntermediateState]]:
    states: list[PipelineIntermediateState] = []
    out = denoise(
        transformer=transformer,
        latents=inputs["latents"],
        sigmas=get_moss_sigmas(NUM_STEPS) if sigmas is None else sigmas,
        positive=inputs["positive"],
        negative=inputs["negative"],
        cfg_scale=cfg_scale,
        uncond_skip_layers=uncond_skip_layers,
        step_callback=states.append,
        inpaint_extension=inpaint_extension,
    )
    return out, states


@torch.no_grad()
def test_guided_denoise_with_a_skipped_negative_block_matches_the_reference(transformer, inputs) -> None:
    out, states = _denoise(transformer, inputs, cfg_scale=3.0, uncond_skip_layers=[1])

    expected, expected_x0s = _reference_loop(transformer, **inputs, cfg_scale=3.0, uncond_skip_layers=[1])
    unskipped, _ = _reference_loop(transformer, **inputs, cfg_scale=3.0, uncond_skip_layers=[])

    assert torch.equal(out, expected)
    # Guard: skipping the block changes the result, so the comparison above does test the skip.
    assert not torch.allclose(expected, unskipped)
    # Previews are each step's x0 estimate, unpatchified and still BN-normalized.
    assert [(s.step, s.total_steps, s.order) for s in states] == [(i, NUM_STEPS, 1) for i in range(1, NUM_STEPS + 1)]
    for state, x0 in zip(states, expected_x0s, strict=True):
        torch.testing.assert_close(state.latents, x0)


@torch.no_grad()
def test_guidance_scale_one_never_runs_the_negative_branch(transformer, inputs) -> None:
    text_lengths: list[int] = []

    def log_text_length(module, args, kwargs) -> None:
        text = kwargs["encoder_hidden_states"] if "encoder_hidden_states" in kwargs else args[2]
        text_lengths.append(text.shape[1])

    handle = transformer.register_forward_pre_hook(log_text_length, with_kwargs=True)
    try:
        out, _ = _denoise(transformer, inputs, cfg_scale=1.0, uncond_skip_layers=[1])
    finally:
        handle.remove()

    assert text_lengths == [inputs["positive"].shape[1]] * NUM_STEPS
    expected, _ = _reference_loop(transformer, **inputs, cfg_scale=1.0, uncond_skip_layers=[1])
    assert torch.equal(out, expected)


@torch.no_grad()
def test_inpainting_returns_the_kept_region_and_denoises_the_rest(transformer, inputs) -> None:
    generator = torch.Generator().manual_seed(2)
    init = torch.randn(LATENT_SHAPE, generator=generator)
    mask = torch.zeros(LATENT_SHAPE)
    mask[..., :3] = 1.0  # left half is regenerated, right half kept
    extension = RectifiedFlowInpaintExtension(init_latents=init, inpaint_mask=mask, noise=inputs["latents"])

    out, states = _denoise(transformer, inputs, cfg_scale=3.0, uncond_skip_layers=[1], inpaint_extension=extension)

    assert torch.equal(out[..., 3:], init[..., 3:])
    assert not torch.allclose(out[..., :3], init[..., :3])
    # The preview shows the kept region as the init latents themselves (patch columns 3.. are latent columns 6..).
    for state in states:
        assert torch.equal(state.latents[..., 6:], _unpatchify(init)[..., 6:])


def test_single_sigma_returns_the_latents_unchanged(transformer, inputs) -> None:
    out, states = _denoise(transformer, inputs, cfg_scale=3.0, uncond_skip_layers=[1], sigmas=[0.5])

    assert torch.equal(out, inputs["latents"])
    assert states == []


@pytest.mark.parametrize(
    ("cfg_scale", "uncond_skip_layers", "drop_negative", "message"),
    [
        (3.0, [3], False, "uncond_skip_layers"),
        (3.0, [-1], False, "uncond_skip_layers"),
        (3.0, [1], True, "Negative conditioning is required"),
    ],
)
def test_invalid_guidance_settings_are_rejected(
    transformer, inputs, cfg_scale: float, uncond_skip_layers: list[int], drop_negative: bool, message: str
) -> None:
    if drop_negative:
        inputs = {**inputs, "negative": None}
    with pytest.raises(ValueError, match=message):
        _denoise(transformer, inputs, cfg_scale=cfg_scale, uncond_skip_layers=uncond_skip_layers)
