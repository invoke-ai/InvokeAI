"""The FLUX.3 schedule, position ids, noise and reference sizing, against the reference implementation.

The expected values are restated from BFL's `src/flux3/sampling.py` / `pipeline.py` (Apache-2.0) rather than
imported, so a change to either side shows up here.
"""

import math

import pytest
import torch
from PIL import Image

import invokeai.backend.flux3.ref_images as ref_images
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.flux2.latent_norm import bn_normalize
from invokeai.backend.flux2.sampling_utils import pack_flux2
from invokeai.backend.flux3.ref_images import encode_flux3_references, fit_reference_image
from invokeai.backend.flux3.sampling import (
    get_noise_flux3,
    get_schedule_flux3,
    image_position_ids,
    redensify_schedule_flux3,
    text_position_ids,
)


def _bfl_timesteps(steps: int, alpha: float = 6.93) -> list[float]:
    t = torch.linspace(1.0, 0.0, steps + 1)
    return (alpha * t / (1.0 + (alpha - 1.0) * t)).tolist()


@pytest.mark.parametrize("steps", [4, 16, 50])
def test_the_schedule_is_the_reference_samplers(steps: int) -> None:
    schedule = get_schedule_flux3(steps)
    assert len(schedule) == steps + 1
    assert schedule[0] == 1.0 and schedule[-1] == 0.0
    assert schedule == pytest.approx(_bfl_timesteps(steps), abs=1e-6)


def test_a_clipped_window_is_respaced_to_the_requested_step_count() -> None:
    clipped = clip_timestep_schedule_fractional(get_schedule_flux3(16), 0.6, 1.0)
    assert len(clipped) - 1 < 16, "the shift leaves few steps in a low-strength window"

    respaced = redensify_schedule_flux3(clipped, 16)

    assert len(respaced) == 17
    assert respaced[0] == clipped[0] and respaced[-1] == clipped[-1]
    assert all(a > b for a, b in zip(respaced, respaced[1:], strict=False))


def test_a_full_schedule_is_left_alone() -> None:
    schedule = get_schedule_flux3(16)
    assert redensify_schedule_flux3(schedule, 16) is schedule


def test_image_ids_scale_rows_and_columns_and_carry_the_reference_index() -> None:
    ids = image_position_ids(2, 3, l_coord=-2)
    assert ids.shape == (6, 4) and ids.dtype == torch.float32
    # Row-major, as the packed tokens are.
    assert ids[:, 1].tolist() == [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    assert ids[:, 2].tolist() == [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    assert set(ids[:, 0].tolist()) == {0.0} and set(ids[:, 3].tolist()) == {-2.0}


def test_text_advances_along_the_last_axis() -> None:
    ids = text_position_ids(3)
    assert ids.tolist() == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]]


def test_the_noise_reproduces_the_reference_draw() -> None:
    """BFL draws (h*w, 128) image tokens from a seeded CPU generator and rounds them to bf16."""
    seed, height, width = 7, 64, 96
    expected = torch.randn((height // 16 * width // 16, 128), generator=torch.Generator().manual_seed(seed))
    expected = expected.to(torch.bfloat16).float()

    noise = get_noise_flux3(height, width, seed)

    assert noise.shape == (1, 32, height // 8, width // 8)
    assert torch.equal(pack_flux2(noise)[0], expected)


@pytest.mark.parametrize(
    ("size", "count", "max_pixels"),
    [((3000, 2000), 1, 4 * 1024 * 1024), ((3000, 2000), 3, 1024 * 1024), ((1000, 700), 2, 1024 * 1024)],
)
def test_references_are_fitted_to_the_budget_and_the_grid(size: tuple[int, int], count: int, max_pixels: int) -> None:
    fitted = fit_reference_image(Image.new("RGB", size), count)
    assert fitted.width % 16 == 0 and fitted.height % 16 == 0
    assert fitted.width * fitted.height <= max_pixels
    assert math.isclose(fitted.width / fitted.height, size[0] / size[1], rel_tol=0.03)


def test_a_tiny_reference_is_scaled_up_and_an_alpha_image_becomes_rgb() -> None:
    fitted = fit_reference_image(Image.new("RGBA", (20, 40)), 1)
    assert fitted.mode == "RGB"
    assert min(fitted.size) >= 64


@pytest.mark.parametrize("size", [(49, 49), (21, 50), (63, 200)])
def test_an_upscaled_reference_keeps_the_minimum_side(size: tuple[int, int]) -> None:
    """Flooring the upscaled side gave 63 px for some inputs, which the 16 px crop turned into 48."""
    assert min(fit_reference_image(Image.new("RGB", size), 1).size) >= 64


def test_an_elongated_reference_keeps_both_side_limits() -> None:
    fitted = fit_reference_image(Image.new("RGB", (6000, 100)), 1)
    assert max(fitted.size) <= 4096 and min(fitted.size) >= 64


def test_a_reference_beyond_64_to_1_is_refused() -> None:
    with pytest.raises(ValueError, match="too elongated"):
        fit_reference_image(Image.new("RGB", (8000, 50)), 1)


def test_references_are_normalized_packed_and_numbered_from_minus_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reference i sits at l = -(1 + i): l = 0 is the generated image, so the first reference must not share it."""
    sizes = [(128, 64), (64, 96)]  # (width, height): within the budget and on the 16 px grid, so kept as is
    encoded: list[torch.Tensor] = []

    def fake_encode(vae_info, image: Image.Image, tiling: str) -> torch.Tensor:
        assert tiling == "if_attention_materializes"
        latents = torch.full((1, 32, image.height // 8, image.width // 8), float(len(encoded) + 1))
        encoded.append(latents)
        return latents

    monkeypatch.setattr(ref_images, "encode_reference_image", fake_encode)
    bn_mean, bn_std = torch.full((128,), 0.5), torch.full((128,), 2.0)

    tokens, ids = encode_flux3_references(
        images=[Image.new("RGB", size) for size in sizes],
        vae_info=None,
        bn_mean=bn_mean,
        bn_std=bn_std,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    counts = [(w // 16) * (h // 16) for w, h in sizes]
    assert tokens.shape == (1, sum(counts), 128) and ids.shape == (sum(counts), 4)
    expected = torch.cat([bn_normalize(pack_flux2(latents), bn_mean, bn_std) for latents in encoded], dim=1)
    assert torch.equal(tokens, expected)
    assert ids[: counts[0], 3].unique().tolist() == [-1.0]
    assert ids[counts[0] :, 3].unique().tolist() == [-2.0]
    # Rows and columns follow each reference's own grid: 64 px high, 128 px wide -> 4 x 8 tokens.
    assert ids[: counts[0], 1].max().item() == pytest.approx(1.5)
    assert ids[: counts[0], 2].max().item() == pytest.approx(3.5)
