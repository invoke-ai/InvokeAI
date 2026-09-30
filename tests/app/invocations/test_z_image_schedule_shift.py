"""Z-Image's schedule shift is held at the upper end of the two-point fit it is built on.

The fit runs from 256 tokens to 4096 -- a 1024x1024 frame. Evaluated past that it runs away
(shift 25 at 2048px, 103_777 at 4096px) and `_get_sigmas` flattens with it. Because
`denoising_start` is clipped by index here, the step count survives and the entry point moves
instead, so a pass asked to refine an image starts from noise and generates a new one.

The cells that matter drive the real `_run_diffusion`: the real `_calculate_shift`, the real
`_get_sigmas`, the real clipping and the real img2img blend all run, and the entry sigma is read
back out of the blended tensor rather than recomputed here. A test that recomputed the arithmetic
would agree with a broken node as readily as with a working one.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.z_image.z_image_denoise import (
    Z_IMAGE_MU_FIT_MAX_SEQ_LEN,
    ZImageDenoiseInvocation,
)

# Canvas img2img is the path that reaches this node today. `canvasDenoisingStart` sends
# `1 - strength` for Z-Image (`features/generation/core/canvas/compileCanvasGraph.ts`), so a
# strength of 0.5 arrives as a `denoising_start` of 0.5.
DENOISING_START = 0.5

# The entry sigma at that setting once the shift is held: shift 3.158 applied to t=0.5.
CLAMPED_ENTRY_SIGMA = 0.7595
# The same setting with the shift 2048px used to extrapolate to.
UNCLAMPED_SHIFT_AT_2048 = 25.28
UNCLAMPED_ENTRY_SIGMA = 0.9619


def seq_len(px: int) -> int:
    """Z-Image packs 2x2 patches over an 8x latent, so px/16 tokens per axis."""
    return (px // 16) ** 2


def invocation(**overrides) -> ZImageDenoiseInvocation:
    return ZImageDenoiseInvocation.model_construct(id="test", **overrides)


class _StopAfterBlend(Exception):
    """Ends the run at the first call after the img2img blend."""


def drive_to_blend(px: int, steps: int, shift: float | None = None) -> tuple[float, float]:
    """Run the real `_run_diffusion` as far as the img2img blend.

    Returns the shift `_get_sigmas` was handed, and the entry sigma the blend actually used. The
    input latents are zeros and the noise is ones, so `s * noise + (1 - s) * init` is `s` in every
    element -- the entry sigma falls out of the tensor instead of being derived a second time.

    The latent tensor's size is deliberately unrelated to `px`: the schedule is driven by the
    invocation's `width`/`height`, and a tiny tensor keeps the cell cheap at 4096px.
    """
    node = invocation(
        latents=MagicMock(latents_name="latents"),
        add_noise=True,
        width=px,
        height=px,
        steps=steps,
        denoising_start=DENOISING_START,
        denoising_end=1.0,
        shift=shift,
        positive_conditioning=SimpleNamespace(conditioning_name="positive", mask=None),
        transformer=MagicMock(transformer="transformer"),
        seed=123,
        scheduler="euler",
    )

    seen: dict[str, float] = {}
    real_get_sigmas = node._get_sigmas

    def record_shift(shift_value: float, num_steps: int) -> list[float]:
        seen["shift"] = shift_value
        return real_get_sigmas(shift_value, num_steps)

    captured: dict[str, torch.Tensor] = {}

    def capture(_context, latents):
        captured["latents"] = latents
        raise _StopAfterBlend

    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 16, 8, 8)
    regional_extension = SimpleNamespace(
        regional_text_conditioning=SimpleNamespace(prompt_embeds=torch.zeros(1, 4, 16))
    )

    with (
        patch(
            "invokeai.app.invocations.z_image.z_image_denoise.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ),
        # float32 rather than bfloat16: the entry sigma is read back out of this tensor, and
        # bfloat16 would round it to about three digits.
        patch(
            "invokeai.app.invocations.z_image.z_image_denoise.TorchDevice.choose_bfloat16_safe_dtype",
            return_value=torch.float32,
        ),
        patch("invokeai.app.invocations.z_image.z_image_denoise.ZImageConditioningInfo", object),
        patch(
            "invokeai.app.invocations.z_image.z_image_denoise.ZImageRegionalPromptingExtension.from_text_conditionings",
            return_value=regional_extension,
        ),
        patch.object(
            node,
            "_load_text_conditioning",
            return_value=[SimpleNamespace(prompt_embeds=torch.zeros(1, 4, 16), mask=None)],
        ),
        patch.object(node, "_prepare_noise_tensor", return_value=torch.ones(1, 16, 8, 8)),
        patch.object(node, "_get_sigmas", record_shift),
        patch.object(node, "_prep_inpaint_mask", capture),
        pytest.raises(_StopAfterBlend),
    ):
        node._run_diffusion(mock_context)

    blended = captured["latents"]
    assert torch.allclose(blended, blended.flatten()[:1].expand_as(blended)), (
        "the blend should be uniform, so one element stands for the entry sigma"
    )
    return seen["shift"], float(blended.flatten()[0])


@pytest.mark.parametrize(
    ["px", "expected_shift"],
    [
        (512, 1.878),  # interpolation, below the fit's upper point
        (1024, 3.158),  # 4096 tokens: the upper point, where the fit ends
        (1536, 3.158),  # beyond it, held -- extrapolation would give 7.5
        (2048, 3.158),  # extrapolation would give 25.3
        (3072, 3.158),  # extrapolation would give 809.7
        (4096, 3.158),  # extrapolation would give 103_777
    ],
)
def test_the_shift_is_not_extrapolated_past_the_fit(px: int, expected_shift: float):
    assert invocation()._calculate_shift(seq_len(px)) == pytest.approx(expected_shift, abs=1e-3)


def test_the_fit_is_still_live_below_its_upper_point():
    """Holding the fit at its endpoint is not the same as replacing it with a constant."""
    node = invocation()
    assert node._calculate_shift(256) < node._calculate_shift(1024) < node._calculate_shift(Z_IMAGE_MU_FIT_MAX_SEQ_LEN)


@pytest.mark.parametrize("px", [1024, 1536, 2048, 3072, 4096])
@pytest.mark.parametrize("steps", [8, 30])
def test_the_pass_still_refines_the_input_at_every_frame_size(px: int, steps: int):
    """The defect the clamp exists for, measured through the node's own img2img blend.

    An extrapolated shift pushed the entry sigma to 0.97 at 2048px and 1.0000 at 4096px, so the
    blend kept almost nothing of the input. Both step counts are covered because the shift is a
    claim about the whole schedule: 8 is the node default and the Turbo recommendation, 30 is what
    a canvas pass sends.
    """
    _, entry_sigma = drive_to_blend(px, steps)
    assert entry_sigma == pytest.approx(CLAMPED_ENTRY_SIGMA, abs=1e-3)


def test_an_explicit_shift_overrides_the_clamp():
    """The node exposes `shift` so a user can ask for the pre-clamp behaviour back.

    Both halves go through `_run_diffusion`, so deleting the override in favour of always calling
    `_calculate_shift` fails here rather than passing on a recomputed schedule.
    """
    auto_shift, auto_entry = drive_to_blend(2048, 30)
    forced_shift, forced_entry = drive_to_blend(2048, 30, shift=UNCLAMPED_SHIFT_AT_2048)

    assert auto_shift == pytest.approx(3.158, abs=1e-3)
    assert forced_shift == UNCLAMPED_SHIFT_AT_2048

    assert auto_entry == pytest.approx(CLAMPED_ENTRY_SIGMA, abs=1e-3)
    assert forced_entry == pytest.approx(UNCLAMPED_ENTRY_SIGMA, abs=1e-3)
