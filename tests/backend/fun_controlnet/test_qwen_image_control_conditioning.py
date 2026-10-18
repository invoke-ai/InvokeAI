"""The Qwen-Image Fun control context: which channel carries what, and what the inpaint mask does to each.

Expected values are built here from the recipe (VideoX-Fun `pipeline_qwenimage_control.py`), not read back from the
function: the keep mask is `1 - mask` nearest-sampled to latent size, and the masked image has every pixel the mask
marks (>= 0.5) zeroed before it is encoded.
"""

import pytest
import torch
import torch.nn.functional as F

from invokeai.backend.fun_controlnet.conditioning import build_qwen_image_control_context

SIDE = 32
LATENT_SIDE = SIDE // 8
LATENT_CHANNELS = 16


class _Encoder:
    """Stands in for the VAE: an 8x average pool spread over 16 channels, deterministic and shape-faithful. Records
    what it was asked to encode, which is where the masked image is checked."""

    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        self.inputs.append(pixels.clone())
        pooled = F.avg_pool2d(pixels, 8)
        return torch.cat([pooled] * 5 + [pooled[:, :1]], dim=1)


def _image(seed: int) -> torch.Tensor:
    return torch.rand(1, 3, SIDE, SIDE, generator=torch.Generator().manual_seed(seed)) * 2 - 1


def _quadrant_mask() -> torch.Tensor:
    """White (regenerate) in the top-left quarter, black (keep) elsewhere."""
    mask = torch.zeros(1, 1, SIDE, SIDE)
    mask[..., : SIDE // 2, : SIDE // 2] = 1.0
    return mask


def test_pure_control_fills_the_control_channels_and_leaves_the_inpaint_channels_at_zero() -> None:
    encode = _Encoder()
    control_image = _image(0)

    context = build_qwen_image_control_context(encode=encode, control_image=control_image)

    assert context.shape == (1, 2 * LATENT_CHANNELS + 1, LATENT_SIDE, LATENT_SIDE)
    assert torch.equal(context[:, :LATENT_CHANNELS], _Encoder()(control_image))
    assert torch.equal(context[:, LATENT_CHANNELS:], torch.zeros(1, LATENT_CHANNELS + 1, LATENT_SIDE, LATENT_SIDE))
    assert len(encode.inputs) == 1 and torch.equal(encode.inputs[0], control_image)


def test_pure_inpaint_carries_the_keep_mask_and_the_masked_image_and_no_control() -> None:
    encode = _Encoder()
    inpaint_image = _image(1)
    mask = _quadrant_mask()

    context = build_qwen_image_control_context(encode=encode, inpaint_image=inpaint_image, inpaint_mask=mask)

    assert torch.equal(context[:, :LATENT_CHANNELS], torch.zeros(1, LATENT_CHANNELS, LATENT_SIDE, LATENT_SIDE))
    expected_keep = torch.ones(1, 1, LATENT_SIDE, LATENT_SIDE)
    expected_keep[..., : LATENT_SIDE // 2, : LATENT_SIDE // 2] = 0.0
    assert torch.equal(context[:, LATENT_CHANNELS : LATENT_CHANNELS + 1], expected_keep)
    # The encoder saw the inpaint image with the regenerated quarter zeroed and nothing else changed.
    (encoded,) = encode.inputs
    assert torch.equal(encoded[..., : SIDE // 2, : SIDE // 2], torch.zeros(1, 3, SIDE // 2, SIDE // 2))
    assert torch.equal(encoded[..., SIDE // 2 :, :], inpaint_image[..., SIDE // 2 :, :])
    assert torch.equal(encoded[..., :, SIDE // 2 :], inpaint_image[..., :, SIDE // 2 :])
    assert torch.equal(context[:, LATENT_CHANNELS + 1 :], _Encoder()(encoded))


def test_control_and_inpaint_together_fill_both_halves() -> None:
    encode = _Encoder()
    control_image, inpaint_image = _image(2), _image(3)

    context = build_qwen_image_control_context(
        encode=encode, control_image=control_image, inpaint_image=inpaint_image, inpaint_mask=_quadrant_mask()
    )

    assert torch.equal(context[:, :LATENT_CHANNELS], _Encoder()(control_image))
    assert context[:, LATENT_CHANNELS].sum() == LATENT_SIDE * LATENT_SIDE * 3 / 4
    assert torch.equal(context[:, LATENT_CHANNELS + 1 :], _Encoder()(encode.inputs[1]))


def test_a_soft_mask_is_thresholded_before_it_becomes_the_keep_channel() -> None:
    """A blurred mask (0.5, 0.3, 0.9 bands) must give a 0/1 keep channel that agrees with the zeroed pixels: exactly
    the pixels at or above 0.5 are regenerated and read 0 in the channel, the rest are kept and read 1."""
    encode = _Encoder()
    inpaint_image = torch.ones(1, 3, SIDE, SIDE)
    mask = torch.zeros(1, 1, SIDE, SIDE)
    mask[..., :8, :] = 0.5
    mask[..., 8:16, :] = 0.3
    mask[..., 16:24, :] = 0.9

    context = build_qwen_image_control_context(encode=encode, inpaint_image=inpaint_image, inpaint_mask=mask)

    (encoded,) = encode.inputs
    regenerated_rows = torch.zeros(1, 3, 8, SIDE)
    kept_rows = torch.ones(1, 3, 8, SIDE)
    assert torch.equal(encoded[..., :8, :], regenerated_rows)
    assert torch.equal(encoded[..., 8:16, :], kept_rows)
    assert torch.equal(encoded[..., 16:24, :], regenerated_rows)
    assert torch.equal(encoded[..., 24:, :], kept_rows)
    keep = context[0, LATENT_CHANNELS]
    assert set(keep.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(keep, torch.tensor([[0.0], [1.0], [0.0], [1.0]]).expand(LATENT_SIDE, LATENT_SIDE))


def test_the_keep_mask_is_sampled_nearest_not_averaged() -> None:
    """A one-pixel-wide white line must not smear into a grey band at latent resolution."""
    encode = _Encoder()
    mask = torch.zeros(1, 1, SIDE, SIDE)
    mask[..., 8, :] = 1.0

    context = build_qwen_image_control_context(encode=encode, inpaint_image=_image(4), inpaint_mask=mask)

    keep = context[0, LATENT_CHANNELS]
    assert set(keep.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(keep[1], torch.zeros(LATENT_SIDE))
    assert keep.sum() == LATENT_SIDE * (LATENT_SIDE - 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "control image, an inpaint image"),
        ({"inpaint_mask": _quadrant_mask()}, "go together"),
        ({"inpaint_image": _image(5)}, "go together"),
        ({"inpaint_image": _image(5), "inpaint_mask": torch.zeros(1, 1, SIDE, SIDE // 2)}, "size of the inpaint image"),
        (
            {
                "control_image": _image(6),
                "inpaint_image": _image(7)[..., : SIDE // 2],
                "inpaint_mask": _quadrant_mask()[..., : SIDE // 2],
            },
            "generation size",
        ),
    ],
    ids=["nothing", "mask-alone", "image-alone", "mask-size", "frame-mismatch"],
)
def test_incomplete_or_mismatched_inputs_are_refused(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_qwen_image_control_context(encode=_Encoder(), **kwargs)
