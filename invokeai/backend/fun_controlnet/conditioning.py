from collections.abc import Callable

import torch
import torch.nn.functional as F

Encoder = Callable[[torch.Tensor], torch.Tensor]
"""Encodes pixels ``(1, 3, H, W)`` in [-1, 1] to the normalized latents ``(1, C, H/8, W/8)`` the denoise works in."""


def build_qwen_image_control_context(
    *,
    encode: Encoder,
    control_image: torch.Tensor | None = None,
    inpaint_image: torch.Tensor | None = None,
    inpaint_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The latent-space control context ``(1, 2C + 1, h, w)``: ``[control latents | keep mask | masked-image latents]``.

    VideoX-Fun ``pipeline_qwenimage_control.py``: the control image is encoded like the generation latents; the inpaint
    mask ``(1, 1, H, W)`` in [0, 1] is thresholded at 0.5 (white is regenerated) and its complement nearest-sampled to
    latent resolution as the keep mask, so the channel is exactly 0 or 1 even for a blurred mask; the inpaint image is
    encoded with the regenerated pixels zeroed. Pure control leaves the inpaint channels at zero, pure inpaint the
    control latents; the union checkpoint reads both. The denoise packs the result 2x2 the way it packs its latents.
    """
    if (inpaint_image is None) != (inpaint_mask is None):
        raise ValueError("An inpaint image and its mask go together.")
    if control_image is None and inpaint_image is None:
        raise ValueError("A control context needs a control image, an inpaint image with its mask, or both.")

    control_latents = encode(control_image) if control_image is not None else None
    masked_latents = keep_mask = None
    if inpaint_image is not None and inpaint_mask is not None:
        if inpaint_mask.shape[-2:] != inpaint_image.shape[-2:]:
            raise ValueError(
                f"The inpaint mask {tuple(inpaint_mask.shape[-2:])} must be the size of the inpaint image "
                f"{tuple(inpaint_image.shape[-2:])}."
            )
        regenerate = inpaint_mask >= 0.5
        masked_latents = encode(inpaint_image * (~regenerate).to(inpaint_image.dtype))
        keep_mask = F.interpolate(
            (~regenerate).to(device=masked_latents.device, dtype=masked_latents.dtype),
            size=masked_latents.shape[-2:],
            mode="nearest",
        )
    if control_latents is not None and masked_latents is not None and control_latents.shape != masked_latents.shape:
        raise ValueError(
            f"The control image encodes to {tuple(control_latents.shape)} and the inpaint image to "
            f"{tuple(masked_latents.shape)}; both must be the generation size."
        )

    reference = control_latents if control_latents is not None else masked_latents
    assert reference is not None
    batch, channels, height, width = reference.shape
    return torch.cat(
        [
            control_latents if control_latents is not None else reference.new_zeros((batch, channels, height, width)),
            keep_mask if keep_mask is not None else reference.new_zeros((batch, 1, height, width)),
            masked_latents if masked_latents is not None else reference.new_zeros((batch, channels, height, width)),
        ],
        dim=1,
    )
