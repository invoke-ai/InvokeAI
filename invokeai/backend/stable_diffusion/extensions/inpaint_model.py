from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from diffusers import UNet2DConditionModel

from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext


class InpaintModelExt(ExtensionBase):
    """An extension for inpainting with inpainting models. See `InpaintExt` for inpainting with non-inpainting
    models.
    """

    def __init__(
        self,
        mask: Optional[torch.Tensor],
        masked_latents: Optional[torch.Tensor],
        is_gradient_mask: bool,
    ):
        """Initialize InpaintModelExt.
        Args:
            mask (Optional[torch.Tensor]): The inpainting mask. Shape: (1, 1, latent_height, latent_width). Values are
                expected to be in the range [0, 1]. A value of 1 means that the corresponding 'pixel' should not be
                inpainted.
            masked_latents (Optional[torch.Tensor]): Latents of initial image, with masked out by black color inpainted area.
                If mask provided, then too should be provided. Shape: (1, 1, latent_height, latent_width)
            is_gradient_mask (bool): If True, mask is interpreted as a gradient mask meaning that the mask values range
                from 0 to 1. If False, mask is interpreted as binary mask meaning that the mask values are either 0 or
                1.
        """
        super().__init__()
        if mask is not None and masked_latents is None:
            raise ValueError("Source image required for inpaint mask when inpaint model used!")

        # Inverse mask, because inpaint models treat mask as: 0 - remain same, 1 - inpaint
        self._mask = None
        if mask is not None:
            self._mask = 1 - mask
        self._masked_latents = masked_latents
        self._is_gradient_mask = is_gradient_mask

    @staticmethod
    def _is_inpaint_model(unet: UNet2DConditionModel):
        """Checks if the provided UNet belongs to a regular model.
        The `in_channels` of a UNet vary depending on model type:
        - normal - 4
        - depth - 5
        - inpaint - 9
        """
        return unet.conv_in.in_channels == 9

    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def init_tensors(self, ctx: DenoiseContext):
        if not self._is_inpaint_model(ctx.unet):
            raise ValueError("InpaintModelExt should be used only on inpaint models!")

        if self._mask is None:
            self._mask = torch.ones_like(ctx.latents[:1, :1])
        self._mask = self._mask.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

        if self._masked_latents is None:
            self._masked_latents = torch.zeros_like(ctx.latents[:1])
        self._masked_latents = self._masked_latents.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

    # Do last so that other extensions works with normal latents
    @callback(ExtensionCallbackType.PRE_UNET, order=1000)
    def append_inpaint_layers(self, ctx: DenoiseContext):
        batch_size = ctx.unet_kwargs.sample.shape[0]
        b_mask = torch.cat([self._mask] * batch_size)
        b_masked_latents = torch.cat([self._masked_latents] * batch_size)
        ctx.unet_kwargs.sample = torch.cat(
            [ctx.unet_kwargs.sample, b_mask, b_masked_latents],
            dim=1,
        )

    # Restore unmasked part as inpaint model can change unmasked part slightly
    @callback(ExtensionCallbackType.POST_DENOISE_LOOP)
    def restore_unmasked(self, ctx: DenoiseContext):
        if self._is_gradient_mask:
            ctx.latents = torch.where(self._mask > 0, ctx.latents, ctx.inputs.orig_latents)
        else:
            ctx.latents = torch.lerp(ctx.inputs.orig_latents, ctx.latents, self._mask)
