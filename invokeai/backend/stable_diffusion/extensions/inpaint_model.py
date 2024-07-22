from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from diffusers import UNet2DConditionModel

from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext


class InpaintModelExt(ExtensionBase):
    def __init__(
        self,
        mask: Optional[torch.Tensor],
        masked_latents: Optional[torch.Tensor],
        is_gradient_mask: bool,
    ):
        super().__init__()
        if mask is not None and masked_latents is None:
            raise ValueError("Source image required for inpaint mask when inpaint model used!")

        self._mask = mask
        self._masked_latents = masked_latents
        self._is_gradient_mask = is_gradient_mask

    @staticmethod
    def _is_inpaint_model(unet: UNet2DConditionModel):
        return unet.conv_in.in_channels == 9

    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def init_tensors(self, ctx: DenoiseContext):
        if not self._is_inpaint_model(ctx.unet):
            raise Exception("InpaintModelExt should be used only on inpaint models!")

        if self._mask is None:
            self._mask = torch.ones_like(ctx.latents[:1, :1])
        self._mask = self._mask.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

        if self._masked_latents is None:
            self._masked_latents = torch.zeros_like(ctx.latents[:1])
        self._masked_latents = self._masked_latents.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

    # TODO: any ideas about order value?
    # do last so that other extensions works with normal latents
    @callback(ExtensionCallbackType.PRE_UNET, order=1000)
    def append_inpaint_layers(self, ctx: DenoiseContext):
        batch_size = ctx.unet_kwargs.sample.shape[0]
        b_mask = torch.cat([self._mask] * batch_size)
        b_masked_latents = torch.cat([self._masked_latents] * batch_size)
        ctx.unet_kwargs.sample = torch.cat(
            [ctx.unet_kwargs.sample, b_mask, b_masked_latents],
            dim=1,
        )

    # TODO: should here be used order?
    # restore unmasked part as inpaint model can change unmasked part slightly
    @callback(ExtensionCallbackType.POST_DENOISE_LOOP)
    def restore_unmasked(self, ctx: DenoiseContext):
        if self._is_gradient_mask:
            ctx.latents = torch.where(self._mask > 0, ctx.latents, ctx.inputs.orig_latents)
        else:
            ctx.latents = torch.lerp(ctx.inputs.orig_latents, ctx.latents, self._mask)
