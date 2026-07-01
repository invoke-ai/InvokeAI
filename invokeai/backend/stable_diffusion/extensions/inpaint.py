from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import einops
import torch
from diffusers import UNet2DConditionModel

from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext


class InpaintExt(ExtensionBase):
    """An extension for inpainting with non-inpainting models. See `InpaintModelExt` for inpainting with inpainting
    models.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        is_gradient_mask: bool,
    ):
        """Initialize InpaintExt.
        Args:
            mask (torch.Tensor): The inpainting mask. Shape: (1, 1, latent_height, latent_width). Values are
                expected to be in the range [0, 1]. A value of 1 means that the corresponding 'pixel' should not be
                inpainted.
            is_gradient_mask (bool): If True, mask is interpreted as a gradient mask meaning that the mask values range
                from 0 to 1. If False, mask is interpreted as binary mask meaning that the mask values are either 0 or
                1.
        """
        super().__init__()
        self._mask = mask
        self._is_gradient_mask = is_gradient_mask

        # Noise, which used to noisify unmasked part of image
        # if noise provided to context, then it will be used
        # if no noise provided, then noise will be generated based on seed
        self._noise: Optional[torch.Tensor] = None

    @staticmethod
    def _is_normal_model(unet: UNet2DConditionModel):
        """Checks if the provided UNet belongs to a regular model.
        The `in_channels` of a UNet vary depending on model type:
        - normal - 4
        - depth - 5
        - inpaint - 9
        """
        return unet.conv_in.in_channels == 4

    def _apply_mask(self, ctx: DenoiseContext, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = latents.size(0)
        mask = einops.repeat(self._mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if t.dim() == 0:
            # some schedulers expect t to be one-dimensional.
            # TODO: file diffusers bug about inconsistency?
            t = einops.repeat(t, "-> batch", batch=batch_size)
        # Noise shouldn't be re-randomized between steps here. The multistep schedulers
        # get very confused about what is happening from step to step when we do that.
        mask_latents = ctx.scheduler.add_noise(ctx.inputs.orig_latents, self._noise, t)
        # TODO: Do we need to also apply scheduler.scale_model_input? Or is add_noise appropriately scaled already?
        # mask_latents = self.scheduler.scale_model_input(mask_latents, t)
        mask_latents = einops.repeat(mask_latents, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if self._is_gradient_mask:
            threshold = (t.item()) / ctx.scheduler.config.num_train_timesteps
            mask_bool = mask < 1 - threshold
            masked_input = torch.where(mask_bool, latents, mask_latents)
        else:
            masked_input = torch.lerp(latents, mask_latents.to(dtype=latents.dtype), mask.to(dtype=latents.dtype))
        return masked_input

    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def init_tensors(self, ctx: DenoiseContext):
        if not self._is_normal_model(ctx.unet):
            raise ValueError(
                "InpaintExt should be used only on normal (non-inpainting) models. This could be caused by an "
                "inpainting model that was incorrectly marked as a non-inpainting model. In some cases, this can be "
                "fixed by removing and re-adding the model (so that it gets re-probed)."
            )

        self._mask = self._mask.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

        self._noise = ctx.inputs.noise
        # 'noise' might be None if the latents have already been noised (e.g. when running the SDXL refiner).
        # We still need noise for inpainting, so we generate it from the seed here.
        if self._noise is None:
            self._noise = torch.randn(
                ctx.latents.shape,
                dtype=torch.float32,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(ctx.seed),
            ).to(device=ctx.latents.device, dtype=ctx.latents.dtype)

    # Use negative order to make extensions with default order work with patched latents
    @callback(ExtensionCallbackType.PRE_STEP, order=-100)
    def apply_mask_to_initial_latents(self, ctx: DenoiseContext):
        ctx.latents = self._apply_mask(ctx, ctx.latents, ctx.timestep)

    # TODO: redo this with preview events rewrite
    # Use negative order to make extensions with default order work with patched latents
    @callback(ExtensionCallbackType.POST_STEP, order=-100)
    def apply_mask_to_step_output(self, ctx: DenoiseContext):
        timestep = ctx.scheduler.timesteps[-1]
        if hasattr(ctx.step_output, "denoised"):
            ctx.step_output.denoised = self._apply_mask(ctx, ctx.step_output.denoised, timestep)
        elif hasattr(ctx.step_output, "pred_original_sample"):
            ctx.step_output.pred_original_sample = self._apply_mask(ctx, ctx.step_output.pred_original_sample, timestep)
        else:
            ctx.step_output.pred_original_sample = self._apply_mask(ctx, ctx.step_output.prev_sample, timestep)

    # Restore unmasked part after the last step is completed
    @callback(ExtensionCallbackType.POST_DENOISE_LOOP)
    def restore_unmasked(self, ctx: DenoiseContext):
        if self._is_gradient_mask:
            ctx.latents = torch.where(self._mask < 1, ctx.latents, ctx.inputs.orig_latents)
        else:
            ctx.latents = torch.lerp(ctx.latents, ctx.inputs.orig_latents, self._mask)
