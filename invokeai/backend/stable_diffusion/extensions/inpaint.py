from __future__ import annotations

from typing import TYPE_CHECKING

import einops
import torch
from diffusers import UNet2DConditionModel

from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext


class InpaintExt(ExtensionBase):
    def __init__(
        self,
        mask: torch.Tensor,
        is_gradient_mask: bool,
    ):
        super().__init__()
        self.mask = mask
        self.is_gradient_mask = is_gradient_mask

    @staticmethod
    def _is_normal_model(unet: UNet2DConditionModel):
        return unet.conv_in.in_channels == 4

    def _apply_mask(self, ctx: DenoiseContext, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = latents.size(0)
        mask = einops.repeat(self.mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if t.dim() == 0:
            # some schedulers expect t to be one-dimensional.
            # TODO: file diffusers bug about inconsistency?
            t = einops.repeat(t, "-> batch", batch=batch_size)
        # Noise shouldn't be re-randomized between steps here. The multistep schedulers
        # get very confused about what is happening from step to step when we do that.
        mask_latents = ctx.scheduler.add_noise(ctx.inputs.orig_latents, self.noise, t)
        # TODO: Do we need to also apply scheduler.scale_model_input? Or is add_noise appropriately scaled already?
        # mask_latents = self.scheduler.scale_model_input(mask_latents, t)
        mask_latents = einops.repeat(mask_latents, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if self.is_gradient_mask:
            threshhold = (t.item()) / ctx.scheduler.config.num_train_timesteps
            mask_bool = mask > threshhold  # I don't know when mask got inverted, but it did
            masked_input = torch.where(mask_bool, latents, mask_latents)
        else:
            masked_input = torch.lerp(mask_latents.to(dtype=latents.dtype), latents, mask.to(dtype=latents.dtype))
        return masked_input

    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def init_tensors(self, ctx: DenoiseContext):
        if not self._is_normal_model(ctx.unet):
            raise Exception("InpaintExt should be used only on normal models!")

        self.mask = self.mask.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

        self.noise = ctx.inputs.noise
        if self.noise is None:
            self.noise = torch.randn(
                ctx.latents.shape,
                dtype=torch.float32,
                device="cpu",
                generator=torch.Generator(device="cpu").manual_seed(ctx.seed),
            ).to(device=ctx.latents.device, dtype=ctx.latents.dtype)

    # TODO: order value
    @callback(ExtensionCallbackType.PRE_STEP, order=-100)
    def apply_mask_to_initial_latents(self, ctx: DenoiseContext):
        ctx.latents = self._apply_mask(ctx, ctx.latents, ctx.timestep)

    # TODO: order value
    # TODO: redo this with preview events rewrite
    @callback(ExtensionCallbackType.POST_STEP, order=-100)
    def apply_mask_to_step_output(self, ctx: DenoiseContext):
        timestep = ctx.scheduler.timesteps[-1]
        if hasattr(ctx.step_output, "denoised"):
            ctx.step_output.denoised = self._apply_mask(ctx, ctx.step_output.denoised, timestep)
        elif hasattr(ctx.step_output, "pred_original_sample"):
            ctx.step_output.pred_original_sample = self._apply_mask(ctx, ctx.step_output.pred_original_sample, timestep)
        else:
            ctx.step_output.pred_original_sample = self._apply_mask(ctx, ctx.step_output.prev_sample, timestep)

    # TODO: should here be used order?
    # restore unmasked part after the last step is completed
    @callback(ExtensionCallbackType.POST_DENOISE_LOOP)
    def restore_unmasked(self, ctx: DenoiseContext):
        if self.is_gradient_mask:
            ctx.latents = torch.where(self.mask > 0, ctx.latents, ctx.inputs.orig_latents)
        else:
            ctx.latents = torch.lerp(ctx.inputs.orig_latents, ctx.latents, self.mask)
