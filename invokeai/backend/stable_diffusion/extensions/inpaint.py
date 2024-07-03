import torch
import einops
from typing import Optional
from .base import ExtensionBase, modifier
from ..denoise_context import DenoiseContext
from diffusers import UNet2DConditionModel


class InpaintExt(ExtensionBase):
    def __init__(
        self,
        mask: Optional[torch.Tensor],
        masked_latents: Optional[torch.Tensor],
        is_gradient_mask: bool,
        priority: int,
    ):
        super().__init__(priority=priority)
        self.mask = mask
        self.masked_latents = masked_latents
        self.is_gradient_mask = is_gradient_mask
        self.noise = None

    def _is_inpaint_model(self, unet: UNet2DConditionModel):
        return unet.conv_in.in_channels == 9

    def _apply_mask(self, ctx: DenoiseContext, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = latents.size(0)
        mask = einops.repeat(self.mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if t.dim() == 0:
            # some schedulers expect t to be one-dimensional.
            # TODO: file diffusers bug about inconsistency?
            t = einops.repeat(t, "-> batch", batch=batch_size)
        # Noise shouldn't be re-randomized between steps here. The multistep schedulers
        # get very confused about what is happening from step to step when we do that.
        mask_latents = ctx.scheduler.add_noise(ctx.orig_latents, self.noise, t)
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


    @modifier("pre_denoise_loop")
    def init_tensors(self, ctx: DenoiseContext):
        if self._is_inpaint_model(ctx.unet):
            if self.mask is None:
                self.mask = torch.ones_like(ctx.latents[:1, :1])
            self.mask.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

            if self.masked_latents is None:
                self.masked_latents = torch.zeros_like(ctx.latents[:1])
            self.masked_latents.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

        else:
            #self.orig_latents = ctx.orig_latents
            self.noise = ctx.noise
            if self.noise is None:
                self.noise = torch.randn(
                    ctx.orig_latents.shape,
                    dtype=torch.float32,
                    device="cpu",
                    generator=torch.Generator(device="cpu").manual_seed(ctx.seed),
                ).to(device=ctx.orig_latents.device, dtype=ctx.orig_latents.dtype)

    # do first to make other extensions works with changed latents
    @modifier("pre_step", order="first")
    def apply_mask_to_latents(self, ctx: DenoiseContext):
        if self._is_inpaint_model(ctx.unet) or self.mask is None:
            return
        ctx.latents = self._apply_mask(ctx, ctx.latents, ctx.timestep)

    # do last so that other extensions works with normal latents
    @modifier("pre_unet_forward", order="last")
    def append_inpaint_layers(self, ctx: DenoiseContext):
        if not self._is_inpaint_model(ctx.unet):
            return

        batch_size = ctx.unet_kwargs.sample.shape[0]
        b_mask = torch.cat([self.mask] * batch_size)
        b_masked_latents = torch.cat([self.masked_latents] * batch_size)
        ctx.unet_kwargs.sample = torch.cat(
            [ctx.unet_kwargs.sample, b_mask, b_masked_latents],
            dim=1,
        )

    @modifier("post_step", order="first")
    def apply_mask_to_preview(self, ctx: DenoiseContext):
        if self._is_inpaint_model(ctx.unet) or self.mask is None:
            return

        timestep = ctx.scheduler.timesteps[-1]
        if hasattr(ctx.step_output, "denoised"):
            ctx.step_output.denoised = self._apply_mask(ctx, ctx.step_output.denoised, timestep)
        elif hasattr(ctx.step_output, "pred_original_sample"):
            ctx.step_output.pred_original_sample = self._apply_mask(ctx, ctx.step_output.pred_original_sample, timestep)
        else:
            ctx.step_output.pred_original_sample = self._apply_mask(ctx, ctx.step_output.prev_sample, timestep)

    @modifier("post_denoise_loop") # last?
    def restore_unmasked(self, ctx: DenoiseContext):
        if self.mask is None:
            return

        # restore unmasked part after the last step is completed
        # in-process masking happens before each step
        if self.is_gradient_mask:
            ctx.latents = torch.where(self.mask > 0, ctx.latents, ctx.orig_latents)
        else:
            ctx.latents = torch.lerp(
                ctx.orig_latents,
                ctx.latents.to(dtype=ctx.orig_latents.dtype),
                self.mask.to(dtype=ctx.orig_latents.dtype),
            )
