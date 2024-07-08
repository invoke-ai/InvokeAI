from __future__ import annotations

import math
import torch
from PIL.Image import Image
from contextlib import contextmanager
from typing import TYPE_CHECKING, Union, List
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, modifier
from invokeai.backend.stable_diffusion.denoise_context import UNetKwargs

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.util.hotfixes import ControlNetModel


class ControlNetExt(ExtensionBase):
    def __init__(
        self,
        model: ControlNetModel,
        image: Image,
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
        control_mode: str,
        resize_mode: str,
        priority: int,
    ):
        super().__init__(priority=priority)
        self.model = model
        self.image = image
        self.weight = weight
        self.begin_step_percent = begin_step_percent
        self.end_step_percent = end_step_percent
        self.control_mode = control_mode
        self.resize_mode = resize_mode

        self.image_tensor: Optional[torch.Tensor] = None

    @contextmanager
    def patch_attention_processor(self, attention_processor_cls):
        try:
            original_processors = self.model.attn_processors
            self.model.set_attn_processor(attention_processor_cls())

            yield None
        finally:
            self.model.set_attn_processor(original_processors)

    @modifier("pre_denoise_loop")
    def resize_image(self, ctx: DenoiseContext):
        _, _, latent_height, latent_width = ctx.latents.shape
        image_height = latent_height * LATENT_SCALE_FACTOR
        image_width = latent_width * LATENT_SCALE_FACTOR

        from invokeai.app.util.controlnet_utils import prepare_control_image
        self.image_tensor = prepare_control_image(
            image=self.image,
            do_classifier_free_guidance=True,
            width=image_width,
            height=image_height,
            # batch_size=batch_size * num_images_per_prompt,
            # num_images_per_prompt=num_images_per_prompt,
            device=ctx.latents.device,
            dtype=ctx.latents.dtype,
            control_mode=self.control_mode,
            resize_mode=self.resize_mode,
        )

    @modifier("pre_unet_forward")
    def pre_unet_step(self, ctx: DenoiseContext):
        # skip if model not active in current step
        total_steps = len(ctx.timesteps)
        first_step = math.floor(self.begin_step_percent * total_steps)
        last_step  = math.ceil(self.end_step_percent * total_steps)
        if ctx.step_index < first_step or ctx.step_index > last_step:
            return

        # convert mode to internal flags
        soft_injection = self.control_mode in ["more_prompt", "more_control"]
        cfg_injection  = self.control_mode in ["more_control", "unbalanced"]

        # no negative conditioning in cfg_injection mode
        if cfg_injection:
            if ctx.conditioning_mode == "negative":
                return
            down_samples, mid_sample = self._run(ctx, soft_injection, "positive")

            if ctx.conditioning_mode == "both":
                # add zeros as samples for negative conditioning
                down_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_samples]
                mid_sample = torch.cat([torch.zeros_like(mid_sample), mid_sample])

        else:
            down_samples, mid_sample = self._run(ctx, soft_injection, ctx.conditioning_mode)


        if ctx.unet_kwargs.down_block_additional_residuals is None and ctx.unet_kwargs.mid_block_additional_residual is None:
            ctx.unet_kwargs.down_block_additional_residuals, ctx.unet_kwargs.mid_block_additional_residual = down_samples, mid_sample
        else:
            # add controlnet outputs together if have multiple controlnets
            ctx.unet_kwargs.down_block_additional_residuals = [
                samples_prev + samples_curr
                for samples_prev, samples_curr in zip(ctx.unet_kwargs.down_block_additional_residuals, down_samples, strict=True)
            ]
            ctx.unet_kwargs.mid_block_additional_residual += mid_sample

    def _run(self, ctx: DenoiseContext, soft_injection, conditioning_mode):
        total_steps = len(ctx.timesteps)
        model_input = ctx.latent_model_input
        if conditioning_mode == "both":
            model_input = torch.cat([model_input] * 2)

        cn_unet_kwargs = UNetKwargs(
            sample=model_input,
            timestep=ctx.timestep,
            encoder_hidden_states=None, # set later by conditoning

            cross_attention_kwargs=dict(
                percent_through=ctx.step_index / total_steps,
            ),
        )

        ctx.conditioning_data.to_unet_kwargs(cn_unet_kwargs, conditioning_mode=conditioning_mode)

        # get static weight, or weight corresponding to current step
        weight = self.weight
        if isinstance(weight, list):
            weight = weight[ctx.step_index]

        tmp_kwargs = vars(cn_unet_kwargs)
        tmp_kwargs.pop("down_block_additional_residuals", None)
        tmp_kwargs.pop("mid_block_additional_residual", None)
        tmp_kwargs.pop("down_intrablock_additional_residuals", None)

        image_tensor = self.image_tensor
        tile_coords = ctx.extra.get("tile_coords", None)
        if tile_coords is not None:
            image_tensor = image_tensor[
                :,
                :,
                tile_coords.top*LATENT_SCALE_FACTOR:tile_coords.bottom*LATENT_SCALE_FACTOR,
                tile_coords.left*LATENT_SCALE_FACTOR:tile_coords.right*LATENT_SCALE_FACTOR
            ]

        # controlnet(s) inference
        down_samples, mid_sample = self.model(
            controlnet_cond=image_tensor,
            conditioning_scale=weight,  # controlnet specific, NOT the guidance scale
            guess_mode=soft_injection,  # this is still called guess_mode in diffusers ControlNetModel
            return_dict=False,
            **vars(cn_unet_kwargs),
        )

        return down_samples, mid_sample
