from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from PIL.Image import Image

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.util.controlnet_utils import CONTROLNET_MODE_VALUES, CONTROLNET_RESIZE_VALUES, prepare_control_image
from invokeai.backend.stable_diffusion.denoise_context import UNetKwargs
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback

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
        control_mode: CONTROLNET_MODE_VALUES,
        resize_mode: CONTROLNET_RESIZE_VALUES,
    ):
        super().__init__()
        self._model = model
        self._image = image
        self._weight = weight
        self._begin_step_percent = begin_step_percent
        self._end_step_percent = end_step_percent
        self._control_mode = control_mode
        self._resize_mode = resize_mode

        self._image_tensor: Optional[torch.Tensor] = None

    @contextmanager
    def patch_extension(self, ctx: DenoiseContext):
        original_processors = self._model.attn_processors
        try:
            self._model.set_attn_processor(ctx.inputs.attention_processor_cls())

            yield None
        finally:
            self._model.set_attn_processor(original_processors)

    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def resize_image(self, ctx: DenoiseContext):
        _, _, latent_height, latent_width = ctx.latents.shape
        image_height = latent_height * LATENT_SCALE_FACTOR
        image_width = latent_width * LATENT_SCALE_FACTOR

        self._image_tensor = prepare_control_image(
            image=self._image,
            do_classifier_free_guidance=False,
            width=image_width,
            height=image_height,
            device=ctx.latents.device,
            dtype=ctx.latents.dtype,
            control_mode=self._control_mode,
            resize_mode=self._resize_mode,
        )

    @callback(ExtensionCallbackType.PRE_UNET)
    def pre_unet_step(self, ctx: DenoiseContext):
        # skip if model not active in current step
        total_steps = len(ctx.inputs.timesteps)
        first_step = math.floor(self._begin_step_percent * total_steps)
        last_step = math.ceil(self._end_step_percent * total_steps)
        if ctx.step_index < first_step or ctx.step_index > last_step:
            return

        # convert mode to internal flags
        soft_injection = self._control_mode in ["more_prompt", "more_control"]
        cfg_injection = self._control_mode in ["more_control", "unbalanced"]

        # no negative conditioning in cfg_injection mode
        if cfg_injection:
            if ctx.conditioning_mode == ConditioningMode.Negative:
                return
            down_samples, mid_sample = self._run(ctx, soft_injection, ConditioningMode.Positive)

            if ctx.conditioning_mode == ConditioningMode.Both:
                # add zeros as samples for negative conditioning
                down_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_samples]
                mid_sample = torch.cat([torch.zeros_like(mid_sample), mid_sample])

        else:
            down_samples, mid_sample = self._run(ctx, soft_injection, ctx.conditioning_mode)

        if (
            ctx.unet_kwargs.down_block_additional_residuals is None
            and ctx.unet_kwargs.mid_block_additional_residual is None
        ):
            ctx.unet_kwargs.down_block_additional_residuals = down_samples
            ctx.unet_kwargs.mid_block_additional_residual = mid_sample
        else:
            # add controlnet outputs together if have multiple controlnets
            ctx.unet_kwargs.down_block_additional_residuals = [
                samples_prev + samples_curr
                for samples_prev, samples_curr in zip(
                    ctx.unet_kwargs.down_block_additional_residuals, down_samples, strict=True
                )
            ]
            ctx.unet_kwargs.mid_block_additional_residual += mid_sample

    def _run(self, ctx: DenoiseContext, soft_injection: bool, conditioning_mode: ConditioningMode):
        total_steps = len(ctx.inputs.timesteps)

        model_input = ctx.latent_model_input
        image_tensor = self._image_tensor
        if conditioning_mode == ConditioningMode.Both:
            model_input = torch.cat([model_input] * 2)
            image_tensor = torch.cat([image_tensor] * 2)

        cn_unet_kwargs = UNetKwargs(
            sample=model_input,
            timestep=ctx.timestep,
            encoder_hidden_states=None,  # set later by conditioning
            cross_attention_kwargs=dict(  # noqa: C408
                percent_through=ctx.step_index / total_steps,
            ),
        )

        ctx.inputs.conditioning_data.to_unet_kwargs(cn_unet_kwargs, conditioning_mode=conditioning_mode)

        # get static weight, or weight corresponding to current step
        weight = self._weight
        if isinstance(weight, list):
            weight = weight[ctx.step_index]

        tmp_kwargs = vars(cn_unet_kwargs)

        # Remove kwargs not related to ControlNet unet
        # ControlNet guidance fields
        del tmp_kwargs["down_block_additional_residuals"]
        del tmp_kwargs["mid_block_additional_residual"]

        # T2i Adapter guidance fields
        del tmp_kwargs["down_intrablock_additional_residuals"]

        # controlnet(s) inference
        down_samples, mid_sample = self._model(
            controlnet_cond=image_tensor,
            conditioning_scale=weight,  # controlnet specific, NOT the guidance scale
            guess_mode=soft_injection,  # this is still called guess_mode in diffusers ControlNetModel
            return_dict=False,
            **vars(cn_unet_kwargs),
        )

        return down_samples, mid_sample
