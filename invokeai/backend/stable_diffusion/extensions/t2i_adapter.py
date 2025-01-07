from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from diffusers import T2IAdapter
from PIL.Image import Image

from invokeai.app.util.controlnet_utils import prepare_control_image
from invokeai.backend.model_manager import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.backend.util.devices import TorchDevice

if TYPE_CHECKING:
    from invokeai.app.invocations.model import ModelIdentifierField
    from invokeai.app.services.shared.invocation_context import InvocationContext
    from invokeai.app.util.controlnet_utils import CONTROLNET_RESIZE_VALUES
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext


class T2IAdapterExt(ExtensionBase):
    def __init__(
        self,
        node_context: InvocationContext,
        model_id: ModelIdentifierField,
        image: Image,
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
        resize_mode: CONTROLNET_RESIZE_VALUES,
    ):
        super().__init__()
        self._node_context = node_context
        self._model_id = model_id
        self._image = image
        self._weight = weight
        self._resize_mode = resize_mode
        self._begin_step_percent = begin_step_percent
        self._end_step_percent = end_step_percent

        self._adapter_state: Optional[List[torch.Tensor]] = None

        # The max_unet_downscale is the maximum amount that the UNet model downscales the latent image internally.
        model_config = self._node_context.models.get_config(self._model_id.key)
        if model_config.base == BaseModelType.StableDiffusion1:
            self._max_unet_downscale = 8
        elif model_config.base == BaseModelType.StableDiffusionXL:
            self._max_unet_downscale = 4
        else:
            raise ValueError(f"Unexpected T2I-Adapter base model type: '{model_config.base}'.")

    @callback(ExtensionCallbackType.SETUP)
    def setup(self, ctx: DenoiseContext):
        t2i_model: T2IAdapter
        with self._node_context.models.load(self._model_id) as t2i_model:
            _, _, latents_height, latents_width = ctx.inputs.orig_latents.shape

            self._adapter_state = self._run_model(
                model=t2i_model,
                image=self._image,
                latents_height=latents_height,
                latents_width=latents_width,
            )

    def _run_model(
        self,
        model: T2IAdapter,
        image: Image,
        latents_height: int,
        latents_width: int,
    ):
        # Resize the T2I-Adapter input image.
        # We select the resize dimensions so that after the T2I-Adapter's total_downscale_factor is applied, the
        # result will match the latent image's dimensions after max_unet_downscale is applied.
        input_height = latents_height // self._max_unet_downscale * model.total_downscale_factor
        input_width = latents_width // self._max_unet_downscale * model.total_downscale_factor

        # Note: We have hard-coded `do_classifier_free_guidance=False`. This is because we only want to prepare
        # a single image. If CFG is enabled, we will duplicate the resultant tensor after applying the
        # T2I-Adapter model.
        #
        # Note: We re-use the `prepare_control_image(...)` from ControlNet for T2I-Adapter, because it has many
        # of the same requirements (e.g. preserving binary masks during resize).
        t2i_image = prepare_control_image(
            image=image,
            do_classifier_free_guidance=False,
            width=input_width,
            height=input_height,
            num_channels=model.config["in_channels"],
            device=TorchDevice.choose_torch_device(),
            dtype=model.dtype,
            resize_mode=self._resize_mode,
        )

        return model(t2i_image)

    @callback(ExtensionCallbackType.PRE_UNET)
    def pre_unet_step(self, ctx: DenoiseContext):
        # skip if model not active in current step
        total_steps = len(ctx.inputs.timesteps)
        first_step = math.floor(self._begin_step_percent * total_steps)
        last_step = math.ceil(self._end_step_percent * total_steps)
        if ctx.step_index < first_step or ctx.step_index > last_step:
            return

        weight = self._weight
        if isinstance(weight, list):
            weight = weight[ctx.step_index]

        adapter_state = self._adapter_state
        if ctx.conditioning_mode == ConditioningMode.Both:
            adapter_state = [torch.cat([v] * 2) for v in adapter_state]

        if ctx.unet_kwargs.down_intrablock_additional_residuals is None:
            ctx.unet_kwargs.down_intrablock_additional_residuals = [v * weight for v in adapter_state]
        else:
            for i, value in enumerate(adapter_state):
                ctx.unet_kwargs.down_intrablock_additional_residuals[i] += value * weight
