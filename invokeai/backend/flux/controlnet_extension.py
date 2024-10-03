import math
from typing import List, Union

import torch
from PIL.Image import Image

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.util.controlnet_utils import CONTROLNET_MODE_VALUES, CONTROLNET_RESIZE_VALUES, prepare_control_image
from invokeai.backend.flux.controlnet.controlnet_flux import ControlNetFlux


class ControlNetExtension:
    def __init__(
        self,
        model: ControlNetFlux,
        controlnet_cond: torch.Tensor,
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
    ):
        self._model = model
        # _controlnet_cond is the control image passed to the ControlNet model.
        # Pixel values are in the range [-1, 1]. Shape: (batch_size, 3, height, width).
        self._controlnet_cond = controlnet_cond

        self._weight = weight
        self._begin_step_percent = begin_step_percent
        self._end_step_percent = end_step_percent

    @classmethod
    def from_controlnet_image(
        cls,
        model: ControlNetFlux,
        controlnet_image: Image,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
        control_mode: CONTROLNET_MODE_VALUES,
        resize_mode: CONTROLNET_RESIZE_VALUES,
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
    ):
        image_height = latent_height * LATENT_SCALE_FACTOR
        image_width = latent_width * LATENT_SCALE_FACTOR

        controlnet_cond = prepare_control_image(
            image=controlnet_image,
            do_classifier_free_guidance=False,
            width=image_width,
            height=image_height,
            device=device,
            dtype=dtype,
            control_mode=control_mode,
            resize_mode=resize_mode,
        )

        # Map pixel values from [0, 1] to [-1, 1].
        controlnet_cond = controlnet_cond * 2 - 1

        return cls(
            model=model,
            controlnet_cond=controlnet_cond,
            weight=weight,
            begin_step_percent=begin_step_percent,
            end_step_percent=end_step_percent,
        )

    def run_controlnet(
        self,
        timestep_index: int,
        total_num_timesteps: int,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        y: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> list[torch.Tensor] | None:
        first_step = math.floor(self._begin_step_percent * total_num_timesteps)
        last_step = math.ceil(self._end_step_percent * total_num_timesteps)
        if timestep_index < first_step or timestep_index > last_step:
            return
        weight = self._weight

        controlnet_block_res_samples = self._model(
            img=img,
            img_ids=img_ids,
            controlnet_cond=self._controlnet_cond,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=y,
            guidance=guidance,
        )

        # Apply weight to the residuals.
        for block_res_sample in controlnet_block_res_samples:
            block_res_sample *= weight

        return controlnet_block_res_samples
