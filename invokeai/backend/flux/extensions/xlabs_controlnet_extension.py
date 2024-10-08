from typing import List, Union

import torch
from PIL.Image import Image

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.util.controlnet_utils import CONTROLNET_MODE_VALUES, CONTROLNET_RESIZE_VALUES, prepare_control_image
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux import XLabsControlNetFlux
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux_output import XLabsControlNetFluxOutput
from invokeai.backend.flux.extensions.base_controlnet_extension import BaseControlNetExtension


class XLabsControlNetExtension(BaseControlNetExtension):
    def __init__(
        self,
        model: XLabsControlNetFlux,
        controlnet_cond: torch.Tensor,
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
    ):
        super().__init__(
            weight=weight,
            begin_step_percent=begin_step_percent,
            end_step_percent=end_step_percent,
        )

        self._model = model
        # _controlnet_cond is the control image passed to the ControlNet model.
        # Pixel values are in the range [-1, 1]. Shape: (batch_size, 3, height, width).
        self._controlnet_cond = controlnet_cond

    @classmethod
    def from_controlnet_image(
        cls,
        model: XLabsControlNetFlux,
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
    ) -> XLabsControlNetFluxOutput | None:
        weight = self._get_weight(timestep_index=timestep_index, total_num_timesteps=total_num_timesteps)
        if weight < 1e-6:
            return None

        output: XLabsControlNetFluxOutput = self._model(
            img=img,
            img_ids=img_ids,
            controlnet_cond=self._controlnet_cond,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=y,
            guidance=guidance,
        )

        output.apply_weight(weight)
        return output
