from typing import List, Union

import torch
from PIL.Image import Image

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.util.controlnet_utils import CONTROLNET_RESIZE_VALUES, prepare_control_image
from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux import XLabsControlNetFlux, XLabsControlNetFluxOutput
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

        # TODO(ryand): Pass in these params if a new base transformer / XLabs ControlNet pair get released.
        self._flux_transformer_num_double_blocks = 19
        self._flux_transformer_num_single_blocks = 38

    @classmethod
    def prepare_controlnet_cond(
        cls,
        controlnet_image: Image,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
        resize_mode: CONTROLNET_RESIZE_VALUES,
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
            control_mode="balanced",
            resize_mode=resize_mode,
        )

        # Map pixel values from [0, 1] to [-1, 1].
        controlnet_cond = controlnet_cond * 2 - 1

        return controlnet_cond

    @classmethod
    def from_controlnet_image(
        cls,
        model: XLabsControlNetFlux,
        controlnet_image: Image,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
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
            control_mode="balanced",
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

    def _xlabs_output_to_controlnet_output(self, xlabs_output: XLabsControlNetFluxOutput) -> ControlNetFluxOutput:
        # The modulo index logic used here is based on:
        # https://github.com/XLabs-AI/x-flux/blob/47495425dbed499be1e8e5a6e52628b07349cba2/src/flux/model.py#L198-L200

        # Handle double block residuals.
        double_block_residuals: list[torch.Tensor] = []
        xlabs_double_block_residuals = xlabs_output.controlnet_double_block_residuals
        if xlabs_double_block_residuals is not None:
            for i in range(self._flux_transformer_num_double_blocks):
                double_block_residuals.append(xlabs_double_block_residuals[i % len(xlabs_double_block_residuals)])

        return ControlNetFluxOutput(
            double_block_residuals=double_block_residuals,
            single_block_residuals=None,
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
    ) -> ControlNetFluxOutput:
        weight = self._get_weight(timestep_index=timestep_index, total_num_timesteps=total_num_timesteps)
        if weight < 1e-6:
            return ControlNetFluxOutput(single_block_residuals=None, double_block_residuals=None)

        xlabs_output: XLabsControlNetFluxOutput = self._model(
            img=img,
            img_ids=img_ids,
            controlnet_cond=self._controlnet_cond,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=y,
            guidance=guidance,
        )

        controlnet_output = self._xlabs_output_to_controlnet_output(xlabs_output)
        controlnet_output.apply_weight(weight)
        return controlnet_output
