from typing import List, Union

import torch
from PIL.Image import Image

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.app.util.controlnet_utils import CONTROLNET_MODE_VALUES, CONTROLNET_RESIZE_VALUES, prepare_control_image
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import (
    InstantXControlNetFlux,
    InstantXControlNetFluxOutput,
)
from invokeai.backend.flux.extensions.base_controlnet_extension import BaseControlNetExtension
from invokeai.backend.model_manager.load.load_base import LoadedModel


class InstantXControlNetExtension(BaseControlNetExtension):
    def __init__(
        self,
        model: InstantXControlNetFlux,
        controlnet_cond: torch.Tensor,
        instantx_control_mode: torch.Tensor | None,
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
        self._controlnet_cond = controlnet_cond
        # TODO(ryand): Should we define an enum for the instantx_control_mode? Is it likely to change for future models?
        self._instantx_control_mode = instantx_control_mode

    @classmethod
    def from_controlnet_image(
        cls,
        model: InstantXControlNetFlux,
        controlnet_image: Image,
        instantx_control_mode: torch.Tensor | None,
        vae_info: LoadedModel,
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

        resized_controlnet_image = prepare_control_image(
            image=controlnet_image,
            do_classifier_free_guidance=False,
            width=image_width,
            height=image_height,
            device=device,
            dtype=dtype,
            control_mode=control_mode,
            resize_mode=resize_mode,
        )

        # Run VAE encoder.
        controlnet_cond = FluxVaeEncodeInvocation.vae_encode(vae_info=vae_info, image_tensor=resized_controlnet_image)

        return cls(
            model=model,
            controlnet_cond=controlnet_cond,
            instantx_control_mode=instantx_control_mode,
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
    ) -> InstantXControlNetFluxOutput | None:
        weight = self._get_weight(timestep_index=timestep_index, total_num_timesteps=total_num_timesteps)
        if weight < 1e-6:
            return None

        output: InstantXControlNetFluxOutput = self._model(
            controlnet_cond=self._controlnet_cond,
            controlnet_mode=self._instantx_control_mode,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=y,
            guidance=guidance,
        )

        output.apply_weight(weight)
        return output
