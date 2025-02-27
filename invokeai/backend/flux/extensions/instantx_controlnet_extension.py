import math
from typing import List, Union

import torch
from PIL.Image import Image

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.app.util.controlnet_utils import CONTROLNET_RESIZE_VALUES, prepare_control_image
from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import (
    InstantXControlNetFlux,
    InstantXControlNetFluxOutput,
)
from invokeai.backend.flux.extensions.base_controlnet_extension import BaseControlNetExtension
from invokeai.backend.flux.sampling_utils import pack
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
        # The VAE-encoded and 'packed' control image to pass to the ControlNet model.
        self._controlnet_cond = controlnet_cond
        # TODO(ryand): Should we define an enum for the instantx_control_mode? Is it likely to change for future models?
        # The control mode for InstantX ControlNet union models.
        # See the values defined here: https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union#control-mode
        # Expected shape: (batch_size, 1), Expected dtype: torch.long
        # If None, a zero-embedding will be used.
        self._instantx_control_mode = instantx_control_mode

        # TODO(ryand): Pass in these params if a new base transformer / InstantX ControlNet pair get released.
        self._flux_transformer_num_double_blocks = 19
        self._flux_transformer_num_single_blocks = 38

    @classmethod
    def prepare_controlnet_cond(
        cls,
        controlnet_image: Image,
        vae_info: LoadedModel,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
        resize_mode: CONTROLNET_RESIZE_VALUES,
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
            control_mode="balanced",
            resize_mode=resize_mode,
        )

        # Shift the image from [0, 1] to [-1, 1].
        resized_controlnet_image = resized_controlnet_image * 2 - 1

        # Run VAE encoder.
        controlnet_cond = FluxVaeEncodeInvocation.vae_encode(vae_info=vae_info, image_tensor=resized_controlnet_image)
        controlnet_cond = pack(controlnet_cond)

        return controlnet_cond

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
            control_mode="balanced",
            resize_mode=resize_mode,
        )

        # Shift the image from [0, 1] to [-1, 1].
        resized_controlnet_image = resized_controlnet_image * 2 - 1

        # Run VAE encoder.
        controlnet_cond = FluxVaeEncodeInvocation.vae_encode(vae_info=vae_info, image_tensor=resized_controlnet_image)
        controlnet_cond = pack(controlnet_cond)

        return cls(
            model=model,
            controlnet_cond=controlnet_cond,
            instantx_control_mode=instantx_control_mode,
            weight=weight,
            begin_step_percent=begin_step_percent,
            end_step_percent=end_step_percent,
        )

    def _instantx_output_to_controlnet_output(
        self, instantx_output: InstantXControlNetFluxOutput
    ) -> ControlNetFluxOutput:
        # The `interval_control` logic here is based on
        # https://github.com/huggingface/diffusers/blob/31058cdaef63ca660a1a045281d156239fba8192/src/diffusers/models/transformers/transformer_flux.py#L507-L511

        # Handle double block residuals.
        double_block_residuals: list[torch.Tensor] = []
        double_block_samples = instantx_output.controlnet_block_samples
        if double_block_samples:
            interval_control = self._flux_transformer_num_double_blocks / len(double_block_samples)
            interval_control = int(math.ceil(interval_control))
            for i in range(self._flux_transformer_num_double_blocks):
                double_block_residuals.append(double_block_samples[i // interval_control])

        # Handle single block residuals.
        single_block_residuals: list[torch.Tensor] = []
        single_block_samples = instantx_output.controlnet_single_block_samples
        if single_block_samples:
            interval_control = self._flux_transformer_num_single_blocks / len(single_block_samples)
            interval_control = int(math.ceil(interval_control))
            for i in range(self._flux_transformer_num_single_blocks):
                single_block_residuals.append(single_block_samples[i // interval_control])

        return ControlNetFluxOutput(
            double_block_residuals=double_block_residuals or None,
            single_block_residuals=single_block_residuals or None,
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

        # Make sure inputs have correct device and dtype.
        self._controlnet_cond = self._controlnet_cond.to(device=img.device, dtype=img.dtype)
        self._instantx_control_mode = (
            self._instantx_control_mode.to(device=img.device) if self._instantx_control_mode is not None else None
        )

        instantx_output: InstantXControlNetFluxOutput = self._model(
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

        controlnet_output = self._instantx_output_to_controlnet_output(instantx_output)
        controlnet_output.apply_weight(weight)
        return controlnet_output
