from __future__ import annotations

import math
import torch
from PIL.Image import Image
from typing import TYPE_CHECKING, List, Union
from contextlib import ExitStack
from diffusers import T2IAdapter
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, modifier
#from invokeai.backend.model_manager import BaseModelType # TODO:

if TYPE_CHECKING:
    from invokeai.app.invocations.model import ModelIdentifierField
    from invokeai.app.services.shared.invocation_context import InvocationContext
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager


class T2IAdapterExt(ExtensionBase):
    def __init__(
        self,
        node_context: InvocationContext,
        exit_stack: ExitStack,
        model_id: ModelIdentifierField,
        image: Image,
        adapter_state: List[torch.Tensor],
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
        resize_mode: str,
        priority: int,
    ):
        super().__init__(priority=priority)
        self.node_context = node_context
        self.exit_stack = exit_stack
        self.model_id = model_id
        self.image = image
        self.weight = weight
        self.resize_mode = resize_mode
        self.begin_step_percent = begin_step_percent
        self.end_step_percent = end_step_percent

        self.adapter_state: Optional[Tuple[torch.Tensor]] = None

    @staticmethod
    def tile_coords_to_key(tile_coords):
        return f"{tile_coords.top}:{tile_coords.bottom}:{tile_coords.left}:{tile_coords.right}"

    @modifier("pre_unet_load")
    def run_model(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        t2i_model: T2IAdapter
        with self.node_context.models.load(self.model_id) as t2i_model:
            # used in tiled generation(maybe we should send more info in extra field instead)
            self.latents_height = ctx.latents.shape[2]
            self.latents_width = ctx.latents.shape[3]

            self.adapter_state = self._run_model(
                ctx=ctx,
                model=t2i_model,
                image=self.image,
                latents_height=self.latents_height,
                latents_width=self.latents_width,
            )

    def _run_model(
        self,
        ctx: DenoiseContext,
        model: T2IAdapter,
        image: Image,
        latents_height: int,
        latents_width: int,
    ):
        model_config = self.node_context.models.get_config(self.model_id.key)

        # The max_unet_downscale is the maximum amount that the UNet model downscales the latent image internally.
        from invokeai.backend.model_manager import BaseModelType
        if model_config.base == BaseModelType.StableDiffusion1:
            max_unet_downscale = 8
        elif model_config.base == BaseModelType.StableDiffusionXL:
            max_unet_downscale = 4
        else:
            raise ValueError(f"Unexpected T2I-Adapter base model type: '{model_config.base}'.")

        input_height = latents_height // max_unet_downscale * model.total_downscale_factor
        input_width = latents_width // max_unet_downscale * model.total_downscale_factor

        from invokeai.app.util.controlnet_utils import prepare_control_image
        t2i_image = prepare_control_image(
            image=image,
            do_classifier_free_guidance=False,
            width=input_width,
            height=input_height,
            num_channels=model.config["in_channels"],  # mypy treats this as a FrozenDict
            device=model.device,
            dtype=model.dtype,
            resize_mode=self.resize_mode,
        )

        adapter_state = model(t2i_image)
        #if do_classifier_free_guidance:
        for idx, value in enumerate(adapter_state):
            adapter_state[idx] = torch.cat([value] * 2, dim=0)

        return adapter_state

    @modifier("pre_unet_forward")
    def pre_unet_step(self, ctx: DenoiseContext):
        # skip if model not active in current step
        total_steps = len(ctx.timesteps)
        first_step = math.floor(self.begin_step_percent * total_steps)
        last_step  = math.ceil(self.end_step_percent * total_steps)
        if ctx.step_index < first_step or ctx.step_index > last_step:
            return

        weight = self.weight
        if isinstance(weight, list):
            weight = weight[ctx.step_index]

        tile_coords = ctx.extra.get("tile_coords", None)
        if tile_coords is not None:
            if not isinstance(self.adapter_state, dict):
                self.model = self.exit_stack.enter_context(self.node_context.models.load(self.model_id))
                self.adapter_state = dict()

            tile_key = self.tile_coords_to_key(tile_coords)
            if tile_key not in self.adapter_state:
                tile_height = tile_coords.bottom - tile_coords.top
                tile_width = tile_coords.right - tile_coords.left

                self.adapter_state[tile_key] = self._run_model(
                    ctx=ctx,
                    model=self.model,
                    latents_height=tile_height,
                    latents_width=tile_width,
                    image=self.image.resize((
                        self.latents_width * LATENT_SCALE_FACTOR,
                        self.latents_height * LATENT_SCALE_FACTOR
                    )).crop((
                        tile_coords.left * LATENT_SCALE_FACTOR,
                        tile_coords.top * LATENT_SCALE_FACTOR,
                        tile_coords.right * LATENT_SCALE_FACTOR,
                        tile_coords.bottom * LATENT_SCALE_FACTOR,
                    ))
                )


            adapter_state = self.adapter_state[tile_key]
        else:
            adapter_state = self.adapter_state

        # TODO: conditioning_mode?
        if ctx.unet_kwargs.down_intrablock_additional_residuals is None:
            ctx.unet_kwargs.down_intrablock_additional_residuals = [v * weight for v in adapter_state]
        else:
            for i, value in enumerate(adapter_state):
                ctx.unet_kwargs.down_intrablock_additional_residuals[i] += value * weight
