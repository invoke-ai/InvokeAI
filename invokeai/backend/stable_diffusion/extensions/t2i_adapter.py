import math
import torch
from PIL.Image import Image
from typing import List, Union
from .base import ExtensionBase, modifier
from ..denoise_context import DenoiseContext
from ..extensions_manager import ExtensionsManager
#from invokeai.backend.model_manager import BaseModelType


class T2IAdapterExt(ExtensionBase):
    adapter_state: List[torch.Tensor]
    weight: Union[float, List[float]]
    begin_step_percent: float
    end_step_percent: float

    def __init__(
        self,
        node_context: "InvocationContext",
        model_id: "ModelIdentifierField",
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
        self.model_id = model_id
        self.image = image
        self.weight = weight
        self.resize_mode = resize_mode
        self.begin_step_percent = begin_step_percent
        self.end_step_percent = end_step_percent

        self.adapter_state: Optional[Tuple[torch.Tensor]] = None


    @modifier("pre_unet_load")
    def run_model(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        model_config = self.node_context.models.get_config(self.model_id.key)
        model_loader = self.node_context.models.load(self.model_id)

        from invokeai.backend.model_manager import BaseModelType
        # The max_unet_downscale is the maximum amount that the UNet model downscales the latent image internally.
        if model_config.base == BaseModelType.StableDiffusion1:
            max_unet_downscale = 8
        elif model_config.base == BaseModelType.StableDiffusionXL:
            max_unet_downscale = 4
        else:
            raise ValueError(f"Unexpected T2I-Adapter base model type: '{model_config.base}'.")

        t2i_model: T2IAdapter
        with model_loader as t2i_model:
            total_downscale_factor = t2i_model.total_downscale_factor

            # Resize the T2I-Adapter input image.
            # We select the resize dimensions so that after the T2I-Adapter's total_downscale_factor is applied, the
            # result will match the latent image's dimensions after max_unet_downscale is applied.
            input_height = ctx.latents.shape[2] // max_unet_downscale * total_downscale_factor
            input_width = ctx.latents.shape[3] // max_unet_downscale * total_downscale_factor

            # Note: We have hard-coded `do_classifier_free_guidance=False`. This is because we only want to prepare
            # a single image. If CFG is enabled, we will duplicate the resultant tensor after applying the
            # T2I-Adapter model.
            #
            # Note: We re-use the `prepare_control_image(...)` from ControlNet for T2I-Adapter, because it has many
            # of the same requirements (e.g. preserving binary masks during resize).
            from invokeai.app.util.controlnet_utils import prepare_control_image
            t2i_image = prepare_control_image(
                image=self.image,
                do_classifier_free_guidance=False,
                width=input_width,
                height=input_height,
                num_channels=t2i_model.config["in_channels"],  # mypy treats this as a FrozenDict
                device=t2i_model.device,
                dtype=t2i_model.dtype,
                resize_mode=self.resize_mode,
            )

            self.adapter_state = t2i_model(t2i_image)

        #if do_classifier_free_guidance:
        for idx, value in enumerate(self.adapter_state):
            self.adapter_state[idx] = torch.cat([value] * 2, dim=0)


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

        # TODO: conditioning_mode?
        if ctx.unet_kwargs.down_intrablock_additional_residuals is None:
            ctx.unet_kwargs.down_intrablock_additional_residuals = [v * weight for v in self.adapter_state]
        else:
            for i, value in enumerate(self.adapter_state):
                ctx.unet_kwargs.down_intrablock_additional_residuals[i] += value * weight
