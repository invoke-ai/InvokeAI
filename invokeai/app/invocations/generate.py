# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from functools import partial
from typing import Literal, Optional, get_args

import torch
from pydantic import Field

from invokeai.app.models.image import ColorField, ImageCategory, ImageField, ResourceOrigin
from invokeai.app.util.misc import SEED_MAX, get_random_seed
from invokeai.backend.generator.inpaint import infill_methods

from ...backend.generator import Inpaint, InvokeAIGenerator
from ...backend.stable_diffusion import PipelineIntermediateState
from ..util.step_callback import stable_diffusion_step_callback
from .baseinvocation import BaseInvocation, InvocationConfig, InvocationContext
from .image import ImageOutput

from ...backend.model_management.lora import ModelPatcher
from ...backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline
from .model import UNetField, VaeField
from .compel import ConditioningField
from contextlib import contextmanager, ExitStack, ContextDecorator

SAMPLER_NAME_VALUES = Literal[tuple(InvokeAIGenerator.schedulers())]
INFILL_METHODS = Literal[tuple(infill_methods())]
DEFAULT_INFILL_METHOD = "patchmatch" if "patchmatch" in get_args(INFILL_METHODS) else "tile"


from .latent import get_scheduler


class OldModelContext(ContextDecorator):
    model: StableDiffusionGeneratorPipeline

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, *exc):
        return False


class OldModelInfo:
    name: str
    hash: str
    context: OldModelContext

    def __init__(self, name: str, hash: str, model: StableDiffusionGeneratorPipeline):
        self.name = name
        self.hash = hash
        self.context = OldModelContext(
            model=model,
        )


class InpaintInvocation(BaseInvocation):
    """Generates an image using inpaint."""

    type: Literal["inpaint"] = "inpaint"

    positive_conditioning: Optional[ConditioningField] = Field(description="Positive conditioning for generation")
    negative_conditioning: Optional[ConditioningField] = Field(description="Negative conditioning for generation")
    seed: int = Field(
        ge=0, le=SEED_MAX, description="The seed to use (omit for random)", default_factory=get_random_seed
    )
    steps: int = Field(default=30, gt=0, description="The number of steps to use to generate the image")
    width: int = Field(
        default=512,
        multiple_of=8,
        gt=0,
        description="The width of the resulting image",
    )
    height: int = Field(
        default=512,
        multiple_of=8,
        gt=0,
        description="The height of the resulting image",
    )
    cfg_scale: float = Field(
        default=7.5,
        ge=1,
        description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt",
    )
    scheduler: SAMPLER_NAME_VALUES = Field(default="euler", description="The scheduler to use")
    unet: UNetField = Field(default=None, description="UNet model")
    vae: VaeField = Field(default=None, description="Vae model")

    # Inputs
    image: Optional[ImageField] = Field(description="The input image")
    strength: float = Field(default=0.75, gt=0, le=1, description="The strength of the original image")
    fit: bool = Field(
        default=True,
        description="Whether or not the result should be fit to the aspect ratio of the input image",
    )

    # Inputs
    mask: Optional[ImageField] = Field(description="The mask")
    seam_size: int = Field(default=96, ge=1, description="The seam inpaint size (px)")
    seam_blur: int = Field(default=16, ge=0, description="The seam inpaint blur radius (px)")
    seam_strength: float = Field(default=0.75, gt=0, le=1, description="The seam inpaint strength")
    seam_steps: int = Field(default=30, ge=1, description="The number of steps to use for seam inpaint")
    tile_size: int = Field(default=32, ge=1, description="The tile infill method size (px)")
    infill_method: INFILL_METHODS = Field(
        default=DEFAULT_INFILL_METHOD,
        description="The method used to infill empty regions (px)",
    )
    inpaint_width: Optional[int] = Field(
        default=None,
        multiple_of=8,
        gt=0,
        description="The width of the inpaint region (px)",
    )
    inpaint_height: Optional[int] = Field(
        default=None,
        multiple_of=8,
        gt=0,
        description="The height of the inpaint region (px)",
    )
    inpaint_fill: Optional[ColorField] = Field(
        default=ColorField(r=127, g=127, b=127, a=255),
        description="The solid infill method color",
    )
    inpaint_replace: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The amount by which to replace masked areas with latent noise",
    )

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"tags": ["stable-diffusion", "image"], "title": "Inpaint"},
        }

    def dispatch_progress(
        self,
        context: InvocationContext,
        source_node_id: str,
        intermediate_state: PipelineIntermediateState,
    ) -> None:
        stable_diffusion_step_callback(
            context=context,
            intermediate_state=intermediate_state,
            node=self.dict(),
            source_node_id=source_node_id,
        )

    def get_conditioning(self, context, unet):
        positive_cond_data = context.services.latents.get(self.positive_conditioning.conditioning_name)
        c = positive_cond_data.conditionings[0].embeds.to(device=unet.device, dtype=unet.dtype)
        extra_conditioning_info = positive_cond_data.conditionings[0].extra_conditioning

        negative_cond_data = context.services.latents.get(self.negative_conditioning.conditioning_name)
        uc = negative_cond_data.conditionings[0].embeds.to(device=unet.device, dtype=unet.dtype)

        return (uc, c, extra_conditioning_info)

    @contextmanager
    def load_model_old_way(self, context, scheduler):
        def _lora_loader():
            for lora in self.unet.loras:
                lora_info = context.services.model_manager.get_model(
                    **lora.dict(exclude={"weight"}),
                    context=context,
                )
                yield (lora_info.context.model, lora.weight)
                del lora_info
            return

        unet_info = context.services.model_manager.get_model(
            **self.unet.unet.dict(),
            context=context,
        )
        vae_info = context.services.model_manager.get_model(
            **self.vae.vae.dict(),
            context=context,
        )

        with vae_info as vae, ModelPatcher.apply_lora_unet(unet_info.context.model, _lora_loader()), unet_info as unet:
            device = context.services.model_manager.mgr.cache.execution_device
            dtype = context.services.model_manager.mgr.cache.precision

            pipeline = StableDiffusionGeneratorPipeline(
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                precision="float16" if dtype == torch.float16 else "float32",
                execution_device=device,
            )

            yield OldModelInfo(
                name=self.unet.unet.model_name,
                hash="<NO-HASH>",
                model=pipeline,
            )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = None if self.image is None else context.services.images.get_pil_image(self.image.image_name)
        mask = None if self.mask is None else context.services.images.get_pil_image(self.mask.image_name)

        # Get the source node id (we are invoking the prepared node)
        graph_execution_state = context.services.graph_execution_manager.get(context.graph_execution_state_id)
        source_node_id = graph_execution_state.prepared_source_mapping[self.id]

        scheduler = get_scheduler(
            context=context,
            scheduler_info=self.unet.scheduler,
            scheduler_name=self.scheduler,
        )

        with self.load_model_old_way(context, scheduler) as model:
            conditioning = self.get_conditioning(context, model.context.model.unet)

            outputs = Inpaint(model).generate(
                conditioning=conditioning,
                scheduler=scheduler,
                init_image=image,
                mask_image=mask,
                step_callback=partial(self.dispatch_progress, context, source_node_id),
                **self.dict(
                    exclude={"positive_conditioning", "negative_conditioning", "scheduler", "image", "mask"}
                ),  # Shorthand for passing all of the parameters above manually
            )

            # Outputs is an infinite iterator that will return a new InvokeAIGeneratorOutput object
            # each time it is called. We only need the first one.
            generator_output = next(outputs)

            image_dto = context.services.images.create(
                image=generator_output.image,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                session_id=context.graph_execution_state_id,
                node_id=self.id,
                is_intermediate=self.is_intermediate,
            )

            return ImageOutput(
                image=ImageField(image_name=image_dto.image_name),
                width=image_dto.width,
                height=image_dto.height,
            )
