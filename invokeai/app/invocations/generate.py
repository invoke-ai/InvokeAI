# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from functools import partial
from typing import Literal, Optional, Union, get_args

import torch
from diffusers import ControlNetModel
from pydantic import BaseModel, Field

from invokeai.app.models.image import (ColorField, ImageCategory, ImageField,
                                       ResourceOrigin)
from invokeai.app.util.misc import SEED_MAX, get_random_seed
from invokeai.backend.generator.inpaint import infill_methods

from ...backend.generator import Img2Img, Inpaint, InvokeAIGenerator, Txt2Img
from ...backend.stable_diffusion import PipelineIntermediateState
from ..util.step_callback import stable_diffusion_step_callback
from .baseinvocation import BaseInvocation, InvocationConfig, InvocationContext
from .image import ImageOutput

import re
from ...backend.model_management.lora import ModelPatcher
from ...backend.stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline
from .model import UNetField, ClipField, VaeField
from contextlib import contextmanager, ExitStack, ContextDecorator

SAMPLER_NAME_VALUES = Literal[tuple(InvokeAIGenerator.schedulers())]
INFILL_METHODS = Literal[tuple(infill_methods())]
DEFAULT_INFILL_METHOD = (
    "patchmatch" if "patchmatch" in get_args(INFILL_METHODS) else "tile"
)


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

    prompt: Optional[str] = Field(description="The prompt to generate an image from")
    seed:        int = Field(ge=0, le=SEED_MAX, description="The seed to use (omit for random)", default_factory=get_random_seed)
    steps:       int = Field(default=30, gt=0, description="The number of steps to use to generate the image")
    width:       int = Field(default=512, multiple_of=8, gt=0, description="The width of the resulting image", )
    height:      int = Field(default=512, multiple_of=8, gt=0, description="The height of the resulting image", )
    cfg_scale: float = Field(default=7.5, ge=1, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt", )
    scheduler: SAMPLER_NAME_VALUES = Field(default="euler", description="The scheduler to use" )
    #model:       str = Field(default="", description="The model to use (currently ignored)")
    #progress_images: bool = Field(default=False, description="Whether or not to produce progress images during generation",  )
    #control_model: Optional[str] = Field(default=None, description="The control model to use")
    #control_image: Optional[ImageField] = Field(default=None, description="The processed control image")
    unet: UNetField = Field(default=None, description="UNet model")
    clip: ClipField = Field(default=None, description="Clip model")
    vae: VaeField = Field(default=None, description="Vae model")

    # Inputs
    image: Union[ImageField, None] = Field(description="The input image")
    strength: float = Field(
        default=0.75, gt=0, le=1, description="The strength of the original image"
    )
    fit: bool = Field(
        default=True,
        description="Whether or not the result should be fit to the aspect ratio of the input image",
    )

    # Inputs
    mask: Union[ImageField, None] = Field(description="The mask")
    seam_size: int = Field(default=96, ge=1, description="The seam inpaint size (px)")
    seam_blur: int = Field(
        default=16, ge=0, description="The seam inpaint blur radius (px)"
    )
    seam_strength: float = Field(
        default=0.75, gt=0, le=1, description="The seam inpaint strength"
    )
    seam_steps: int = Field(
        default=30, ge=1, description="The number of steps to use for seam inpaint"
    )
    tile_size: int = Field(
        default=32, ge=1, description="The tile infill method size (px)"
    )
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
            "ui": {
                "tags": ["stable-diffusion", "image"],
            },
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

    @contextmanager
    def load_model_old_way(self, context):
        with ExitStack() as stack:
            unet_info = context.services.model_manager.get_model(**self.unet.unet.dict())
            tokenizer_info = context.services.model_manager.get_model(**self.clip.tokenizer.dict())
            text_encoder_info = context.services.model_manager.get_model(**self.clip.text_encoder.dict())
            vae_info = context.services.model_manager.get_model(**self.vae.vae.dict())

            #unet = stack.enter_context(unet_info)
            #tokenizer = stack.enter_context(tokenizer_info)
            #text_encoder = stack.enter_context(text_encoder_info)
            #vae = stack.enter_context(vae_info)
            with vae_info as vae:
                device = vae.device
                dtype = vae.dtype

            # not load models to gpu as it should be handled by pipeline
            unet = unet_info.context.model
            tokenizer = tokenizer_info.context.model
            text_encoder = text_encoder_info.context.model
            vae = vae_info.context.model

            scheduler = get_scheduler(
                context=context,
                scheduler_info=self.unet.scheduler,
                scheduler_name=self.scheduler,
            )

            loras = [(stack.enter_context(context.services.model_manager.get_model(**lora.dict(exclude={"weight"}))), lora.weight) for lora in self.unet.loras]
            ti_list = []
            for trigger in re.findall(r"<[a-zA-Z0-9., _-]+>", self.prompt):
                name = trigger[1:-1]
                try:
                    ti_list.append(
                        stack.enter_context(
                            context.services.model_manager.get_model(
                                model_name=name,
                                base_model=self.clip.text_encoder.base_model,
                                model_type=ModelType.TextualInversion,
                            )
                        )
                    )
                except Exception:
                    #print(e)
                    #import traceback
                    #print(traceback.format_exc())
                    print(f"Warn: trigger: \"{trigger}\" not found")


            with ModelPatcher.apply_lora_unet(unet, loras),\
                 ModelPatcher.apply_lora_text_encoder(text_encoder, loras),\
                 ModelPatcher.apply_ti(tokenizer, text_encoder, ti_list) as (ti_tokenizer, ti_manager):

                pipeline = StableDiffusionGeneratorPipeline(
                    # TODO: ti_manager
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=ti_tokenizer,
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
        image = (
            None
            if self.image is None
            else context.services.images.get_pil_image(self.image.image_name)
        )
        mask = (
            None
            if self.mask is None
            else context.services.images.get_pil_image(self.mask.image_name)
        )

        # Get the source node id (we are invoking the prepared node)
        graph_execution_state = context.services.graph_execution_manager.get(
            context.graph_execution_state_id
        )
        source_node_id = graph_execution_state.prepared_source_mapping[self.id]

        with self.load_model_old_way(context) as model:
            outputs = Inpaint(model).generate(
                prompt=self.prompt,
                init_image=image,
                mask_image=mask,
                step_callback=partial(self.dispatch_progress, context, source_node_id),
                **self.dict(
                    exclude={"prompt", "image", "mask"}
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
