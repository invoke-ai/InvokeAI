# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from contextlib import ExitStack
from typing import List, Literal, Optional, Union

import einops
import torch
from diffusers import ControlNetModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import SchedulerMixin as Scheduler
from pydantic import BaseModel, Field, validator

from invokeai.app.invocations.metadata import CoreMetadata
from invokeai.app.util.step_callback import stable_diffusion_step_callback
from invokeai.backend.model_management.models import ModelType, SilenceWarnings

from ...backend.model_management.lora import ModelPatcher
from ...backend.stable_diffusion import PipelineIntermediateState
from ...backend.stable_diffusion.diffusers_pipeline import (
    ConditioningData,
    ControlNetData,
    StableDiffusionGeneratorPipeline,
    image_resized_to_grid_as_tensor,
)
from ...backend.stable_diffusion.diffusion.shared_invokeai_diffusion import PostprocessingSettings
from ...backend.stable_diffusion.schedulers import SCHEDULER_MAP
from ...backend.util.devices import choose_torch_device, torch_dtype, choose_precision
from ..models.image import ImageCategory, ImageField, ResourceOrigin
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationConfig, InvocationContext
from .compel import ConditioningField
from .controlnet_image_processors import ControlField
from .image import ImageOutput
from .model import ModelInfo, UNetField, VaeField
from invokeai.app.util.controlnet_utils import prepare_control_image

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)


DEFAULT_PRECISION = choose_precision(choose_torch_device())


class LatentsField(BaseModel):
    """A latents field used for passing latents between invocations"""

    latents_name: Optional[str] = Field(default=None, description="The name of the latents")

    class Config:
        schema_extra = {"required": ["latents_name"]}


class LatentsOutput(BaseInvocationOutput):
    """Base class for invocations that output latents"""

    # fmt: off
    type: Literal["latents_output"] = "latents_output"

    # Inputs
    latents: LatentsField          = Field(default=None, description="The output latents")
    width:                     int = Field(description="The width of the latents in pixels")
    height:                    int = Field(description="The height of the latents in pixels")
    # fmt: on


def build_latents_output(latents_name: str, latents: torch.Tensor):
    return LatentsOutput(
        latents=LatentsField(latents_name=latents_name),
        width=latents.size()[3] * 8,
        height=latents.size()[2] * 8,
    )


SAMPLER_NAME_VALUES = Literal[tuple(list(SCHEDULER_MAP.keys()))]


def get_scheduler(
    context: InvocationContext,
    scheduler_info: ModelInfo,
    scheduler_name: str,
) -> Scheduler:
    scheduler_class, scheduler_extra_config = SCHEDULER_MAP.get(scheduler_name, SCHEDULER_MAP["ddim"])
    orig_scheduler_info = context.services.model_manager.get_model(
        **scheduler_info.dict(),
        context=context,
    )
    with orig_scheduler_info as orig_scheduler:
        scheduler_config = orig_scheduler.config

    if "_backup" in scheduler_config:
        scheduler_config = scheduler_config["_backup"]
    scheduler_config = {
        **scheduler_config,
        **scheduler_extra_config,
        "_backup": scheduler_config,
    }
    scheduler = scheduler_class.from_config(scheduler_config)

    # hack copied over from generate.py
    if not hasattr(scheduler, "uses_inpainting_model"):
        scheduler.uses_inpainting_model = lambda: False
    return scheduler


# Text to image
class TextToLatentsInvocation(BaseInvocation):
    """Generates latents from conditionings."""

    type: Literal["t2l"] = "t2l"

    # Inputs
    # fmt: off
    positive_conditioning: Optional[ConditioningField] = Field(description="Positive conditioning for generation")
    negative_conditioning: Optional[ConditioningField] = Field(description="Negative conditioning for generation")
    noise: Optional[LatentsField] = Field(description="The noise to use")
    steps:       int = Field(default=10, gt=0, description="The number of steps to use to generate the image")
    cfg_scale: Union[float, List[float]] = Field(default=7.5, ge=1, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt", )
    scheduler: SAMPLER_NAME_VALUES = Field(default="euler", description="The scheduler to use" )
    unet: UNetField = Field(default=None, description="UNet submodel")
    control: Union[ControlField, list[ControlField]] = Field(default=None, description="The control to use")
    # seamless:   bool = Field(default=False, description="Whether or not to generate an image that can tile without seams", )
    # seamless_axes: str = Field(default="", description="The axes to tile the image on, 'x' and/or 'y'")
    # fmt: on

    @validator("cfg_scale")
    def ge_one(cls, v):
        """validate that all cfg_scale values are >= 1"""
        if isinstance(v, list):
            for i in v:
                if i < 1:
                    raise ValueError("cfg_scale must be greater than 1")
        else:
            if v < 1:
                raise ValueError("cfg_scale must be greater than 1")
        return v

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Text To Latents",
                "tags": ["latents"],
                "type_hints": {
                    "model": "model",
                    "control": "control",
                    # "cfg_scale": "float",
                    "cfg_scale": "number",
                },
            },
        }

    # TODO: pass this an emitter method or something? or a session for dispatching?
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

    def get_conditioning_data(
        self,
        context: InvocationContext,
        scheduler,
        unet,
    ) -> ConditioningData:
        positive_cond_data = context.services.latents.get(self.positive_conditioning.conditioning_name)
        c = positive_cond_data.conditionings[0].embeds.to(device=unet.device, dtype=unet.dtype)
        extra_conditioning_info = positive_cond_data.conditionings[0].extra_conditioning

        negative_cond_data = context.services.latents.get(self.negative_conditioning.conditioning_name)
        uc = negative_cond_data.conditionings[0].embeds.to(device=unet.device, dtype=unet.dtype)

        conditioning_data = ConditioningData(
            unconditioned_embeddings=uc,
            text_embeddings=c,
            guidance_scale=self.cfg_scale,
            extra=extra_conditioning_info,
            postprocessing_settings=PostprocessingSettings(
                threshold=0.0,  # threshold,
                warmup=0.2,  # warmup,
                h_symmetry_time_pct=None,  # h_symmetry_time_pct,
                v_symmetry_time_pct=None,  # v_symmetry_time_pct,
            ),
        )

        conditioning_data = conditioning_data.add_scheduler_args_if_applicable(
            scheduler,
            # for ddim scheduler
            eta=0.0,  # ddim_eta
            # for ancestral and sde schedulers
            generator=torch.Generator(device=unet.device).manual_seed(0),
        )
        return conditioning_data

    def create_pipeline(
        self,
        unet,
        scheduler,
    ) -> StableDiffusionGeneratorPipeline:
        # TODO:
        # configure_model_padding(
        #    unet,
        #    self.seamless,
        #    self.seamless_axes,
        # )

        class FakeVae:
            class FakeVaeConfig:
                def __init__(self):
                    self.block_out_channels = [0]

            def __init__(self):
                self.config = FakeVae.FakeVaeConfig()

        return StableDiffusionGeneratorPipeline(
            vae=FakeVae(),  # TODO: oh...
            text_encoder=None,
            tokenizer=None,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            precision="float16" if unet.dtype == torch.float16 else "float32",
        )

    def prep_control_data(
        self,
        context: InvocationContext,
        # really only need model for dtype and device
        model: StableDiffusionGeneratorPipeline,
        control_input: List[ControlField],
        latents_shape: List[int],
        exit_stack: ExitStack,
        do_classifier_free_guidance: bool = True,
    ) -> List[ControlNetData]:
        # assuming fixed dimensional scaling of 8:1 for image:latents
        control_height_resize = latents_shape[2] * 8
        control_width_resize = latents_shape[3] * 8
        if control_input is None:
            control_list = None
        elif isinstance(control_input, list) and len(control_input) == 0:
            control_list = None
        elif isinstance(control_input, ControlField):
            control_list = [control_input]
        elif isinstance(control_input, list) and len(control_input) > 0 and isinstance(control_input[0], ControlField):
            control_list = control_input
        else:
            control_list = None
        if control_list is None:
            control_data = None
            # from above handling, any control that is not None should now be of type list[ControlField]
        else:
            # FIXME: add checks to skip entry if model or image is None
            #        and if weight is None, populate with default 1.0?
            control_data = []
            control_models = []
            for control_info in control_list:
                control_model = exit_stack.enter_context(
                    context.services.model_manager.get_model(
                        model_name=control_info.control_model.model_name,
                        model_type=ModelType.ControlNet,
                        base_model=control_info.control_model.base_model,
                        context=context,
                    )
                )

                control_models.append(control_model)
                control_image_field = control_info.image
                input_image = context.services.images.get_pil_image(control_image_field.image_name)
                # self.image.image_type, self.image.image_name
                # FIXME: still need to test with different widths, heights, devices, dtypes
                #        and add in batch_size, num_images_per_prompt?
                #        and do real check for classifier_free_guidance?
                # prepare_control_image should return torch.Tensor of shape(batch_size, 3, height, width)
                control_image = prepare_control_image(
                    image=input_image,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    width=control_width_resize,
                    height=control_height_resize,
                    # batch_size=batch_size * num_images_per_prompt,
                    # num_images_per_prompt=num_images_per_prompt,
                    device=control_model.device,
                    dtype=control_model.dtype,
                    control_mode=control_info.control_mode,
                    resize_mode=control_info.resize_mode,
                )
                control_item = ControlNetData(
                    model=control_model,
                    image_tensor=control_image,
                    weight=control_info.control_weight,
                    begin_step_percent=control_info.begin_step_percent,
                    end_step_percent=control_info.end_step_percent,
                    control_mode=control_info.control_mode,
                    # any resizing needed should currently be happening in prepare_control_image(),
                    #    but adding resize_mode to ControlNetData in case needed in the future
                    resize_mode=control_info.resize_mode,
                )
                control_data.append(control_item)
                # MultiControlNetModel has been refactored out, just need list[ControlNetData]
        return control_data

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        with SilenceWarnings():
            noise = context.services.latents.get(self.noise.latents_name)

            # Get the source node id (we are invoking the prepared node)
            graph_execution_state = context.services.graph_execution_manager.get(context.graph_execution_state_id)
            source_node_id = graph_execution_state.prepared_source_mapping[self.id]

            def step_callback(state: PipelineIntermediateState):
                self.dispatch_progress(context, source_node_id, state)

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
            with ExitStack() as exit_stack, ModelPatcher.apply_lora_unet(
                unet_info.context.model, _lora_loader()
            ), unet_info as unet:
                noise = noise.to(device=unet.device, dtype=unet.dtype)

                scheduler = get_scheduler(
                    context=context,
                    scheduler_info=self.unet.scheduler,
                    scheduler_name=self.scheduler,
                )

                pipeline = self.create_pipeline(unet, scheduler)
                conditioning_data = self.get_conditioning_data(context, scheduler, unet)

                control_data = self.prep_control_data(
                    model=pipeline,
                    context=context,
                    control_input=self.control,
                    latents_shape=noise.shape,
                    # do_classifier_free_guidance=(self.cfg_scale >= 1.0))
                    do_classifier_free_guidance=True,
                    exit_stack=exit_stack,
                )

                # TODO: Verify the noise is the right size
                result_latents, result_attention_map_saver = pipeline.latents_from_embeddings(
                    latents=torch.zeros_like(noise, dtype=torch_dtype(unet.device)),
                    noise=noise,
                    num_inference_steps=self.steps,
                    conditioning_data=conditioning_data,
                    control_data=control_data,  # list[ControlNetData]
                    callback=step_callback,
                )

            # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
            result_latents = result_latents.to("cpu")
            torch.cuda.empty_cache()

            name = f"{context.graph_execution_state_id}__{self.id}"
            context.services.latents.save(name, result_latents)
            return build_latents_output(latents_name=name, latents=result_latents)


class LatentsToLatentsInvocation(TextToLatentsInvocation):
    """Generates latents using latents as base image."""

    type: Literal["l2l"] = "l2l"

    # Inputs
    latents: Optional[LatentsField] = Field(description="The latents to use as a base image")
    strength: float = Field(default=0.7, ge=0, le=1, description="The strength of the latents to use")

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Latent To Latents",
                "tags": ["latents"],
                "type_hints": {
                    "model": "model",
                    "control": "control",
                    "cfg_scale": "number",
                },
            },
        }

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        with SilenceWarnings():  # this quenches NSFW nag from diffusers
            noise = context.services.latents.get(self.noise.latents_name)
            latent = context.services.latents.get(self.latents.latents_name)

            # Get the source node id (we are invoking the prepared node)
            graph_execution_state = context.services.graph_execution_manager.get(context.graph_execution_state_id)
            source_node_id = graph_execution_state.prepared_source_mapping[self.id]

            def step_callback(state: PipelineIntermediateState):
                self.dispatch_progress(context, source_node_id, state)

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
            with ExitStack() as exit_stack, ModelPatcher.apply_lora_unet(
                unet_info.context.model, _lora_loader()
            ), unet_info as unet:
                noise = noise.to(device=unet.device, dtype=unet.dtype)
                latent = latent.to(device=unet.device, dtype=unet.dtype)

                scheduler = get_scheduler(
                    context=context,
                    scheduler_info=self.unet.scheduler,
                    scheduler_name=self.scheduler,
                )

                pipeline = self.create_pipeline(unet, scheduler)
                conditioning_data = self.get_conditioning_data(context, scheduler, unet)

                control_data = self.prep_control_data(
                    model=pipeline,
                    context=context,
                    control_input=self.control,
                    latents_shape=noise.shape,
                    # do_classifier_free_guidance=(self.cfg_scale >= 1.0))
                    do_classifier_free_guidance=True,
                    exit_stack=exit_stack,
                )

                # TODO: Verify the noise is the right size
                initial_latents = (
                    latent if self.strength < 1.0 else torch.zeros_like(latent, device=unet.device, dtype=latent.dtype)
                )

                timesteps, _ = pipeline.get_img2img_timesteps(
                    self.steps,
                    self.strength,
                    device=unet.device,
                )

                result_latents, result_attention_map_saver = pipeline.latents_from_embeddings(
                    latents=initial_latents,
                    timesteps=timesteps,
                    noise=noise,
                    num_inference_steps=self.steps,
                    conditioning_data=conditioning_data,
                    control_data=control_data,  # list[ControlNetData]
                    callback=step_callback,
                )

            # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
            result_latents = result_latents.to("cpu")
            torch.cuda.empty_cache()

            name = f"{context.graph_execution_state_id}__{self.id}"
            context.services.latents.save(name, result_latents)
        return build_latents_output(latents_name=name, latents=result_latents)


# Latent to image
class LatentsToImageInvocation(BaseInvocation):
    """Generates an image from latents."""

    type: Literal["l2i"] = "l2i"

    # Inputs
    latents: Optional[LatentsField] = Field(description="The latents to generate an image from")
    vae: VaeField = Field(default=None, description="Vae submodel")
    tiled: bool = Field(default=False, description="Decode latents by overlaping tiles (less memory consumption)")
    fp32: bool = Field(DEFAULT_PRECISION == "float32", description="Decode in full precision")
    metadata: Optional[CoreMetadata] = Field(
        default=None, description="Optional core metadata to be written to the image"
    )

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Latents To Image",
                "tags": ["latents", "image"],
            },
        }

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.services.latents.get(self.latents.latents_name)

        vae_info = context.services.model_manager.get_model(
            **self.vae.vae.dict(),
            context=context,
        )

        with vae_info as vae:
            latents = latents.to(vae.device)
            if self.fp32:
                vae.to(dtype=torch.float32)

                use_torch_2_0_or_xformers = isinstance(
                    vae.decoder.mid_block.attentions[0].processor,
                    (
                        AttnProcessor2_0,
                        XFormersAttnProcessor,
                        LoRAXFormersAttnProcessor,
                        LoRAAttnProcessor2_0,
                    ),
                )
                # if xformers or torch_2_0 is used attention block does not need
                # to be in float32 which can save lots of memory
                if use_torch_2_0_or_xformers:
                    vae.post_quant_conv.to(latents.dtype)
                    vae.decoder.conv_in.to(latents.dtype)
                    vae.decoder.mid_block.to(latents.dtype)
                else:
                    latents = latents.float()

            else:
                vae.to(dtype=torch.float16)
                latents = latents.half()

            if self.tiled or context.services.configuration.tiled_decode:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

            # clear memory as vae decode can request a lot
            torch.cuda.empty_cache()

            with torch.inference_mode():
                # copied from diffusers pipeline
                latents = latents / vae.config.scaling_factor
                image = vae.decode(latents, return_dict=False)[0]
                image = (image / 2 + 0.5).clamp(0, 1)  # denormalize
                # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                np_image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                image = VaeImageProcessor.numpy_to_pil(np_image)[0]

        torch.cuda.empty_cache()

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata.dict() if self.metadata else None,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )


LATENTS_INTERPOLATION_MODE = Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"]


class ResizeLatentsInvocation(BaseInvocation):
    """Resizes latents to explicit width/height (in pixels). Provided dimensions are floor-divided by 8."""

    type: Literal["lresize"] = "lresize"

    # Inputs
    latents: Optional[LatentsField] = Field(description="The latents to resize")
    width: Union[int, None] = Field(default=512, ge=64, multiple_of=8, description="The width to resize to (px)")
    height: Union[int, None] = Field(default=512, ge=64, multiple_of=8, description="The height to resize to (px)")
    mode: LATENTS_INTERPOLATION_MODE = Field(default="bilinear", description="The interpolation mode")
    antialias: bool = Field(
        default=False, description="Whether or not to antialias (applied in bilinear and bicubic modes only)"
    )

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Resize Latents", "tags": ["latents", "resize"]},
        }

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.services.latents.get(self.latents.latents_name)

        # TODO:
        device = choose_torch_device()

        resized_latents = torch.nn.functional.interpolate(
            latents.to(device),
            size=(self.height // 8, self.width // 8),
            mode=self.mode,
            antialias=self.antialias if self.mode in ["bilinear", "bicubic"] else False,
        )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        resized_latents = resized_latents.to("cpu")
        torch.cuda.empty_cache()

        name = f"{context.graph_execution_state_id}__{self.id}"
        # context.services.latents.set(name, resized_latents)
        context.services.latents.save(name, resized_latents)
        return build_latents_output(latents_name=name, latents=resized_latents)


class ScaleLatentsInvocation(BaseInvocation):
    """Scales latents by a given factor."""

    type: Literal["lscale"] = "lscale"

    # Inputs
    latents: Optional[LatentsField] = Field(description="The latents to scale")
    scale_factor: float = Field(gt=0, description="The factor by which to scale the latents")
    mode: LATENTS_INTERPOLATION_MODE = Field(default="bilinear", description="The interpolation mode")
    antialias: bool = Field(
        default=False, description="Whether or not to antialias (applied in bilinear and bicubic modes only)"
    )

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Scale Latents", "tags": ["latents", "scale"]},
        }

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.services.latents.get(self.latents.latents_name)

        # TODO:
        device = choose_torch_device()

        # resizing
        resized_latents = torch.nn.functional.interpolate(
            latents.to(device),
            scale_factor=self.scale_factor,
            mode=self.mode,
            antialias=self.antialias if self.mode in ["bilinear", "bicubic"] else False,
        )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        resized_latents = resized_latents.to("cpu")
        torch.cuda.empty_cache()

        name = f"{context.graph_execution_state_id}__{self.id}"
        # context.services.latents.set(name, resized_latents)
        context.services.latents.save(name, resized_latents)
        return build_latents_output(latents_name=name, latents=resized_latents)


class ImageToLatentsInvocation(BaseInvocation):
    """Encodes an image into latents."""

    type: Literal["i2l"] = "i2l"

    # Inputs
    image: Optional[ImageField] = Field(description="The image to encode")
    vae: VaeField = Field(default=None, description="Vae submodel")
    tiled: bool = Field(default=False, description="Encode latents by overlaping tiles(less memory consumption)")
    fp32: bool = Field(DEFAULT_PRECISION == "float32", description="Decode in full precision")

    # Schema customisation
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {"title": "Image To Latents", "tags": ["latents", "image"]},
        }

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # image = context.services.images.get(
        #     self.image.image_type, self.image.image_name
        # )
        image = context.services.images.get_pil_image(self.image.image_name)

        # vae_info = context.services.model_manager.get_model(**self.vae.vae.dict())
        vae_info = context.services.model_manager.get_model(
            **self.vae.vae.dict(),
            context=context,
        )

        image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        if image_tensor.dim() == 3:
            image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        with vae_info as vae:
            orig_dtype = vae.dtype
            if self.fp32:
                vae.to(dtype=torch.float32)

                use_torch_2_0_or_xformers = isinstance(
                    vae.decoder.mid_block.attentions[0].processor,
                    (
                        AttnProcessor2_0,
                        XFormersAttnProcessor,
                        LoRAXFormersAttnProcessor,
                        LoRAAttnProcessor2_0,
                    ),
                )
                # if xformers or torch_2_0 is used attention block does not need
                # to be in float32 which can save lots of memory
                if use_torch_2_0_or_xformers:
                    vae.post_quant_conv.to(orig_dtype)
                    vae.decoder.conv_in.to(orig_dtype)
                    vae.decoder.mid_block.to(orig_dtype)
                # else:
                #    latents = latents.float()

            else:
                vae.to(dtype=torch.float16)
                # latents = latents.half()

            if self.tiled:
                vae.enable_tiling()
            else:
                vae.disable_tiling()

            # non_noised_latents_from_image
            image_tensor = image_tensor.to(device=vae.device, dtype=vae.dtype)
            with torch.inference_mode():
                image_tensor_dist = vae.encode(image_tensor).latent_dist
                latents = image_tensor_dist.sample().to(dtype=vae.dtype)  # FIXME: uses torch.randn. make reproducible!

            latents = vae.config.scaling_factor * latents
            latents = latents.to(dtype=orig_dtype)

        name = f"{context.graph_execution_state_id}__{self.id}"
        latents = latents.to("cpu")
        context.services.latents.save(name, latents)
        return build_latents_output(latents_name=name, latents=latents)
