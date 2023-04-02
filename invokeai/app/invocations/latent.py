# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654)

from typing import Literal, Optional
from pydantic import BaseModel, Field
from torch import Tensor
import torch

from ...backend.model_management.model_manager import ModelManager
from ...backend.util.devices import CUDA_DEVICE, torch_dtype
from ...backend.stable_diffusion.diffusion.shared_invokeai_diffusion import PostprocessingSettings
from ...backend.image_util.seamless import configure_model_padding
from ...backend.prompting.conditioning import get_uc_and_c_and_ec
from ...backend.stable_diffusion.diffusers_pipeline import ConditioningData, StableDiffusionGeneratorPipeline
from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext
import numpy as np
from accelerate.utils import set_seed
from ..services.image_storage import ImageType
from .baseinvocation import BaseInvocation, InvocationContext
from .image import ImageField, ImageOutput
from ...backend.generator import Generator
from ...backend.stable_diffusion import PipelineIntermediateState
from ...backend.util.util import image_to_dataURL
from diffusers.schedulers import SchedulerMixin as Scheduler
import diffusers
from diffusers import DiffusionPipeline


class LatentsField(BaseModel):
    """A latents field used for passing latents between invocations"""

    latents_name: Optional[str] = Field(default=None, description="The name of the latents")


class LatentsOutput(BaseInvocationOutput):
    """Base class for invocations that output latents"""
    #fmt: off
    type: Literal["latent_output"] = "latent_output"
    latents: LatentsField            = Field(default=None, description="The output latents")
    #fmt: on

class NoiseOutput(BaseInvocationOutput):
    """Invocation noise output"""
    #fmt: off
    type: Literal["noise_output"] = "noise_output"
    noise: LatentsField            = Field(default=None, description="The output noise")
    #fmt: on


# TODO: this seems like a hack
scheduler_map = dict(
    ddim=diffusers.DDIMScheduler,
    dpmpp_2=diffusers.DPMSolverMultistepScheduler,
    k_dpm_2=diffusers.KDPM2DiscreteScheduler,
    k_dpm_2_a=diffusers.KDPM2AncestralDiscreteScheduler,
    k_dpmpp_2=diffusers.DPMSolverMultistepScheduler,
    k_euler=diffusers.EulerDiscreteScheduler,
    k_euler_a=diffusers.EulerAncestralDiscreteScheduler,
    k_heun=diffusers.HeunDiscreteScheduler,
    k_lms=diffusers.LMSDiscreteScheduler,
    plms=diffusers.PNDMScheduler,
)


SAMPLER_NAME_VALUES = Literal[
    tuple(list(scheduler_map.keys()))
]


def get_scheduler(scheduler_name:str, model: StableDiffusionGeneratorPipeline)->Scheduler:
    scheduler_class = scheduler_map.get(scheduler_name,'ddim')
    scheduler = scheduler_class.from_config(model.scheduler.config)
    # hack copied over from generate.py
    if not hasattr(scheduler, 'uses_inpainting_model'):
        scheduler.uses_inpainting_model = lambda: False
    return scheduler


def get_noise(width:int, height:int, device:torch.device, seed:int = 0, latent_channels:int=4, use_mps_noise:bool=False, downsampling_factor:int = 8):
    # limit noise to only the diffusion image channels, not the mask channels
    input_channels = min(latent_channels, 4)
    use_device = "cpu" if (use_mps_noise or device.type == "mps") else device
    generator = torch.Generator(device=use_device).manual_seed(seed)
    x = torch.randn(
        [
            1,
            input_channels,
            height // downsampling_factor,
            width //  downsampling_factor,
        ],
        dtype=torch_dtype(device),
        device=use_device,
        generator=generator,
    ).to(device)
    # if self.perlin > 0.0:
    #     perlin_noise = self.get_perlin_noise(
    #         width // self.downsampling_factor, height // self.downsampling_factor
    #     )
    #     x = (1 - self.perlin) * x + self.perlin * perlin_noise
    return x


class NoiseInvocation(BaseInvocation):
    """Generates latent noise."""

    type: Literal["noise"] = "noise"

    # Inputs
    seed:        int = Field(default=0, ge=0, le=np.iinfo(np.uint32).max, description="The seed to use", )
    width:       int = Field(default=512, multiple_of=64, gt=0, description="The width of the resulting noise", )
    height:      int = Field(default=512, multiple_of=64, gt=0, description="The height of the resulting noise", )

    def invoke(self, context: InvocationContext) -> NoiseOutput:
        device = torch.device(CUDA_DEVICE)
        noise = get_noise(self.width, self.height, device, self.seed)

        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.set(name, noise)
        return NoiseOutput(
            noise=LatentsField(latents_name=name)
        )


# Text to image
class TextToLatentsInvocation(BaseInvocation):
    """Generates latents from a prompt."""

    type: Literal["t2l"] = "t2l"

    # Inputs
    # TODO: consider making prompt optional to enable providing prompt through a link
    # fmt: off
    prompt: Optional[str] = Field(description="The prompt to generate an image from")
    seed:        int = Field(default=-1,ge=-1, le=np.iinfo(np.uint32).max, description="The seed to use (-1 for a random seed)", )
    noise: Optional[LatentsField] = Field(description="The noise to use")
    steps:       int = Field(default=10, gt=0, description="The number of steps to use to generate the image")
    width:       int = Field(default=512, multiple_of=64, gt=0, description="The width of the resulting image", )
    height:      int = Field(default=512, multiple_of=64, gt=0, description="The height of the resulting image", )
    cfg_scale: float = Field(default=7.5, gt=0, description="The Classifier-Free Guidance, higher values may result in a result closer to the prompt", )
    sampler_name: SAMPLER_NAME_VALUES = Field(default="k_lms", description="The sampler to use" )
    seamless:   bool = Field(default=False, description="Whether or not to generate an image that can tile without seams", )
    seamless_axes: str = Field(default="", description="The axes to tile the image on, 'x' and/or 'y'")
    model:       str = Field(default="", description="The model to use (currently ignored)")
    progress_images: bool = Field(default=False, description="Whether or not to produce progress images during generation",  )
    # fmt: on

    # TODO: pass this an emitter method or something? or a session for dispatching?
    def dispatch_progress(
        self, context: InvocationContext, sample: Tensor, step: int
    ) -> None:  
        # TODO: only output a preview image when requested
        image = Generator.sample_to_lowres_estimated_image(sample)

        (width, height) = image.size
        width *= 8
        height *= 8

        dataURL = image_to_dataURL(image, image_format="JPEG")

        context.services.events.emit_generator_progress(
            context.graph_execution_state_id,
            self.id,
            {
                "width": width,
                "height": height,
                "dataURL": dataURL
            },
            step,
            self.steps,
        )
    
    def get_model(self, model_manager: ModelManager) -> StableDiffusionGeneratorPipeline:
        model_info = model_manager.get_model(self.model)
        model_name = model_info['model_name']
        model_hash = model_info['hash']
        model: StableDiffusionGeneratorPipeline = model_info['model']
        model.scheduler = get_scheduler(
            model=model,
            scheduler_name=self.sampler_name
        )
        
        if isinstance(model, DiffusionPipeline):
            for component in [model.unet, model.vae]:
                configure_model_padding(component,
                                        self.seamless,
                                        self.seamless_axes
                                        )
        else:
            configure_model_padding(model,
                                    self.seamless,
                                    self.seamless_axes
                                    )

        return model


    def get_conditioning_data(self, model: StableDiffusionGeneratorPipeline) -> ConditioningData:
        uc, c, extra_conditioning_info = get_uc_and_c_and_ec(self.prompt, model=model)
        conditioning_data = ConditioningData(
            uc,
            c,
            self.cfg_scale,
            extra_conditioning_info,
            postprocessing_settings=PostprocessingSettings(
                threshold=0.0,#threshold,
                warmup=0.2,#warmup,
                h_symmetry_time_pct=None,#h_symmetry_time_pct,
                v_symmetry_time_pct=None#v_symmetry_time_pct,
            ),
        ).add_scheduler_args_if_applicable(model.scheduler, eta=None)#ddim_eta)
        return conditioning_data


    def invoke(self, context: InvocationContext) -> LatentsOutput:
        noise = context.services.latents.get(self.noise.latents_name)

        def step_callback(state: PipelineIntermediateState):
            self.dispatch_progress(context, state.latents, state.step)

        model = self.get_model(context.services.model_manager)
        conditioning_data = self.get_conditioning_data(model)

        # TODO: Verify the noise is the right size

        result_latents, result_attention_map_saver = model.latents_from_embeddings(
            latents=torch.zeros_like(noise, dtype=torch_dtype(model.device)),
            noise=noise,
            num_inference_steps=self.steps,
            conditioning_data=conditioning_data,
            callback=step_callback
        )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        torch.cuda.empty_cache()

        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.set(name, result_latents)
        return LatentsOutput(
            latents=LatentsField(latents_name=name)
        )


class LatentsToLatentsInvocation(TextToLatentsInvocation):
    """Generates latents using latents as base image."""

    type: Literal["l2l"] = "l2l"

    # Inputs
    latents: Optional[LatentsField] = Field(description="The latents to use as a base image")
    strength: float = Field(default=0.5, description="The strength of the latents to use")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        noise = context.services.latents.get(self.noise.latents_name)
        latent = context.services.latents.get(self.latents.latents_name)

        def step_callback(state: PipelineIntermediateState):
            self.dispatch_progress(context, state.latents, state.step)

        model = self.get_model(context.services.model_manager)
        conditioning_data = self.get_conditioning_data(model)

        # TODO: Verify the noise is the right size

        initial_latents = latent if self.strength < 1.0 else torch.zeros_like(
            latent, device=model.device, dtype=latent.dtype
        )
        
        timesteps, _ = model.get_img2img_timesteps(
            self.steps,
            self.strength,
            device=model.device,
        )

        result_latents, result_attention_map_saver = model.latents_from_embeddings(
            latents=initial_latents,
            timesteps=timesteps,
            noise=noise,
            num_inference_steps=self.steps,
            conditioning_data=conditioning_data,
            callback=step_callback
        )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        torch.cuda.empty_cache()

        name = f'{context.graph_execution_state_id}__{self.id}'
        context.services.latents.set(name, result_latents)
        return LatentsOutput(
            latents=LatentsField(latents_name=name)
        )


# Latent to image
class LatentsToImageInvocation(BaseInvocation):
    """Generates an image from latents."""

    type: Literal["l2i"] = "l2i"

    # Inputs
    latents: Optional[LatentsField] = Field(description="The latents to generate an image from")
    model: str = Field(default="", description="The model to use")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        latents = context.services.latents.get(self.latents.latents_name)

        # TODO: this only really needs the vae
        model_info = context.services.model_manager.get_model(self.model)
        model: StableDiffusionGeneratorPipeline = model_info['model']

        with torch.inference_mode():
            np_image = model.decode_latents(latents)
            image = model.numpy_to_pil(np_image)[0]

            image_type = ImageType.RESULT
            image_name = context.services.images.create_name(
                context.graph_execution_state_id, self.id
            )
            context.services.images.save(image_type, image_name, image)
            return ImageOutput(
                image=ImageField(image_type=image_type, image_name=image_name)
            )
