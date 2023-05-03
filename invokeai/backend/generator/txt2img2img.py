"""
invokeai.backend.generator.txt2img inherits from invokeai.backend.generator
"""

import math
from typing import Callable, Optional

import torch
from diffusers.utils.logging import get_verbosity, set_verbosity, set_verbosity_error

from ..stable_diffusion import PostprocessingSettings
from .base import Generator
from ..stable_diffusion.diffusers_pipeline import StableDiffusionGeneratorPipeline
from ..stable_diffusion.diffusers_pipeline import ConditioningData
from ..stable_diffusion.diffusers_pipeline import trim_to_multiple_of

import invokeai.backend.util.logging as logger

class Txt2Img2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)
        self.init_latent = None  # for get_noise()

    def get_make_image(
        self,
        prompt: str,
        sampler,
        steps: int,
        cfg_scale: float,
        ddim_eta,
        conditioning,
        width: int,
        height: int,
        strength: float,
        step_callback: Optional[Callable] = None,
        threshold=0.0,
        warmup=0.2,
        perlin=0.0,
        h_symmetry_time_pct=None,
        v_symmetry_time_pct=None,
        attention_maps_callback=None,
        **kwargs,
    ):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        kwargs are 'width' and 'height'
        """
        self.perlin = perlin

        # noinspection PyTypeChecker
        pipeline: StableDiffusionGeneratorPipeline = self.model
        pipeline.scheduler = sampler

        uc, c, extra_conditioning_info = conditioning
        conditioning_data = ConditioningData(
            uc,
            c,
            cfg_scale,
            extra_conditioning_info,
            postprocessing_settings=PostprocessingSettings(
                threshold=threshold,
                warmup=0.2,
                h_symmetry_time_pct=h_symmetry_time_pct,
                v_symmetry_time_pct=v_symmetry_time_pct,
            ),
        ).add_scheduler_args_if_applicable(pipeline.scheduler, eta=ddim_eta)

        def make_image(x_T: torch.Tensor, _: int):
            first_pass_latent_output, _ = pipeline.latents_from_embeddings(
                latents=torch.zeros_like(x_T),
                num_inference_steps=steps,
                conditioning_data=conditioning_data,
                noise=x_T,
                callback=step_callback,
            )

            # Get our initial generation width and height directly from the latent output so
            # the message below is accurate.
            init_width = first_pass_latent_output.size()[3] * self.downsampling_factor
            init_height = first_pass_latent_output.size()[2] * self.downsampling_factor
            logger.info(
                f"Interpolating from {init_width}x{init_height} to {width}x{height} using DDIM sampling"
            )

            # resizing
            resized_latents = torch.nn.functional.interpolate(
                first_pass_latent_output,
                size=(
                    height // self.downsampling_factor,
                    width // self.downsampling_factor,
                ),
                mode="bilinear",
            )

            # Free up memory from the last generation.
            clear_cuda_cache = kwargs["clear_cuda_cache"] or None
            if clear_cuda_cache is not None:
                clear_cuda_cache()

            second_pass_noise = self.get_noise_like(
                resized_latents, override_perlin=True
            )

            # Clear symmetry for the second pass
            from dataclasses import replace

            new_postprocessing_settings = replace(
                conditioning_data.postprocessing_settings, h_symmetry_time_pct=None
            )
            new_postprocessing_settings = replace(
                new_postprocessing_settings, v_symmetry_time_pct=None
            )
            new_conditioning_data = replace(
                conditioning_data, postprocessing_settings=new_postprocessing_settings
            )

            verbosity = get_verbosity()
            set_verbosity_error()
            pipeline_output = pipeline.img2img_from_latents_and_embeddings(
                resized_latents,
                num_inference_steps=steps,
                conditioning_data=new_conditioning_data,
                strength=strength,
                noise=second_pass_noise,
                callback=step_callback,
            )
            set_verbosity(verbosity)

            if (
                pipeline_output.attention_map_saver is not None
                and attention_maps_callback is not None
            ):
                attention_maps_callback(pipeline_output.attention_map_saver)

            return pipeline.numpy_to_pil(pipeline_output.images)[0]

        # FIXME: do we really need something entirely different for the inpainting model?

        # in the case of the inpainting model being loaded, the trick of
        # providing an interpolated latent doesn't work, so we transiently
        # create a 512x512 PIL image, upscale it, and run the inpainting
        # over it in img2img mode. Because the inpaing model is so conservative
        # it doesn't change the image (much)

        return make_image

    def get_noise_like(self, like: torch.Tensor, override_perlin: bool = False):
        device = like.device
        if device.type == "mps":
            x = torch.randn_like(like, device="cpu", dtype=self.torch_dtype()).to(
                device
            )
        else:
            x = torch.randn_like(like, device=device, dtype=self.torch_dtype())
        if self.perlin > 0.0 and override_perlin == False:
            shape = like.shape
            x = (1 - self.perlin) * x + self.perlin * self.get_perlin_noise(
                shape[3], shape[2]
            )
        return x

    # returns a tensor filled with random numbers from a normal distribution
    def get_noise(self, width, height, scale=True):
        # print(f"Get noise: {width}x{height}")
        if scale:
            # Scale the input width and height for the initial generation
            # Make their area equivalent to the model's resolution area (e.g. 512*512 = 262144),
            # while keeping the minimum dimension at least 0.5 * resolution (e.g. 512*0.5 = 256)

            aspect = width / height
            dimension = self.model.unet.config.sample_size * self.model.vae_scale_factor
            min_dimension = math.floor(dimension * 0.5)
            model_area = (
                dimension * dimension
            )  # hardcoded for now since all models are trained on square images

            if aspect > 1.0:
                init_height = max(min_dimension, math.sqrt(model_area / aspect))
                init_width = init_height * aspect
            else:
                init_width = max(min_dimension, math.sqrt(model_area * aspect))
                init_height = init_width / aspect

            scaled_width, scaled_height = trim_to_multiple_of(
                math.floor(init_width), math.floor(init_height)
            )

        else:
            scaled_width = width
            scaled_height = height

        device = self.model.device
        channels = self.latent_channels
        if channels == 9:
            channels = 4  # we don't really want noise for all the mask channels
        shape = (
            1,
            channels,
            scaled_height // self.downsampling_factor,
            scaled_width // self.downsampling_factor,
        )
        if self.use_mps_noise or device.type == "mps":
            tensor = torch.empty(size=shape, device="cpu")
            tensor = self.get_noise_like(like=tensor).to(device)
        else:
            tensor = torch.empty(size=shape, device=device)
            tensor = self.get_noise_like(like=tensor)
        return tensor
