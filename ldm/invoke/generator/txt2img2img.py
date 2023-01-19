'''
ldm.invoke.generator.txt2img inherits from ldm.invoke.generator
'''

import math
from diffusers.utils.logging import get_verbosity, set_verbosity, set_verbosity_error
from typing import Callable, Optional

import torch

from ldm.invoke.generator.base import Generator
from ldm.invoke.generator.diffusers_pipeline import trim_to_multiple_of, StableDiffusionGeneratorPipeline, \
    ConditioningData
from ldm.models.diffusion.shared_invokeai_diffusion import ThresholdSettings


class Txt2Img2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)
        self.init_latent = None    # for get_noise()

    def get_make_image(self, prompt:str, sampler, steps:int, cfg_scale:float, ddim_eta,
                       conditioning, width:int, height:int, strength:float,
                       step_callback:Optional[Callable]=None, threshold=0.0, **kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        kwargs are 'width' and 'height'
        """

        # noinspection PyTypeChecker
        pipeline: StableDiffusionGeneratorPipeline = self.model
        pipeline.scheduler = sampler

        uc, c, extra_conditioning_info = conditioning
        conditioning_data = (
            ConditioningData(
                uc, c, cfg_scale, extra_conditioning_info,
                threshold = ThresholdSettings(threshold, warmup=0.2) if threshold else None)
            .add_scheduler_args_if_applicable(pipeline.scheduler, eta=ddim_eta))
        scale_dim = min(width, height)
        scale = 512 / scale_dim

        init_width, init_height = trim_to_multiple_of(scale * width, scale * height)

        def make_image(x_T):

            first_pass_latent_output, _ = pipeline.latents_from_embeddings(
                latents=torch.zeros_like(x_T),
                num_inference_steps=steps,
                conditioning_data=conditioning_data,
                noise=x_T,
                callback=step_callback,
                # TODO: threshold = threshold,
            )

            print(
                  f"\n>> Interpolating from {init_width}x{init_height} to {width}x{height} using DDIM sampling"
                 )

            # resizing
            resized_latents = torch.nn.functional.interpolate(
                first_pass_latent_output,
                size=(height // self.downsampling_factor, width // self.downsampling_factor),
                mode="bilinear"
            )

            second_pass_noise = self.get_noise_like(resized_latents)

            verbosity = get_verbosity()
            set_verbosity_error()
            pipeline_output = pipeline.img2img_from_latents_and_embeddings(
                resized_latents,
                num_inference_steps=steps,
                conditioning_data=conditioning_data,
                strength=strength,
                noise=second_pass_noise,
                callback=step_callback)
            set_verbosity(verbosity)

            return pipeline.numpy_to_pil(pipeline_output.images)[0]


        # FIXME: do we really need something entirely different for the inpainting model?

        # in the case of the inpainting model being loaded, the trick of
        # providing an interpolated latent doesn't work, so we transiently
        # create a 512x512 PIL image, upscale it, and run the inpainting
        # over it in img2img mode. Because the inpaing model is so conservative
        # it doesn't change the image (much)

        return make_image

    def get_noise_like(self, like: torch.Tensor):
        device = like.device
        if device.type == 'mps':
            x = torch.randn_like(like, device='cpu', dtype=self.torch_dtype()).to(device)
        else:
            x = torch.randn_like(like, device=device, dtype=self.torch_dtype())
        if self.perlin > 0.0:
            shape = like.shape
            x = (1-self.perlin)*x + self.perlin*self.get_perlin_noise(shape[3], shape[2])
        return x

    # returns a tensor filled with random numbers from a normal distribution
    def get_noise(self,width,height,scale = True):
        # print(f"Get noise: {width}x{height}")
        if scale:
            trained_square = 512 * 512
            actual_square = width * height
            scale = math.sqrt(trained_square / actual_square)
            scaled_width = math.ceil(scale * width / 64) * 64
            scaled_height = math.ceil(scale * height / 64) * 64
        else:
            scaled_width = width
            scaled_height = height

        device      = self.model.device
        if self.use_mps_noise or device.type == 'mps':
            return torch.randn([1,
                                self.latent_channels,
                                scaled_height // self.downsampling_factor,
                                scaled_width  // self.downsampling_factor],
                               dtype=self.torch_dtype(),
                               device='cpu').to(device)
        else:
            return torch.randn([1,
                                self.latent_channels,
                                scaled_height // self.downsampling_factor,
                                scaled_width  // self.downsampling_factor],
                               dtype=self.torch_dtype(),
                               device=device)
