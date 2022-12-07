'''
ldm.invoke.generator.txt2img inherits from ldm.invoke.generator
'''

import math
from typing import Callable, Optional

import torch

from ldm.invoke.generator.base import Generator
from ldm.invoke.generator.diffusers_pipeline import trim_to_multiple_of, StableDiffusionGeneratorPipeline


class Txt2Img2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)
        self.init_latent = None    # for get_noise()

    def get_make_image(self, prompt:str, sampler, steps:int, cfg_scale:float, ddim_eta,
                       conditioning, width:int, height:int, strength:float,
                       step_callback:Optional[Callable]=None, **kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        kwargs are 'width' and 'height'
        """
        uc, c, extra_conditioning_info = conditioning
        scale_dim = min(width, height)
        scale = 512 / scale_dim

        init_width, init_height = trim_to_multiple_of(scale * width, scale * height)

        # noinspection PyTypeChecker
        pipeline: StableDiffusionGeneratorPipeline = self.model
        pipeline.scheduler = sampler

        def make_image(x_T):

            pipeline_output = pipeline.latents_from_embeddings(
                latents=x_T,
                num_inference_steps=steps,
                text_embeddings=c,
                unconditioned_embeddings=uc,
                guidance_scale=cfg_scale,
                callback=step_callback,
                extra_conditioning_info=extra_conditioning_info,
                # TODO: eta = ddim_eta,
                # TODO: threshold = threshold,
            )

            first_pass_latent_output = pipeline_output.latents

            print(
                  f"\n>> Interpolating from {init_width}x{init_height} to {width}x{height} using DDIM sampling"
                 )

            # resizing
            resized_latents = torch.nn.functional.interpolate(
                first_pass_latent_output,
                size=(height // self.downsampling_factor, width // self.downsampling_factor),
                mode="bilinear"
            )

            pipeline_output = pipeline.img2img_from_latents_and_embeddings(
                resized_latents,
                num_inference_steps=steps,
                text_embeddings=c,
                unconditioned_embeddings=uc,
                guidance_scale=cfg_scale, strength=strength,
                extra_conditioning_info=extra_conditioning_info,
                noise_func=self.get_noise_like,
                callback=step_callback)

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
            x = torch.randn_like(like, device='cpu').to(device)
        else:
            x = torch.randn_like(like, device=device)
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
                                device='cpu').to(device)
        else:
            return torch.randn([1,
                                self.latent_channels,
                                scaled_height // self.downsampling_factor,
                                scaled_width  // self.downsampling_factor],
                                device=device)
