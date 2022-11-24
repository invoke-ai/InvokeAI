'''
ldm.invoke.generator.img2img descends from ldm.invoke.generator
'''

import torch

from ldm.invoke.generator.base import Generator
from ldm.invoke.generator.diffusers_pipeline import StableDiffusionGeneratorPipeline


class Img2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)
        self.init_latent = None    # by get_noise()

    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,strength,step_callback=None,threshold=0.0,perlin=0.0,**kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it.
        """
        self.perlin = perlin

        uc, c, extra_conditioning_info   = conditioning

        # noinspection PyTypeChecker
        pipeline: StableDiffusionGeneratorPipeline = self.model
        pipeline.scheduler = sampler

        def make_image(x_T):
            # FIXME: use x_T for initial seeded noise
            pipeline_output = pipeline.img2img_from_embeddings(
                init_image, strength, steps, c, uc, cfg_scale,
                extra_conditioning_info=extra_conditioning_info,
                noise_func=self.get_noise_like,
                callback=step_callback
            )

            return pipeline.numpy_to_pil(pipeline_output.images)[0]

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

    def get_noise(self,width,height):
        # copy of the Txt2Img.get_noise
        device         = self.model.device
        if self.use_mps_noise or device.type == 'mps':
            x = torch.randn([1,
                                self.latent_channels,
                                height // self.downsampling_factor,
                                width  // self.downsampling_factor],
                               device='cpu').to(device)
        else:
            x = torch.randn([1,
                                self.latent_channels,
                                height // self.downsampling_factor,
                                width  // self.downsampling_factor],
                               device=device)
        if self.perlin > 0.0:
            x = (1-self.perlin)*x + self.perlin*self.get_perlin_noise(width  // self.downsampling_factor, height // self.downsampling_factor)
        return x
