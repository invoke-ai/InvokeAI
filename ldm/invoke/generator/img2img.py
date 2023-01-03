'''
ldm.invoke.generator.img2img descends from ldm.invoke.generator
'''

import torch
from diffusers import logging

from ldm.invoke.generator.base import Generator
from ldm.invoke.generator.diffusers_pipeline import StableDiffusionGeneratorPipeline, ConditioningData
from ldm.models.diffusion.shared_invokeai_diffusion import ThresholdSettings


class Img2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)
        self.init_latent = None    # by get_noise()

    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,strength,step_callback=None,threshold=0.0,perlin=0.0,
                       attention_maps_callback=None,
                       **kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it.
        """
        self.perlin = perlin

        # noinspection PyTypeChecker
        pipeline: StableDiffusionGeneratorPipeline = self.model
        pipeline.scheduler = sampler

        uc, c, extra_conditioning_info   = conditioning
        conditioning_data = (
            ConditioningData(
                uc, c, cfg_scale, extra_conditioning_info,
                threshold = ThresholdSettings(threshold, warmup=0.2) if threshold else None)
            .add_scheduler_args_if_applicable(pipeline.scheduler, eta=ddim_eta))


        def make_image(x_T):
            # FIXME: use x_T for initial seeded noise
            # We're not at the moment because the pipeline automatically resizes init_image if
            # necessary, which the x_T input might not match.
            logging.set_verbosity_error()   # quench safety check warnings
            pipeline_output = pipeline.img2img_from_embeddings(
                init_image, strength, steps, conditioning_data,
                noise_func=self.get_noise_like,
                callback=step_callback
            )
            if pipeline_output.attention_map_saver is not None and attention_maps_callback is not None:
                attention_maps_callback(pipeline_output.attention_map_saver)
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
