'''
ldm.invoke.ckpt_generator.txt2img inherits from ldm.invoke.ckpt_generator
'''

import torch
import numpy as  np
from ldm.invoke.ckpt_generator.base import CkptGenerator
from ldm.models.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent
import gc


class CkptTxt2Img(CkptGenerator):
    def __init__(self, model, precision):
        super().__init__(model, precision)

    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,width,height,step_callback=None,threshold=0.0,perlin=0.0,
                       attention_maps_callback=None,
                       **kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it
        kwargs are 'width' and 'height'
        """
        self.perlin = perlin
        uc, c, extra_conditioning_info   = conditioning

        @torch.no_grad()
        def make_image(x_T):
            shape = [
                self.latent_channels,
                height // self.downsampling_factor,
                width  // self.downsampling_factor,
            ]

            if self.free_gpu_mem and self.model.model.device != self.model.device:
                self.model.model.to(self.model.device)

            sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False)

            samples, _ = sampler.sample(
                batch_size                   = 1,
                S                            = steps,
                x_T                          = x_T,
                conditioning                 = c,
                shape                        = shape,
                verbose                      = False,
                unconditional_guidance_scale = cfg_scale,
                unconditional_conditioning   = uc,
                extra_conditioning_info      = extra_conditioning_info,
                eta                          = ddim_eta,
                img_callback                 = step_callback,
                threshold                    = threshold,
                attention_maps_callback      = attention_maps_callback,
            )

            if self.free_gpu_mem:
                self.model.model.to('cpu')
                self.model.cond_stage_model.device = 'cpu'
                self.model.cond_stage_model.to('cpu')
                gc.collect()
                torch.cuda.empty_cache()

            return self.sample_to_image(samples)

        return make_image


    # returns a tensor filled with random numbers from a normal distribution
    def get_noise(self,width,height):
        device         = self.model.device
        if self.use_mps_noise or device.type == 'mps':
            x = torch.randn([1,
                             self.latent_channels,
                             height // self.downsampling_factor,
                             width  // self.downsampling_factor],
                            dtype=self.torch_dtype(),
                            device='cpu').to(device)
        else:
            x = torch.randn([1,
                             self.latent_channels,
                             height // self.downsampling_factor,
                             width  // self.downsampling_factor],
                            dtype=self.torch_dtype(),
                            device=device)
        if self.perlin > 0.0:
            x = (1-self.perlin)*x + self.perlin*self.get_perlin_noise(width  // self.downsampling_factor, height // self.downsampling_factor)
        return x

