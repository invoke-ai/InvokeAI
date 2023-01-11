'''
ldm.invoke.ckpt_generator.img2img descends from ldm.invoke.generator
'''

import torch
import numpy as  np
import PIL
from torch import Tensor
from PIL import Image
from ldm.invoke.devices import choose_autocast
from ldm.invoke.ckpt_generator.base import CkptGenerator
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent

class CkptImg2Img(CkptGenerator):
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

        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        if isinstance(init_image, PIL.Image.Image):
            init_image = self._image_to_tensor(init_image.convert('RGB'))

        scope = choose_autocast(self.precision)
        with scope(self.model.device.type):
            self.init_latent = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_image)
            ) # move to latent space

        t_enc = int(strength * steps)
        uc, c, extra_conditioning_info   = conditioning

        def make_image(x_T):
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                self.init_latent,
                torch.tensor([t_enc - 1]).to(self.model.device),
                noise=x_T
            )

            if self.free_gpu_mem and self.model.model.device != self.model.device:
                self.model.model.to(self.model.device)

            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                img_callback = step_callback,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
                init_latent = self.init_latent, # changes how noising is performed in ksampler
                extra_conditioning_info = extra_conditioning_info,
                all_timesteps_count = steps
            )

            if self.free_gpu_mem:
                self.model.model.to("cpu")

            return self.sample_to_image(samples)

        return make_image

    def get_noise(self,width,height):
        device      = self.model.device
        init_latent = self.init_latent
        assert init_latent is not None,'call to get_noise() when init_latent not set'
        if device.type == 'mps':
            x = torch.randn_like(init_latent, device='cpu').to(device)
        else:
            x = torch.randn_like(init_latent, device=device)
        if self.perlin > 0.0:
            shape = init_latent.shape
            x = (1-self.perlin)*x + self.perlin*self.get_perlin_noise(shape[3], shape[2])
        return x

    def _image_to_tensor(self, image:Image, normalize:bool=True)->Tensor:
        image = np.array(image).astype(np.float32) / 255.0
        if len(image.shape) == 2:  # 'L' image, as in a mask
            image = image[None,None]
        else:                      # 'RGB' image
            image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        if normalize:
            image = 2.0 * image - 1.0
        return image.to(self.model.device)
