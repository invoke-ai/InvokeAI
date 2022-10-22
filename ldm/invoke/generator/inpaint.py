'''
ldm.invoke.generator.inpaint descends from ldm.invoke.generator
'''

import torch
import numpy as  np
import PIL
from PIL import Image, ImageFilter
from einops import rearrange, repeat
from ldm.invoke.devices import choose_autocast
from ldm.invoke.generator.base import downsampling
from ldm.invoke.generator.img2img import Img2Img
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ksampler import KSampler

class Inpaint(Img2Img):
    def __init__(self, model, precision):
        self.init_latent = None
        self.original_image = None
        self.original_mask = None
        super().__init__(model, precision)

    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,mask_image,strength,
                       step_callback=None,inpaint_replace=False,**kwargs):
        """
        Returns a function returning an image derived from the prompt and
        the initial image + mask.  Return value depends on the seed at
        the time you call it.  kwargs are 'init_latent' and 'strength'
        """
        # klms samplers not supported yet, so ignore previous sampler
        if isinstance(sampler,KSampler):
            print(
                f">> Using recommended DDIM sampler for inpainting."
            )
            sampler = DDIMSampler(self.model, device=self.model.device)
        
        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        if isinstance(init_image, PIL.Image.Image):
            self.original_image = init_image
            init_image = self._image_to_tensor(init_image)

        if isinstance(mask_image, PIL.Image.Image):
            self.original_mask = mask_image
            mask_image = mask_image.resize(
                (
                    mask_image.width // downsampling,
                    mask_image.height // downsampling
                ),
                resample=Image.Resampling.NEAREST
            )
            mask_image = self._image_to_tensor(mask_image,normalize=False)

        mask_image = mask_image[0][0].unsqueeze(0).repeat(4,1,1).unsqueeze(0)
        mask_image = repeat(mask_image, '1 ... -> b ...', b=1)

        scope = choose_autocast(self.precision)
        with scope(self.model.device.type):
            self.init_latent = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_image)
            ) # move to latent space

        t_enc   = int(strength * steps)
        uc, c   = conditioning

        print(f">> target t_enc is {t_enc} steps")

        @torch.no_grad()
        def make_image(x_T):
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                self.init_latent,
                torch.tensor([t_enc]).to(self.model.device),
                noise=x_T
            )

            # to replace masked area with latent noise, weighted by inpaint_replace strength
            if inpaint_replace > 0.0:
                print(f'>> inpaint will replace what was under the mask with a strength of {inpaint_replace}')
                l_noise = self.get_noise(kwargs['width'],kwargs['height'])
                inverted_mask = 1.0-mask_image  # there will be 1s where the mask is
                masked_region = (1.0-inpaint_replace) * inverted_mask * z_enc + inpaint_replace * inverted_mask * l_noise
                z_enc   = z_enc * mask_image + masked_region

            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                img_callback                 = step_callback,
                unconditional_guidance_scale = cfg_scale,
                unconditional_conditioning = uc,
                mask                       = mask_image,
                init_latent                = self.init_latent
            )

            return self.sample_to_image(samples)

        return make_image


    def sample_to_image(self, samples) -> Image:
        painted_image = super().sample_to_image(samples)
        if self.original_image is not None and self.original_mask is not None:

            print('>> Restoring unmasked regions of initial image')
            mask = self.original_mask.convert('L')
            blur = mask.filter(filter=ImageFilter.GaussianBlur(radius=2))
            painted_image.paste(self.original_image, (0,0), blur)

        return painted_image

