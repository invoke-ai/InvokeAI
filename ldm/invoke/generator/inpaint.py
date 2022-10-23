'''
ldm.invoke.generator.inpaint descends from ldm.invoke.generator
'''

import torch
import torchvision.transforms as T
import numpy as  np
import cv2 as cv
from PIL import Image, ImageFilter
from skimage.exposure.histogram_matching import match_histograms
from einops import rearrange, repeat
from ldm.invoke.devices             import choose_autocast
from ldm.invoke.generator.img2img   import Img2Img
from ldm.models.diffusion.ddim     import DDIMSampler
from ldm.models.diffusion.ksampler import KSampler

class Inpaint(Img2Img):
    def __init__(self, model, precision):
        self.init_latent = None
        super().__init__(model, precision)

    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,mask_image,strength,
                       pil_image: Image.Image, pil_mask: Image.Image,
                       mask_blur_radius: int = 8,
                       step_callback=None,inpaint_replace=False, **kwargs):
        """
        Returns a function returning an image derived from the prompt and
        the initial image + mask.  Return value depends on the seed at
        the time you call it.  kwargs are 'init_latent' and 'strength'
        """

        # Get the alpha channel of the mask
        pil_init_mask = pil_mask.getchannel('A')
        pil_init_image = pil_image.convert('RGBA') # Add an alpha channel if one doesn't exist

        # Build an image with only visible pixels from source to use as reference for color-matching.
        # Note that this doesn't use the mask, which would exclude some source image pixels from the
        # histogram and cause slight color changes.
        init_rgb_pixels = np.asarray(pil_image.convert('RGB'), dtype=np.uint8).reshape(pil_image.width * pil_image.height, 3)
        init_a_pixels = np.asarray(pil_init_image.getchannel('A'), dtype=np.uint8).reshape(pil_init_mask.width * pil_init_mask.height)
        init_rgb_pixels = init_rgb_pixels[init_a_pixels > 0]
        init_rgb_pixels = init_rgb_pixels.reshape(1, init_rgb_pixels.shape[0], init_rgb_pixels.shape[1]) # Filter to just pixels that have any alpha, this is now our histogram

        # klms samplers not supported yet, so ignore previous sampler
        if isinstance(sampler,KSampler):
            print(
                f">> Using recommended DDIM sampler for inpainting."
            )
            sampler = DDIMSampler(self.model, device=self.model.device)
        
        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

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

            # Get PIL result
            gen_result = self.sample_to_image(samples).convert('RGB')

            # Get numpy version
            np_gen_result = np.asarray(gen_result, dtype=np.uint8)

            # Color correct
            np_matched_result = match_histograms(np_gen_result, init_rgb_pixels, channel_axis=-1)
            matched_result = Image.fromarray(np_matched_result, mode='RGB')


            # Blur the mask out (into init image) by specified amount
            if mask_blur_radius > 0:
                nm = np.asarray(pil_init_mask, dtype=np.uint8)
                nmd = cv.erode(nm, kernel=np.ones((3,3), dtype=np.uint8), iterations=int(mask_blur_radius / 2))
                pmd = Image.fromarray(nmd, mode='L')
                blurred_init_mask = pmd.filter(ImageFilter.BoxBlur(mask_blur_radius))
            else:
                blurred_init_mask = pil_init_mask

            # Paste original on color-corrected generation (using blurred mask)
            matched_result.paste(pil_image, (0,0), mask = blurred_init_mask)

            return matched_result

        return make_image
