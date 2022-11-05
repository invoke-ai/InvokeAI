'''
ldm.invoke.generator.inpaint descends from ldm.invoke.generator
'''

import math
import torch
import torchvision.transforms as T
import numpy as  np
import cv2 as cv
import PIL
from PIL import Image, ImageFilter, ImageOps
from skimage.exposure.histogram_matching import match_histograms
from einops import rearrange, repeat
from ldm.invoke.devices             import choose_autocast
from ldm.invoke.generator.img2img   import Img2Img
from ldm.models.diffusion.ddim     import DDIMSampler
from ldm.models.diffusion.ksampler import KSampler
from ldm.invoke.generator.base import downsampling

class Inpaint(Img2Img):
    def __init__(self, model, precision):
        self.init_latent = None
        self.pil_image = None
        self.pil_mask = None
        self.mask_blur_radius = 0
        super().__init__(model, precision)

    # Outpaint support code
    def get_tile_images(self, image: np.ndarray, width=8, height=8):
        _nrows, _ncols, depth = image.shape
        _strides = image.strides

        nrows, _m = divmod(_nrows, height)
        ncols, _n = divmod(_ncols, width)
        if _m != 0 or _n != 0:
            return None

        return np.lib.stride_tricks.as_strided(
            np.ravel(image),
            shape=(nrows, ncols, height, width, depth),
            strides=(height * _strides[0], width * _strides[1], *_strides),
            writeable=False
        )

    def tile_fill_missing(self, im: Image.Image, tile_size: int = 16, seed: int = None) -> Image:
        # Only fill if there's an alpha layer
        if im.mode != 'RGBA':
            return im

        a = np.asarray(im, dtype=np.uint8)

        tile_size = (tile_size, tile_size)

        # Get the image as tiles of a specified size
        tiles = self.get_tile_images(a,*tile_size).copy()

        # Get the mask as tiles
        tiles_mask = tiles[:,:,:,:,3]

        # Find any mask tiles with any fully transparent pixels (we will be replacing these later)
        tmask_shape = tiles_mask.shape
        tiles_mask = tiles_mask.reshape(math.prod(tiles_mask.shape))
        n,ny = (math.prod(tmask_shape[0:2])), math.prod(tmask_shape[2:])
        tiles_mask = (tiles_mask > 0)
        tiles_mask = tiles_mask.reshape((n,ny)).all(axis = 1)

        # Get RGB tiles in single array and filter by the mask
        tshape = tiles.shape
        tiles_all = tiles.reshape((math.prod(tiles.shape[0:2]), * tiles.shape[2:]))
        filtered_tiles = tiles_all[tiles_mask]

        if len(filtered_tiles) == 0:
            return im

        # Find all invalid tiles and replace with a random valid tile
        replace_count = (tiles_mask == False).sum()
        rng = np.random.default_rng(seed = seed)
        tiles_all[np.logical_not(tiles_mask)] = filtered_tiles[rng.choice(filtered_tiles.shape[0], replace_count),:,:,:]

        # Convert back to an image
        tiles_all = tiles_all.reshape(tshape)
        tiles_all = tiles_all.swapaxes(1,2)
        st = tiles_all.reshape((math.prod(tiles_all.shape[0:2]), math.prod(tiles_all.shape[2:4]), tiles_all.shape[4]))
        si = Image.fromarray(st, mode='RGBA')

        return si


    def mask_edge(self, mask: Image, edge_size: int, edge_blur: int) -> Image:
        npimg = np.asarray(mask, dtype=np.uint8)

        # Detect any partially transparent regions
        npgradient = np.uint8(255 * (1.0 - np.floor(np.abs(0.5 - np.float32(npimg) / 255.0) * 2.0)))

        # Detect hard edges
        npedge = cv.Canny(npimg, threshold1=100, threshold2=200)

        # Combine
        npmask = npgradient + npedge

        # Expand 
        npmask = cv.dilate(npmask, np.ones((3,3), np.uint8), iterations = int(edge_size / 2))

        new_mask = Image.fromarray(npmask)

        if edge_blur > 0:
            new_mask = new_mask.filter(ImageFilter.BoxBlur(edge_blur))

        return ImageOps.invert(new_mask)


    def seam_paint(self,
        im: Image.Image,
        seam_size: int,
        seam_blur: int,
        prompt,sampler,steps,cfg_scale,ddim_eta,
        conditioning,strength,
        noise
    ) -> Image.Image:
        hard_mask = self.pil_image.split()[-1].copy()
        mask = self.mask_edge(hard_mask, seam_size, seam_blur)

        make_image = self.get_make_image(
            prompt,
            sampler,
            steps,
            cfg_scale,
            ddim_eta,
            conditioning,
            init_image = im.copy().convert('RGBA'),
            mask_image = mask.convert('RGB'), # Code currently requires an RGB mask
            strength = strength,
            mask_blur_radius = 0,
            seam_size = 0
        )

        result = make_image(noise)

        return result


    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,mask_image,strength,
                       mask_blur_radius: int = 8,
                       # Seam settings - when 0, doesn't fill seam
                       seam_size: int = 0,
                       seam_blur: int = 0,
                       seam_strength: float = 0.7,
                       seam_steps: int = 10,
                       tile_size: int = 32,
                       step_callback=None,
                       inpaint_replace=False, **kwargs):
        """
        Returns a function returning an image derived from the prompt and
        the initial image + mask.  Return value depends on the seed at
        the time you call it.  kwargs are 'init_latent' and 'strength'
        """

        if isinstance(init_image, PIL.Image.Image):
            self.pil_image = init_image

            # Fill missing areas of original image
            init_filled = self.tile_fill_missing(
                self.pil_image.copy(),
                seed = self.seed,
                tile_size = tile_size
            )
            init_filled.paste(init_image, (0,0), init_image.split()[-1])

            # Create init tensor
            init_image = self._image_to_tensor(init_filled.convert('RGB'))

        if isinstance(mask_image, PIL.Image.Image):
            self.pil_mask = mask_image
            mask_image = mask_image.resize(
                (
                    mask_image.width // downsampling,
                    mask_image.height // downsampling
                ),
                resample=Image.Resampling.NEAREST
            )
            mask_image = self._image_to_tensor(mask_image,normalize=False)

        self.mask_blur_radius = mask_blur_radius

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
        # todo: support cross-attention control
        uc, c, _ = conditioning

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

            result = self.sample_to_image(samples)

            # Seam paint if this is our first pass (seam_size set to 0 during seam painting)
            if seam_size > 0:
                result = self.seam_paint(
                    result,
                    seam_size,
                    seam_blur,
                    prompt,
                    sampler,
                    seam_steps,
                    cfg_scale,
                    ddim_eta,
                    conditioning,
                    seam_strength,
                    x_T)

            return result

        return make_image


    def color_correct(self, image: Image.Image, base_image: Image.Image, mask: Image.Image, mask_blur_radius: int) -> Image.Image:
        # Get the original alpha channel of the mask if there is one.
        # Otherwise it is some other black/white image format ('1', 'L' or 'RGB')
        pil_init_mask = mask.getchannel('A') if mask.mode == 'RGBA' else mask.convert('L')
        pil_init_image = base_image.convert('RGBA') # Add an alpha channel if one doesn't exist

        # Build an image with only visible pixels from source to use as reference for color-matching.
        init_rgb_pixels = np.asarray(base_image.convert('RGB'), dtype=np.uint8)
        init_a_pixels = np.asarray(pil_init_image.getchannel('A'), dtype=np.uint8)
        init_mask_pixels = np.asarray(pil_init_mask, dtype=np.uint8)

        # Get numpy version of result
        np_image = np.asarray(image, dtype=np.uint8)

        # Mask and calculate mean and standard deviation
        mask_pixels = init_a_pixels * init_mask_pixels > 0
        np_init_rgb_pixels_masked = init_rgb_pixels[mask_pixels, :]
        np_image_masked = np_image[mask_pixels, :]

        init_means = np_init_rgb_pixels_masked.mean(axis=0)
        init_std = np_init_rgb_pixels_masked.std(axis=0)
        gen_means = np_image_masked.mean(axis=0)
        gen_std = np_image_masked.std(axis=0)

        # Color correct
        np_matched_result = np_image.copy()
        np_matched_result[:,:,:] = (((np_matched_result[:,:,:].astype(np.float32) - gen_means[None,None,:]) / gen_std[None,None,:]) * init_std[None,None,:] + init_means[None,None,:]).clip(0, 255).astype(np.uint8)
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
        matched_result.paste(base_image, (0,0), mask = blurred_init_mask)
        return matched_result


    def sample_to_image(self, samples)->Image.Image:
        gen_result = super().sample_to_image(samples).convert('RGB')

        if self.pil_image is None or self.pil_mask is None:
            return gen_result
        
        corrected_result = self.color_correct(gen_result, self.pil_image, self.pil_mask, self.mask_blur_radius)

        return corrected_result
