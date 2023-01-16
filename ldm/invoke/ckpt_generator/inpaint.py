'''
ldm.invoke.ckpt_generator.inpaint descends from ldm.invoke.ckpt_generator
'''

import math
import torch
import torchvision.transforms as T
import numpy as  np
import cv2 as cv
import PIL
from PIL import Image, ImageFilter, ImageOps, ImageChops
from skimage.exposure.histogram_matching import match_histograms
from einops import rearrange, repeat
from ldm.invoke.devices             import choose_autocast
from ldm.invoke.ckpt_generator.img2img   import CkptImg2Img
from ldm.models.diffusion.ddim     import DDIMSampler
from ldm.models.diffusion.ksampler import KSampler
from ldm.invoke.generator.base import downsampling
from ldm.util import debug_image
from ldm.invoke.patchmatch import PatchMatch 
from ldm.invoke.globals import Globals

def infill_methods()->list[str]:
    methods = list()
    if PatchMatch.patchmatch_available():
        methods.append('patchmatch')
    methods.append('tile')
    return methods

class CkptInpaint(CkptImg2Img):
    def __init__(self, model, precision):
        self.init_latent = None
        self.pil_image = None
        self.pil_mask = None
        self.mask_blur_radius = 0
        self.infill_method = None
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

    def infill_patchmatch(self, im: Image.Image) -> Image:
        if im.mode != 'RGBA':
            return im

        # Skip patchmatch if patchmatch isn't available
        if not PatchMatch.patchmatch_available():
            return im

        # Patchmatch (note, we may want to expose patch_size? Increasing it significantly impacts performance though)
        im_patched_np = PatchMatch.inpaint(im.convert('RGB'), ImageOps.invert(im.split()[-1]), patch_size = 3)
        im_patched = Image.fromarray(im_patched_np, mode = 'RGB')
        return im_patched

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
        noise,
        step_callback
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
            seam_size = 0,
            step_callback = step_callback,
            inpaint_width = im.width,
            inpaint_height = im.height
        )

        seam_noise = self.get_noise(im.width, im.height)

        result = make_image(seam_noise)

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
                       inpaint_replace=False, enable_image_debugging=False,
                       infill_method = None,
                       inpaint_width=None,
                       inpaint_height=None,
                       **kwargs):
        """
        Returns a function returning an image derived from the prompt and
        the initial image + mask.  Return value depends on the seed at
        the time you call it.  kwargs are 'init_latent' and 'strength'
        """

        self.enable_image_debugging = enable_image_debugging
        self.infill_method = infill_method or infill_methods()[0], # The infill method to use
        
        self.inpaint_width = inpaint_width
        self.inpaint_height = inpaint_height

        if isinstance(init_image, PIL.Image.Image):
            self.pil_image = init_image.copy()

            # Do infill
            if infill_method == 'patchmatch' and PatchMatch.patchmatch_available():
                init_filled = self.infill_patchmatch(self.pil_image.copy())
            else: # if infill_method == 'tile': # Only two methods right now, so always use 'tile' if not patchmatch
                init_filled = self.tile_fill_missing(
                    self.pil_image.copy(),
                    seed = self.seed,
                    tile_size = tile_size
                )
            init_filled.paste(init_image, (0,0), init_image.split()[-1])

            # Resize if requested for inpainting
            if inpaint_width and inpaint_height:
                init_filled = init_filled.resize((inpaint_width, inpaint_height))

            debug_image(init_filled, "init_filled", debug_status=self.enable_image_debugging)

            # Create init tensor
            init_image = self._image_to_tensor(init_filled.convert('RGB'))

        if isinstance(mask_image, PIL.Image.Image):
            self.pil_mask = mask_image.copy()
            debug_image(mask_image, "mask_image BEFORE multiply with pil_image", debug_status=self.enable_image_debugging)

            mask_image = ImageChops.multiply(mask_image, self.pil_image.split()[-1].convert('RGB'))
            self.pil_mask = mask_image

            # Resize if requested for inpainting
            if inpaint_width and inpaint_height:
                mask_image = mask_image.resize((inpaint_width, inpaint_height))

            debug_image(mask_image, "mask_image AFTER multiply with pil_image", debug_status=self.enable_image_debugging)
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
                torch.tensor([t_enc - 1]).to(self.model.device),
                noise=x_T
            )

            # to replace masked area with latent noise, weighted by inpaint_replace strength
            if inpaint_replace > 0.0:
                print(f'>> inpaint will replace what was under the mask with a strength of {inpaint_replace}')
                l_noise = self.get_noise(kwargs['width'],kwargs['height'])
                inverted_mask = 1.0-mask_image  # there will be 1s where the mask is
                masked_region = (1.0-inpaint_replace) * inverted_mask * z_enc + inpaint_replace * inverted_mask * l_noise
                z_enc   = z_enc * mask_image + masked_region

            if self.free_gpu_mem and self.model.model.device != self.model.device:
                self.model.model.to(self.model.device)

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
                old_image = self.pil_image or init_image
                old_mask = self.pil_mask or mask_image

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
                    x_T,
                    step_callback)

                # Restore original settings
                self.get_make_image(prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,
                       old_image,
                       old_mask,
                       strength,
                       mask_blur_radius, seam_size, seam_blur, seam_strength,
                       seam_steps, tile_size, step_callback,
                       inpaint_replace, enable_image_debugging,
                       inpaint_width = inpaint_width,
                       inpaint_height = inpaint_height,
                       infill_method = infill_method,
                       **kwargs)

            return result

        return make_image


    def sample_to_image(self, samples)->Image.Image:
        gen_result = super().sample_to_image(samples).convert('RGB')
        debug_image(gen_result, "gen_result", debug_status=self.enable_image_debugging)

        # Resize if necessary
        if self.inpaint_width and self.inpaint_height:
            gen_result = gen_result.resize(self.pil_image.size)

        if self.pil_image is None or self.pil_mask is None:
            return gen_result

        corrected_result = super().repaste_and_color_correct(gen_result, self.pil_image, self.pil_mask, self.mask_blur_radius)
        debug_image(corrected_result, "corrected_result", debug_status=self.enable_image_debugging)

        return corrected_result
