'''
ldm.invoke.generator.inpaint descends from ldm.invoke.generator
'''
from __future__ import annotations

import math

import PIL
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps, ImageChops

from ldm.invoke.generator.diffusers_pipeline import image_resized_to_grid_as_tensor, StableDiffusionGeneratorPipeline, \
    ConditioningData
from ldm.invoke.generator.img2img import Img2Img
from ldm.invoke.patchmatch import PatchMatch
from ldm.util import debug_image


def infill_methods()->list[str]:
    methods = list()
    if PatchMatch.patchmatch_available():
        methods.append('patchmatch')
    methods.append('tile')
    return methods

class Inpaint(Img2Img):
    def __init__(self, model, precision):
        self.inpaint_height = 0
        self.inpaint_width = 0
        self.enable_image_debugging = False
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
        npedge = cv2.Canny(npimg, threshold1=100, threshold2=200)

        # Combine
        npmask = npgradient + npedge

        # Expand
        npmask = cv2.dilate(npmask, np.ones((3,3), np.uint8), iterations = int(edge_size / 2))

        new_mask = Image.fromarray(npmask)

        if edge_blur > 0:
            new_mask = new_mask.filter(ImageFilter.BoxBlur(edge_blur))

        return ImageOps.invert(new_mask)


    def seam_paint(self, im: Image.Image, seam_size: int, seam_blur: int, prompt, sampler, steps, cfg_scale, ddim_eta,
                   conditioning, strength, noise, infill_method, step_callback) -> Image.Image:
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
            mask_image = mask,
            strength = strength,
            mask_blur_radius = 0,
            seam_size = 0,
            step_callback = step_callback,
            inpaint_width = im.width,
            inpaint_height = im.height,
            infill_method = infill_method
        )

        seam_noise = self.get_noise(im.width, im.height)

        result = make_image(seam_noise)

        return result


    @torch.no_grad()
    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,
                       init_image: PIL.Image.Image | torch.FloatTensor,
                       mask_image: PIL.Image.Image | torch.FloatTensor,
                       strength: float,
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
                       attention_maps_callback=None,
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
            init_image = image_resized_to_grid_as_tensor(init_filled.convert('RGB'))

        if isinstance(mask_image, PIL.Image.Image):
            self.pil_mask = mask_image.copy()
            debug_image(mask_image, "mask_image BEFORE multiply with pil_image", debug_status=self.enable_image_debugging)

            init_alpha = self.pil_image.getchannel("A")
            if mask_image.mode != "L":
                # FIXME: why do we get passed an RGB image here? We can only use single-channel.
                mask_image = mask_image.convert("L")
            mask_image = ImageChops.multiply(mask_image, init_alpha)
            self.pil_mask = mask_image

            # Resize if requested for inpainting
            if inpaint_width and inpaint_height:
                mask_image = mask_image.resize((inpaint_width, inpaint_height))

            debug_image(mask_image, "mask_image AFTER multiply with pil_image", debug_status=self.enable_image_debugging)
            mask: torch.FloatTensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)
        else:
            mask: torch.FloatTensor = mask_image

        self.mask_blur_radius = mask_blur_radius

        # noinspection PyTypeChecker
        pipeline: StableDiffusionGeneratorPipeline = self.model
        pipeline.scheduler = sampler

        # todo: support cross-attention control
        uc, c, _ = conditioning
        conditioning_data = (ConditioningData(uc, c, cfg_scale)
                             .add_scheduler_args_if_applicable(pipeline.scheduler, eta=ddim_eta))


        def make_image(x_T):
            pipeline_output = pipeline.inpaint_from_embeddings(
                init_image=init_image,
                mask=1 - mask,  # expects white means "paint here."
                strength=strength,
                num_inference_steps=steps,
                conditioning_data=conditioning_data,
                noise_func=self.get_noise_like,
                callback=step_callback,
            )

            if pipeline_output.attention_map_saver is not None and attention_maps_callback is not None:
                attention_maps_callback(pipeline_output.attention_map_saver)

            result = self.postprocess_size_and_mask(pipeline.numpy_to_pil(pipeline_output.images)[0])

            # Seam paint if this is our first pass (seam_size set to 0 during seam painting)
            if seam_size > 0:
                old_image = self.pil_image or init_image
                old_mask = self.pil_mask or mask_image

                result = self.seam_paint(result, seam_size, seam_blur, prompt, sampler, seam_steps, cfg_scale, ddim_eta,
                                         conditioning, seam_strength, x_T, infill_method, step_callback)

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
        return self.postprocess_size_and_mask(gen_result)


    def postprocess_size_and_mask(self, gen_result: Image.Image) -> Image.Image:
        debug_image(gen_result, "gen_result", debug_status=self.enable_image_debugging)

        # Resize if necessary
        if self.inpaint_width and self.inpaint_height:
            gen_result = gen_result.resize(self.pil_image.size)

        if self.pil_image is None or self.pil_mask is None:
            return gen_result

        corrected_result = self.repaste_and_color_correct(gen_result, self.pil_image, self.pil_mask, self.mask_blur_radius)
        debug_image(corrected_result, "corrected_result", debug_status=self.enable_image_debugging)

        return corrected_result
