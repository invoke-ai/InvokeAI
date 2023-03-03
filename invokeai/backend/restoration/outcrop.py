import warnings
import math
from PIL import Image, ImageFilter

class Outcrop(object):
    def __init__(
            self,
            image,
            generate,  # current generate object
    ):
        self.image     = image
        self.generate  = generate

    def process (
            self,
            extents:dict,
            opt,                   # current options
            orig_opt,              # ones originally used to generate the image
            image_callback = None,
            prefix = None
    ):
        # grow and mask the image
        extended_image = self._extend_all(extents)

        # switch samplers temporarily
        curr_sampler = self.generate.sampler
        self.generate.sampler_name = opt.sampler_name
        self.generate._set_sampler()

        def wrapped_callback(img,seed,**kwargs):
            preferred_seed = orig_opt.seed if orig_opt.seed is not None and orig_opt.seed >= 0 else seed
            image_callback(img,preferred_seed,use_prefix=prefix,**kwargs)

        result= self.generate.prompt2image(
            opt.prompt,
            seed        = opt.seed or orig_opt.seed,
            sampler     = self.generate.sampler,
            steps       = opt.steps,
            cfg_scale   = opt.cfg_scale,
            ddim_eta    = self.generate.ddim_eta,
            width       = extended_image.width,
            height      = extended_image.height,
            init_img    = extended_image,
            strength    = 0.90,
            image_callback = wrapped_callback if image_callback else None,
            seam_size = opt.seam_size or 96,
            seam_blur = opt.seam_blur or 16,
            seam_strength = opt.seam_strength or 0.7,
            seam_steps = 20,
            tile_size = 32,
            color_match = True,
            force_outpaint = True,  # this just stops the warning about erased regions
        )

        # swap sampler back
        self.generate.sampler = curr_sampler
        return result

    def _extend_all(
            self,
            extents:dict,
    ) -> Image:
        '''
        Extend the image in direction ('top','bottom','left','right') by
        the indicated value. The image canvas is extended, and the empty
        rectangular section will be filled with a blurred copy of the
        adjacent image.
        '''
        image = self.image
        for direction in extents:
            assert direction in ['top', 'left', 'bottom', 'right'],'Direction must be one of "top", "left", "bottom", "right"'
            pixels = extents[direction]
            # round pixels up to the nearest 64
            pixels = math.ceil(pixels/64) * 64
            print(f'>> extending image {direction}ward by {pixels} pixels')
            image = self._rotate(image,direction)
            image = self._extend(image,pixels)
            image = self._rotate(image,direction,reverse=True)
        return image

    def _rotate(self,image:Image,direction:str,reverse=False) -> Image:
        '''
        Rotates image so that the area to extend is always at the top top.
        Simplifies logic later. The reverse argument, if true, will undo the
        previous transpose.
        '''
        transposes = {
            'right':  ['ROTATE_90','ROTATE_270'],
            'bottom': ['ROTATE_180','ROTATE_180'],
            'left':   ['ROTATE_270','ROTATE_90']
        }
        if direction not in transposes:
            return image
        transpose = transposes[direction][1 if reverse else 0]
        return image.transpose(Image.Transpose.__dict__[transpose])

    def _extend(self,image:Image,pixels:int)-> Image:
        extended_img = Image.new('RGBA',(image.width,image.height+pixels))

        extended_img.paste((0,0,0),[0,0,image.width,image.height+pixels])
        extended_img.paste(image,box=(0,pixels))

        # now make the top part transparent to use as a mask
        alpha = extended_img.getchannel('A')
        alpha.paste(0,(0,0,extended_img.width,pixels))
        extended_img.putalpha(alpha)

        return extended_img
