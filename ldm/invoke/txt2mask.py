'''Makes available the Txt2Mask class, which assists in the automatic
assignment of masks via text prompt using clipseg.

Here is typical usage:

    from ldm.invoke.txt2mask import Txt2Mask, SegmentedGrayscale
    from PIL import Image

    txt2mask = Txt2Mask(self.device)
    segmented = txt2mask.segment(Image.open('/path/to/img.png'),'a bagel')

    # this will return a grayscale Image of the segmented data
    grayscale = segmented.to_grayscale()

    # this will return a semi-transparent image in which the
    # selected object(s) are opaque and the rest is at various
    # levels of transparency
    transparent = segmented.to_transparent()

    # this will return a masked image suitable for use in inpainting:
    mask = segmented.to_mask(threshold=0.5)

The threshold used in the call to to_mask() selects pixels for use in
the mask that exceed the indicated confidence threshold. Values range
from 0.0 to 1.0. The higher the threshold, the more confident the
algorithm is. In limited testing, I have found that values around 0.5
work fine.
'''

import torch
import numpy as  np
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image, ImageOps
from torchvision import transforms
from ldm.invoke.globals import global_cache_dir

CLIPSEG_MODEL = 'CIDAS/clipseg-rd64-refined'
CLIPSEG_SIZE = 352

class SegmentedGrayscale(object):
    def __init__(self, image:Image, heatmap:torch.Tensor):
        self.heatmap = heatmap
        self.image = image

    def to_grayscale(self,invert:bool=False)->Image:
        return self._rescale(Image.fromarray(np.uint8(255 - self.heatmap * 255 if invert else self.heatmap * 255)))

    def to_mask(self,threshold:float=0.5)->Image:
        discrete_heatmap = self.heatmap.lt(threshold).int()
        return self._rescale(Image.fromarray(np.uint8(discrete_heatmap*255),mode='L'))

    def to_transparent(self,invert:bool=False)->Image:
        transparent_image = self.image.copy()
        # For img2img, we want the selected regions to be transparent,
        # but to_grayscale() returns the opposite. Thus invert.
        gs = self.to_grayscale(not invert)
        transparent_image.putalpha(gs)
        return transparent_image

    # unscales and uncrops the 352x352 heatmap so that it matches the image again
    def _rescale(self, heatmap:Image)->Image:
        size = self.image.width if (self.image.width > self.image.height) else self.image.height
        resized_image = heatmap.resize(
            (size,size),
            resample=Image.Resampling.LANCZOS
        )
        return resized_image.crop((0,0,self.image.width,self.image.height))

class Txt2Mask(object):
    '''
    Create new Txt2Mask object. The optional device argument can be one of
    'cuda', 'mps' or 'cpu'.
    '''
    def __init__(self,device='cpu',refined=False):
        print('>> Initializing clipseg model for text to mask inference')

        # BUG: we are not doing anything with the device option at this time
        self.device = device
        self.processor = AutoProcessor.from_pretrained(CLIPSEG_MODEL,
                                                       cache_dir=global_cache_dir('hub')
                                                       )
        self.model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL,
                                                                 cache_dir=global_cache_dir('hub')
                                                                 )

    @torch.no_grad()
    def segment(self, image, prompt:str) -> SegmentedGrayscale:
        '''
        Given a prompt string such as "a bagel", tries to identify the object in the
        provided image and returns a SegmentedGrayscale object in which the brighter
        pixels indicate where the object is inferred to be.
        '''
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((CLIPSEG_SIZE, CLIPSEG_SIZE)), # must be multiple of 64...
        ])

        if type(image) is str:
            image = Image.open(image).convert('RGB')

        image = ImageOps.exif_transpose(image)
        img = self._scale_and_crop(image)

        inputs = self.processor(text=[prompt],
                                images=[img],
                                padding=True,
                                return_tensors='pt')
        outputs = self.model(**inputs)
        heatmap = torch.sigmoid(outputs.logits)
        return SegmentedGrayscale(image, heatmap)

    def _scale_and_crop(self, image:Image)->Image:
        scaled_image = Image.new('RGB',(CLIPSEG_SIZE,CLIPSEG_SIZE))
        if image.width > image.height: # width is constraint
            scale = CLIPSEG_SIZE / image.width
        else:
            scale = CLIPSEG_SIZE / image.height
        scaled_image.paste(
            image.resize(
                (int(scale * image.width),
                 int(scale * image.height)
                ),
                resample=Image.Resampling.LANCZOS
            ),box=(0,0)
        )
        return scaled_image
