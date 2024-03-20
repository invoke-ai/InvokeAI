"""Makes available the Txt2Mask class, which assists in the automatic
assignment of masks via text prompt using clipseg.

Here is typical usage:

    from invokeai.backend.image_util.txt2mask import Txt2Mask, SegmentedGrayscale
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
"""

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, CLIPSegForImageSegmentation

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config

CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined"
CLIPSEG_SIZE = 352


class SegmentedGrayscale(object):
    def __init__(self, image: Image.Image, heatmap: torch.Tensor):
        self.heatmap = heatmap
        self.image = image

    def to_grayscale(self, invert: bool = False) -> Image.Image:
        return self._rescale(Image.fromarray(np.uint8(255 - self.heatmap * 255 if invert else self.heatmap * 255)))

    def to_mask(self, threshold: float = 0.5) -> Image.Image:
        discrete_heatmap = self.heatmap.lt(threshold).int()
        return self._rescale(Image.fromarray(np.uint8(discrete_heatmap * 255), mode="L"))

    def to_transparent(self, invert: bool = False) -> Image.Image:
        transparent_image = self.image.copy()
        # For img2img, we want the selected regions to be transparent,
        # but to_grayscale() returns the opposite. Thus invert.
        gs = self.to_grayscale(not invert)
        transparent_image.putalpha(gs)
        return transparent_image

    # unscales and uncrops the 352x352 heatmap so that it matches the image again
    def _rescale(self, heatmap: Image.Image) -> Image.Image:
        size = self.image.width if (self.image.width > self.image.height) else self.image.height
        resized_image = heatmap.resize((size, size), resample=Image.Resampling.LANCZOS)
        return resized_image.crop((0, 0, self.image.width, self.image.height))


class Txt2Mask(object):
    """
    Create new Txt2Mask object. The optional device argument can be one of
    'cuda', 'mps' or 'cpu'.
    """

    def __init__(self, device="cpu", refined=False):
        logger.info("Initializing clipseg model for text to mask inference")

        # BUG: we are not doing anything with the device option at this time
        self.device = device
        self.processor = AutoProcessor.from_pretrained(CLIPSEG_MODEL, cache_dir=get_config().cache_dir)
        self.model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL, cache_dir=get_config().cache_dir)

    @torch.no_grad()
    def segment(self, image: Image.Image, prompt: str) -> SegmentedGrayscale:
        """
        Given a prompt string such as "a bagel", tries to identify the object in the
        provided image and returns a SegmentedGrayscale object in which the brighter
        pixels indicate where the object is inferred to be.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        image = ImageOps.exif_transpose(image)
        img = self._scale_and_crop(image)

        inputs = self.processor(text=[prompt], images=[img], padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        heatmap = torch.sigmoid(outputs.logits)
        return SegmentedGrayscale(image, heatmap)

    def _scale_and_crop(self, image: Image.Image) -> Image.Image:
        scaled_image = Image.new("RGB", (CLIPSEG_SIZE, CLIPSEG_SIZE))
        if image.width > image.height:  # width is constraint
            scale = CLIPSEG_SIZE / image.width
        else:
            scale = CLIPSEG_SIZE / image.height
        scaled_image.paste(
            image.resize(
                (int(scale * image.width), int(scale * image.height)),
                resample=Image.Resampling.LANCZOS,
            ),
            box=(0, 0),
        )
        return scaled_image
