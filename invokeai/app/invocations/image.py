# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy
from PIL import Image, ImageChops, ImageFilter, ImageOps

from invokeai.app.invocations.constants import IMAGE_MODES
from invokeai.app.invocations.fields import (
    ColorField,
    FieldDescriptions,
    ImageField,
    InputField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.image_util.invisible_watermark import InvisibleWatermark
from invokeai.backend.image_util.safety_checker import SafetyChecker

from .baseinvocation import BaseInvocation, Classification, invocation


@invocation("show_image", title="Show Image", tags=["image"], category="image", version="1.0.1")
class ShowImageInvocation(BaseInvocation):
    """Displays a provided image using the OS image viewer, and passes it forward in the pipeline."""

    image: ImageField = InputField(description="The image to show")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        image.show()

        # TODO: how to handle failure?

        return ImageOutput(
            image=ImageField(image_name=self.image.image_name),
            width=image.width,
            height=image.height,
        )


@invocation(
    "blank_image",
    title="Blank Image",
    tags=["image"],
    category="image",
    version="1.2.1",
)
class BlankImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Creates a blank image and forwards it to the pipeline"""

    width: int = InputField(default=512, description="The width of the image")
    height: int = InputField(default=512, description="The height of the image")
    mode: Literal["RGB", "RGBA"] = InputField(default="RGB", description="The mode of the image")
    color: ColorField = InputField(default=ColorField(r=0, g=0, b=0, a=255), description="The color of the image")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = Image.new(mode=self.mode, size=(self.width, self.height), color=self.color.tuple())

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)


@invocation(
    "img_crop",
    title="Crop Image",
    tags=["image", "crop"],
    category="image",
    version="1.2.1",
)
class ImageCropInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Crops an image to a specified box. The box can be outside of the image."""

    image: ImageField = InputField(description="The image to crop")
    x: int = InputField(default=0, description="The left x coordinate of the crop rectangle")
    y: int = InputField(default=0, description="The top y coordinate of the crop rectangle")
    width: int = InputField(default=512, gt=0, description="The width of the crop rectangle")
    height: int = InputField(default=512, gt=0, description="The height of the crop rectangle")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        image_crop = Image.new(mode="RGBA", size=(self.width, self.height), color=(0, 0, 0, 0))
        image_crop.paste(image, (-self.x, -self.y))

        image_dto = context.images.save(image=image_crop)

        return ImageOutput.build(image_dto)


@invocation(
    invocation_type="img_pad_crop",
    title="Center Pad or Crop Image",
    category="image",
    tags=["image", "pad", "crop"],
    version="1.0.0",
)
class CenterPadCropInvocation(BaseInvocation):
    """Pad or crop an image's sides from the center by specified pixels. Positive values are outside of the image."""

    image: ImageField = InputField(description="The image to crop")
    left: int = InputField(
        default=0,
        description="Number of pixels to pad/crop from the left (negative values crop inwards, positive values pad outwards)",
    )
    right: int = InputField(
        default=0,
        description="Number of pixels to pad/crop from the right (negative values crop inwards, positive values pad outwards)",
    )
    top: int = InputField(
        default=0,
        description="Number of pixels to pad/crop from the top (negative values crop inwards, positive values pad outwards)",
    )
    bottom: int = InputField(
        default=0,
        description="Number of pixels to pad/crop from the bottom (negative values crop inwards, positive values pad outwards)",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        # Calculate and create new image dimensions
        new_width = image.width + self.right + self.left
        new_height = image.height + self.top + self.bottom
        image_crop = Image.new(mode="RGBA", size=(new_width, new_height), color=(0, 0, 0, 0))

        # Paste new image onto input
        image_crop.paste(image, (self.left, self.top))

        image_dto = context.images.save(image=image_crop)

        return ImageOutput.build(image_dto)


@invocation(
    "img_paste",
    title="Paste Image",
    tags=["image", "paste"],
    category="image",
    version="1.2.1",
)
class ImagePasteInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Pastes an image into another image."""

    base_image: ImageField = InputField(description="The base image")
    image: ImageField = InputField(description="The image to paste")
    mask: Optional[ImageField] = InputField(
        default=None,
        description="The mask to use when pasting",
    )
    x: int = InputField(default=0, description="The left x coordinate at which to paste the image")
    y: int = InputField(default=0, description="The top y coordinate at which to paste the image")
    crop: bool = InputField(default=False, description="Crop to base image dimensions")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        base_image = context.images.get_pil(self.base_image.image_name)
        image = context.images.get_pil(self.image.image_name)
        mask = None
        if self.mask is not None:
            mask = context.images.get_pil(self.mask.image_name)
            mask = ImageOps.invert(mask.convert("L"))
        # TODO: probably shouldn't invert mask here... should user be required to do it?

        min_x = min(0, self.x)
        min_y = min(0, self.y)
        max_x = max(base_image.width, image.width + self.x)
        max_y = max(base_image.height, image.height + self.y)

        new_image = Image.new(mode="RGBA", size=(max_x - min_x, max_y - min_y), color=(0, 0, 0, 0))
        new_image.paste(base_image, (abs(min_x), abs(min_y)))
        new_image.paste(image, (max(0, self.x), max(0, self.y)), mask=mask)

        if self.crop:
            base_w, base_h = base_image.size
            new_image = new_image.crop((abs(min_x), abs(min_y), abs(min_x) + base_w, abs(min_y) + base_h))

        image_dto = context.images.save(image=new_image)

        return ImageOutput.build(image_dto)


@invocation(
    "tomask",
    title="Mask from Alpha",
    tags=["image", "mask"],
    category="image",
    version="1.2.1",
)
class MaskFromAlphaInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Extracts the alpha channel of an image as a mask."""

    image: ImageField = InputField(description="The image to create the mask from")
    invert: bool = InputField(default=False, description="Whether or not to invert the mask")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        image_mask = image.split()[-1]
        if self.invert:
            image_mask = ImageOps.invert(image_mask)

        image_dto = context.images.save(image=image_mask, image_category=ImageCategory.MASK)

        return ImageOutput.build(image_dto)


@invocation(
    "img_mul",
    title="Multiply Images",
    tags=["image", "multiply"],
    category="image",
    version="1.2.1",
)
class ImageMultiplyInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Multiplies two images together using `PIL.ImageChops.multiply()`."""

    image1: ImageField = InputField(description="The first image to multiply")
    image2: ImageField = InputField(description="The second image to multiply")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image1 = context.images.get_pil(self.image1.image_name)
        image2 = context.images.get_pil(self.image2.image_name)

        multiply_image = ImageChops.multiply(image1, image2)

        image_dto = context.images.save(image=multiply_image)

        return ImageOutput.build(image_dto)


IMAGE_CHANNELS = Literal["A", "R", "G", "B"]


@invocation(
    "img_chan",
    title="Extract Image Channel",
    tags=["image", "channel"],
    category="image",
    version="1.2.1",
)
class ImageChannelInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Gets a channel from an image."""

    image: ImageField = InputField(description="The image to get the channel from")
    channel: IMAGE_CHANNELS = InputField(default="A", description="The channel to get")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        channel_image = image.getchannel(self.channel)

        image_dto = context.images.save(image=channel_image)

        return ImageOutput.build(image_dto)


@invocation(
    "img_conv",
    title="Convert Image Mode",
    tags=["image", "convert"],
    category="image",
    version="1.2.1",
)
class ImageConvertInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Converts an image to a different mode."""

    image: ImageField = InputField(description="The image to convert")
    mode: IMAGE_MODES = InputField(default="L", description="The mode to convert to")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        converted_image = image.convert(self.mode)

        image_dto = context.images.save(image=converted_image)

        return ImageOutput.build(image_dto)


@invocation(
    "img_blur",
    title="Blur Image",
    tags=["image", "blur"],
    category="image",
    version="1.2.1",
)
class ImageBlurInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Blurs an image"""

    image: ImageField = InputField(description="The image to blur")
    radius: float = InputField(default=8.0, ge=0, description="The blur radius")
    # Metadata
    blur_type: Literal["gaussian", "box"] = InputField(default="gaussian", description="The type of blur")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        blur = (
            ImageFilter.GaussianBlur(self.radius) if self.blur_type == "gaussian" else ImageFilter.BoxBlur(self.radius)
        )
        blur_image = image.filter(blur)

        image_dto = context.images.save(image=blur_image)

        return ImageOutput.build(image_dto)


@invocation(
    "unsharp_mask",
    title="Unsharp Mask",
    tags=["image", "unsharp_mask"],
    category="image",
    version="1.2.1",
    classification=Classification.Beta,
)
class UnsharpMaskInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Applies an unsharp mask filter to an image"""

    image: ImageField = InputField(description="The image to use")
    radius: float = InputField(gt=0, description="Unsharp mask radius", default=2)
    strength: float = InputField(ge=0, description="Unsharp mask strength", default=50)

    def pil_from_array(self, arr):
        return Image.fromarray((arr * 255).astype("uint8"))

    def array_from_pil(self, img):
        return numpy.array(img) / 255

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        mode = image.mode

        alpha_channel = image.getchannel("A") if mode == "RGBA" else None
        image = image.convert("RGB")
        image_blurred = self.array_from_pil(image.filter(ImageFilter.GaussianBlur(radius=self.radius)))

        image = self.array_from_pil(image)
        image += (image - image_blurred) * (self.strength / 100.0)
        image = numpy.clip(image, 0, 1)
        image = self.pil_from_array(image)

        image = image.convert(mode)

        # Make the image RGBA if we had a source alpha channel
        if alpha_channel is not None:
            image.putalpha(alpha_channel)

        image_dto = context.images.save(image=image)

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image.width,
            height=image.height,
        )


PIL_RESAMPLING_MODES = Literal[
    "nearest",
    "box",
    "bilinear",
    "hamming",
    "bicubic",
    "lanczos",
]


PIL_RESAMPLING_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@invocation(
    "img_resize",
    title="Resize Image",
    tags=["image", "resize"],
    category="image",
    version="1.2.1",
)
class ImageResizeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Resizes an image to specific dimensions"""

    image: ImageField = InputField(description="The image to resize")
    width: int = InputField(default=512, gt=0, description="The width to resize to (px)")
    height: int = InputField(default=512, gt=0, description="The height to resize to (px)")
    resample_mode: PIL_RESAMPLING_MODES = InputField(default="bicubic", description="The resampling mode")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        resample_mode = PIL_RESAMPLING_MAP[self.resample_mode]

        resize_image = image.resize(
            (self.width, self.height),
            resample=resample_mode,
        )

        image_dto = context.images.save(image=resize_image)

        return ImageOutput.build(image_dto)


@invocation(
    "img_scale",
    title="Scale Image",
    tags=["image", "scale"],
    category="image",
    version="1.2.1",
)
class ImageScaleInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Scales an image by a factor"""

    image: ImageField = InputField(description="The image to scale")
    scale_factor: float = InputField(
        default=2.0,
        gt=0,
        description="The factor by which to scale the image",
    )
    resample_mode: PIL_RESAMPLING_MODES = InputField(default="bicubic", description="The resampling mode")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        resample_mode = PIL_RESAMPLING_MAP[self.resample_mode]
        width = int(image.width * self.scale_factor)
        height = int(image.height * self.scale_factor)

        resize_image = image.resize(
            (width, height),
            resample=resample_mode,
        )

        image_dto = context.images.save(image=resize_image)

        return ImageOutput.build(image_dto)


@invocation(
    "img_lerp",
    title="Lerp Image",
    tags=["image", "lerp"],
    category="image",
    version="1.2.1",
)
class ImageLerpInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Linear interpolation of all pixels of an image"""

    image: ImageField = InputField(description="The image to lerp")
    min: int = InputField(default=0, ge=0, le=255, description="The minimum output value")
    max: int = InputField(default=255, ge=0, le=255, description="The maximum output value")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        image_arr = numpy.asarray(image, dtype=numpy.float32) / 255
        image_arr = image_arr * (self.max - self.min) + self.min

        lerp_image = Image.fromarray(numpy.uint8(image_arr))

        image_dto = context.images.save(image=lerp_image)

        return ImageOutput.build(image_dto)


@invocation(
    "img_ilerp",
    title="Inverse Lerp Image",
    tags=["image", "ilerp"],
    category="image",
    version="1.2.1",
)
class ImageInverseLerpInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Inverse linear interpolation of all pixels of an image"""

    image: ImageField = InputField(description="The image to lerp")
    min: int = InputField(default=0, ge=0, le=255, description="The minimum input value")
    max: int = InputField(default=255, ge=0, le=255, description="The maximum input value")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        image_arr = numpy.asarray(image, dtype=numpy.float32)
        image_arr = numpy.minimum(numpy.maximum(image_arr - self.min, 0) / float(self.max - self.min), 1) * 255  # type: ignore [assignment]

        ilerp_image = Image.fromarray(numpy.uint8(image_arr))

        image_dto = context.images.save(image=ilerp_image)

        return ImageOutput.build(image_dto)


@invocation(
    "img_nsfw",
    title="Blur NSFW Image",
    tags=["image", "nsfw"],
    category="image",
    version="1.2.1",
)
class ImageNSFWBlurInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Add blur to NSFW-flagged images"""

    image: ImageField = InputField(description="The image to check")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        logger = context.logger
        logger.debug("Running NSFW checker")
        if SafetyChecker.has_nsfw_concept(image):
            logger.info("A potentially NSFW image has been detected. Image will be blurred.")
            blurry_image = image.filter(filter=ImageFilter.GaussianBlur(radius=32))
            caution = self._get_caution_img()
            blurry_image.paste(caution, (0, 0), caution)
            image = blurry_image

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)

    def _get_caution_img(self) -> Image.Image:
        import invokeai.app.assets.images as image_assets

        caution = Image.open(Path(image_assets.__path__[0]) / "caution.png")
        return caution.resize((caution.width // 2, caution.height // 2))


@invocation(
    "img_watermark",
    title="Add Invisible Watermark",
    tags=["image", "watermark"],
    category="image",
    version="1.2.1",
)
class ImageWatermarkInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Add an invisible watermark to an image"""

    image: ImageField = InputField(description="The image to check")
    text: str = InputField(default="InvokeAI", description="Watermark text")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        new_image = InvisibleWatermark.add_watermark(image, self.text)
        image_dto = context.images.save(image=new_image)

        return ImageOutput.build(image_dto)


@invocation(
    "mask_edge",
    title="Mask Edge",
    tags=["image", "mask", "inpaint"],
    category="image",
    version="1.2.1",
)
class MaskEdgeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Applies an edge mask to an image"""

    image: ImageField = InputField(description="The image to apply the mask to")
    edge_size: int = InputField(description="The size of the edge")
    edge_blur: int = InputField(description="The amount of blur on the edge")
    low_threshold: int = InputField(description="First threshold for the hysteresis procedure in Canny edge detection")
    high_threshold: int = InputField(
        description="Second threshold for the hysteresis procedure in Canny edge detection"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        mask = context.images.get_pil(self.image.image_name).convert("L")

        npimg = numpy.asarray(mask, dtype=numpy.uint8)
        npgradient = numpy.uint8(255 * (1.0 - numpy.floor(numpy.abs(0.5 - numpy.float32(npimg) / 255.0) * 2.0)))
        npedge = cv2.Canny(npimg, threshold1=self.low_threshold, threshold2=self.high_threshold)
        npmask = npgradient + npedge
        npmask = cv2.dilate(npmask, numpy.ones((3, 3), numpy.uint8), iterations=int(self.edge_size / 2))

        new_mask = Image.fromarray(npmask)

        if self.edge_blur > 0:
            new_mask = new_mask.filter(ImageFilter.BoxBlur(self.edge_blur))

        new_mask = ImageOps.invert(new_mask)

        image_dto = context.images.save(image=new_mask, image_category=ImageCategory.MASK)

        return ImageOutput.build(image_dto)


@invocation(
    "mask_combine",
    title="Combine Masks",
    tags=["image", "mask", "multiply"],
    category="image",
    version="1.2.1",
)
class MaskCombineInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Combine two masks together by multiplying them using `PIL.ImageChops.multiply()`."""

    mask1: ImageField = InputField(description="The first mask to combine")
    mask2: ImageField = InputField(description="The second image to combine")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        mask1 = context.images.get_pil(self.mask1.image_name).convert("L")
        mask2 = context.images.get_pil(self.mask2.image_name).convert("L")

        combined_mask = ImageChops.multiply(mask1, mask2)

        image_dto = context.images.save(image=combined_mask, image_category=ImageCategory.MASK)

        return ImageOutput.build(image_dto)


@invocation(
    "color_correct",
    title="Color Correct",
    tags=["image", "color"],
    category="image",
    version="1.2.1",
)
class ColorCorrectInvocation(BaseInvocation, WithMetadata, WithBoard):
    """
    Shifts the colors of a target image to match the reference image, optionally
    using a mask to only color-correct certain regions of the target image.
    """

    image: ImageField = InputField(description="The image to color-correct")
    reference: ImageField = InputField(description="Reference image for color-correction")
    mask: Optional[ImageField] = InputField(default=None, description="Mask to use when applying color-correction")
    mask_blur_radius: float = InputField(default=8, description="Mask blur radius")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        pil_init_mask = None
        if self.mask is not None:
            pil_init_mask = context.images.get_pil(self.mask.image_name).convert("L")

        init_image = context.images.get_pil(self.reference.image_name)

        result = context.images.get_pil(self.image.image_name).convert("RGBA")

        # if init_image is None or init_mask is None:
        #    return result

        # Get the original alpha channel of the mask if there is one.
        # Otherwise it is some other black/white image format ('1', 'L' or 'RGB')
        # pil_init_mask = (
        #    init_mask.getchannel("A")
        #    if init_mask.mode == "RGBA"
        #    else init_mask.convert("L")
        # )
        pil_init_image = init_image.convert("RGBA")  # Add an alpha channel if one doesn't exist

        # Build an image with only visible pixels from source to use as reference for color-matching.
        init_rgb_pixels = numpy.asarray(init_image.convert("RGB"), dtype=numpy.uint8)
        init_a_pixels = numpy.asarray(pil_init_image.getchannel("A"), dtype=numpy.uint8)
        init_mask_pixels = numpy.asarray(pil_init_mask, dtype=numpy.uint8)

        # Get numpy version of result
        np_image = numpy.asarray(result.convert("RGB"), dtype=numpy.uint8)

        # Mask and calculate mean and standard deviation
        mask_pixels = init_a_pixels * init_mask_pixels > 0
        np_init_rgb_pixels_masked = init_rgb_pixels[mask_pixels, :]
        np_image_masked = np_image[mask_pixels, :]

        if np_init_rgb_pixels_masked.size > 0:
            init_means = np_init_rgb_pixels_masked.mean(axis=0)
            init_std = np_init_rgb_pixels_masked.std(axis=0)
            gen_means = np_image_masked.mean(axis=0)
            gen_std = np_image_masked.std(axis=0)

            # Color correct
            np_matched_result = np_image.copy()
            np_matched_result[:, :, :] = (
                (
                    (
                        (np_matched_result[:, :, :].astype(numpy.float32) - gen_means[None, None, :])
                        / gen_std[None, None, :]
                    )
                    * init_std[None, None, :]
                    + init_means[None, None, :]
                )
                .clip(0, 255)
                .astype(numpy.uint8)
            )
            matched_result = Image.fromarray(np_matched_result, mode="RGB")
        else:
            matched_result = Image.fromarray(np_image, mode="RGB")

        # Blur the mask out (into init image) by specified amount
        if self.mask_blur_radius > 0:
            nm = numpy.asarray(pil_init_mask, dtype=numpy.uint8)
            inverted_nm = 255 - nm
            dilation_size = int(round(self.mask_blur_radius) + 20)
            dilating_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
            inverted_dilated_nm = cv2.dilate(inverted_nm, dilating_kernel)
            dilated_nm = 255 - inverted_dilated_nm
            nmd = cv2.erode(
                dilated_nm,
                kernel=numpy.ones((3, 3), dtype=numpy.uint8),
                iterations=int(self.mask_blur_radius / 2),
            )
            pmd = Image.fromarray(nmd, mode="L")
            blurred_init_mask = pmd.filter(ImageFilter.BoxBlur(self.mask_blur_radius))
        else:
            blurred_init_mask = pil_init_mask

        multiplied_blurred_init_mask = ImageChops.multiply(blurred_init_mask, result.split()[-1])

        # Paste original on color-corrected generation (using blurred mask)
        matched_result.paste(init_image, (0, 0), mask=multiplied_blurred_init_mask)

        image_dto = context.images.save(image=matched_result)

        return ImageOutput.build(image_dto)


@invocation(
    "img_hue_adjust",
    title="Adjust Image Hue",
    tags=["image", "hue"],
    category="image",
    version="1.2.1",
)
class ImageHueAdjustmentInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Adjusts the Hue of an image."""

    image: ImageField = InputField(description="The image to adjust")
    hue: int = InputField(default=0, description="The degrees by which to rotate the hue, 0-360")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        pil_image = context.images.get_pil(self.image.image_name)

        # Convert image to HSV color space
        hsv_image = numpy.array(pil_image.convert("HSV"))

        # Convert hue from 0..360 to 0..256
        hue = int(256 * ((self.hue % 360) / 360))

        # Increment each hue and wrap around at 255
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue) % 256

        # Convert back to PIL format and to original color mode
        pil_image = Image.fromarray(hsv_image, mode="HSV").convert("RGBA")

        image_dto = context.images.save(image=pil_image)

        return ImageOutput.build(image_dto)


COLOR_CHANNELS = Literal[
    "Red (RGBA)",
    "Green (RGBA)",
    "Blue (RGBA)",
    "Alpha (RGBA)",
    "Cyan (CMYK)",
    "Magenta (CMYK)",
    "Yellow (CMYK)",
    "Black (CMYK)",
    "Hue (HSV)",
    "Saturation (HSV)",
    "Value (HSV)",
    "Luminosity (LAB)",
    "A (LAB)",
    "B (LAB)",
    "Y (YCbCr)",
    "Cb (YCbCr)",
    "Cr (YCbCr)",
]

CHANNEL_FORMATS = {
    "Red (RGBA)": ("RGBA", 0),
    "Green (RGBA)": ("RGBA", 1),
    "Blue (RGBA)": ("RGBA", 2),
    "Alpha (RGBA)": ("RGBA", 3),
    "Cyan (CMYK)": ("CMYK", 0),
    "Magenta (CMYK)": ("CMYK", 1),
    "Yellow (CMYK)": ("CMYK", 2),
    "Black (CMYK)": ("CMYK", 3),
    "Hue (HSV)": ("HSV", 0),
    "Saturation (HSV)": ("HSV", 1),
    "Value (HSV)": ("HSV", 2),
    "Luminosity (LAB)": ("LAB", 0),
    "A (LAB)": ("LAB", 1),
    "B (LAB)": ("LAB", 2),
    "Y (YCbCr)": ("YCbCr", 0),
    "Cb (YCbCr)": ("YCbCr", 1),
    "Cr (YCbCr)": ("YCbCr", 2),
}


@invocation(
    "img_channel_offset",
    title="Offset Image Channel",
    tags=[
        "image",
        "offset",
        "red",
        "green",
        "blue",
        "alpha",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "hue",
        "saturation",
        "luminosity",
        "value",
    ],
    category="image",
    version="1.2.1",
)
class ImageChannelOffsetInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Add or subtract a value from a specific color channel of an image."""

    image: ImageField = InputField(description="The image to adjust")
    channel: COLOR_CHANNELS = InputField(description="Which channel to adjust")
    offset: int = InputField(default=0, ge=-255, le=255, description="The amount to adjust the channel by")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        pil_image = context.images.get_pil(self.image.image_name)

        # extract the channel and mode from the input and reference tuple
        mode = CHANNEL_FORMATS[self.channel][0]
        channel_number = CHANNEL_FORMATS[self.channel][1]

        # Convert PIL image to new format
        converted_image = numpy.array(pil_image.convert(mode)).astype(int)
        image_channel = converted_image[:, :, channel_number]

        # Adjust the value, clipping to 0..255
        image_channel = numpy.clip(image_channel + self.offset, 0, 255)

        # Put the channel back into the image
        converted_image[:, :, channel_number] = image_channel

        # Convert back to RGBA format and output
        pil_image = Image.fromarray(converted_image.astype(numpy.uint8), mode=mode).convert("RGBA")

        image_dto = context.images.save(image=pil_image)

        return ImageOutput.build(image_dto)


@invocation(
    "img_channel_multiply",
    title="Multiply Image Channel",
    tags=[
        "image",
        "invert",
        "scale",
        "multiply",
        "red",
        "green",
        "blue",
        "alpha",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "hue",
        "saturation",
        "luminosity",
        "value",
    ],
    category="image",
    version="1.2.1",
)
class ImageChannelMultiplyInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Scale a specific color channel of an image."""

    image: ImageField = InputField(description="The image to adjust")
    channel: COLOR_CHANNELS = InputField(description="Which channel to adjust")
    scale: float = InputField(default=1.0, ge=0.0, description="The amount to scale the channel by.")
    invert_channel: bool = InputField(default=False, description="Invert the channel after scaling")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        pil_image = context.images.get_pil(self.image.image_name)

        # extract the channel and mode from the input and reference tuple
        mode = CHANNEL_FORMATS[self.channel][0]
        channel_number = CHANNEL_FORMATS[self.channel][1]

        # Convert PIL image to new format
        converted_image = numpy.array(pil_image.convert(mode)).astype(float)
        image_channel = converted_image[:, :, channel_number]

        # Adjust the value, clipping to 0..255
        image_channel = numpy.clip(image_channel * self.scale, 0, 255)

        # Invert the channel if requested
        if self.invert_channel:
            image_channel = 255 - image_channel

        # Put the channel back into the image
        converted_image[:, :, channel_number] = image_channel

        # Convert back to RGBA format and output
        pil_image = Image.fromarray(converted_image.astype(numpy.uint8), mode=mode).convert("RGBA")

        image_dto = context.images.save(image=pil_image)

        return ImageOutput.build(image_dto)


@invocation(
    "save_image",
    title="Save Image",
    tags=["primitives", "image"],
    category="primitives",
    version="1.2.1",
    use_cache=False,
)
class SaveImageInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Saves an image. Unlike an image primitive, this invocation stores a copy of the image."""

    image: ImageField = InputField(description=FieldDescriptions.image)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)


@invocation(
    "iai_canvas_paste_back",
    title="InvokeAI Canvas Paste Back",
    tags=["image", "combine"],
    category="image",
    version="1.0.0",
)
class IAICanvasPasteBackInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Combines two images by using the mask provided"""

    source_image: ImageField = InputField(description="The source image")
    target_image: ImageField = InputField(default=None, description="The target image")
    mask: ImageField = InputField(
        description="The mask to use when pasting",
    )
    mask_blur: int = InputField(default=0, ge=0, description="The amount to blur the mask by")

    def _prepare_mask(self, mask: Image.Image):
        mask_array = numpy.array(mask)
        kernel = numpy.ones((self.mask_blur, self.mask_blur), numpy.uint8)
        dilated_mask_array = cv2.erode(mask_array, kernel, iterations=3)
        dilated_mask = Image.fromarray(dilated_mask_array)
        if self.mask_blur > 0:
            mask = dilated_mask.filter(ImageFilter.GaussianBlur(self.mask_blur))
        return ImageOps.invert(mask.convert("L"))

    def invoke(self, context: InvocationContext) -> ImageOutput:
        source_image = context.images.get_pil(self.source_image.image_name)
        target_image = context.images.get_pil(self.target_image.image_name)
        mask = self._prepare_mask(context.images.get_pil(self.mask.image_name))

        # Merge the bands back together
        source_image.paste(target_image, (0, 0), mask)

        image_dto = context.images.save(image=source_image)

        return ImageOutput.build(image_dto)
