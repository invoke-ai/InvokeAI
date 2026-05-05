import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from PIL.Image import Image as ImageType

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, invocation_output
from invokeai.app.invocations.fields import (
    BoundingBoxField,
    ImageField,
    InputField,
    OutputField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.primitives import BoundingBoxCollectionOutput, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.invocation_context import InvocationContext

DetailerFaceSelectionMode = Literal["highest_score", "largest_area", "index"]


def _bbox_area(bbox: BoundingBoxField) -> int:
    return max(0, bbox.x_max - bbox.x_min) * max(0, bbox.y_max - bbox.y_min)


def select_bounding_box(
    collection: list[BoundingBoxField], selection_mode: DetailerFaceSelectionMode, index: int = 0
) -> list[BoundingBoxField]:
    if len(collection) == 0:
        return []

    if selection_mode == "highest_score":
        return [max(collection, key=lambda bbox: bbox.score if bbox.score is not None else -1)]

    if selection_mode == "largest_area":
        return [max(collection, key=_bbox_area)]

    sorted_collection = sorted(collection, key=lambda bbox: (bbox.x_min, bbox.y_min))
    if index < 0 or index >= len(sorted_collection):
        return []

    return [sorted_collection[index]]


@invocation(
    "select_bounding_box",
    title="Select Bounding Box",
    tags=["bounding box", "detailer"],
    category="segmentation",
    version="1.0.0",
)
class SelectBoundingBoxInvocation(BaseInvocation):
    """Select one bounding box from a collection."""

    collection: list[BoundingBoxField] = InputField(default=[], description="The bounding boxes to select from.")
    selection_mode: DetailerFaceSelectionMode = InputField(
        default="highest_score",
        description="How to select the bounding box.",
    )
    index: int = InputField(default=0, ge=0, description="The index to select when selection mode is index.")

    def invoke(self, context: InvocationContext) -> BoundingBoxCollectionOutput:
        return BoundingBoxCollectionOutput(
            collection=select_bounding_box(
                collection=self.collection,
                selection_mode=self.selection_mode,
                index=self.index,
            )
        )


@dataclass(frozen=True)
class DetailerCropResult:
    image: ImageType
    denoise_mask: ImageType
    paste_alpha_mask: ImageType
    x: int
    y: int
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    detected: bool


@invocation_output("detailer_crop_output")
class DetailerCropOutput(ImageOutput):
    """Prepared crop and masks for a detailer denoise pass."""

    denoise_mask: ImageField = OutputField(description="Processing-sized denoise mask.")
    paste_alpha_mask: ImageField = OutputField(description="Original crop-sized alpha mask for paste-back.")
    x: int = OutputField(description="The x coordinate of the original crop's left side.")
    y: int = OutputField(description="The y coordinate of the original crop's top side.")
    original_width: int = OutputField(description="The original crop width.")
    original_height: int = OutputField(description="The original crop height.")
    processed_width: int = OutputField(description="The processing crop width.")
    processed_height: int = OutputField(description="The processing crop height.")
    detected: bool = OutputField(description="Whether a usable mask was detected.")


def _round_up_to_multiple(value: int | float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _round_down_to_multiple(value: int | float, multiple: int) -> int:
    return int(math.floor(value / multiple) * multiple)


def _round_to_multiple(value: int | float, multiple: int) -> int:
    return int(round(value / multiple) * multiple)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _get_mask_bbox(mask: ImageType) -> tuple[int, int, int, int] | None:
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    y_coords, x_coords = np.where(mask_np > 0)

    if len(x_coords) == 0 or len(y_coords) == 0:
        return None

    return (
        int(x_coords.min()),
        int(y_coords.min()),
        int(x_coords.max()) + 1,
        int(y_coords.max()) + 1,
    )


def _make_noop_crop(image: ImageType) -> DetailerCropResult:
    crop_size = min(64, image.width, image.height)
    crop_size = max(8, _round_down_to_multiple(crop_size, 8))
    crop = image.convert("RGBA").crop((0, 0, crop_size, crop_size))
    denoise_mask = Image.new("L", (crop_size, crop_size), color=255)
    paste_alpha_mask = Image.new("L", (crop_size, crop_size), color=0)

    return DetailerCropResult(
        image=crop,
        denoise_mask=denoise_mask,
        paste_alpha_mask=paste_alpha_mask,
        x=0,
        y=0,
        original_width=crop_size,
        original_height=crop_size,
        processed_width=crop_size,
        processed_height=crop_size,
        detected=False,
    )


def _get_square_crop_box(
    mask_bbox: tuple[int, int, int, int], image_size: tuple[int, int], padding: int
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x_min, y_min, x_max, y_max = mask_bbox
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    max_crop_size = max(8, _round_down_to_multiple(min(image_width, image_height), 8))
    crop_size = max(8, _round_up_to_multiple(max(bbox_width, bbox_height) + padding * 2, 8))
    crop_size = min(crop_size, max_crop_size)

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    x = _clamp(int(round(center_x - crop_size / 2)), 0, max(0, image_width - crop_size))
    y = _clamp(int(round(center_y - crop_size / 2)), 0, max(0, image_height - crop_size))

    return (x, y, x + crop_size, y + crop_size)


def _get_process_size(
    original_size: int,
    target_size: int,
    max_upscale: float,
    max_process_size: int,
) -> int:
    target_size = max(64, _round_to_multiple(target_size, 8))
    max_process_size = max(64, _round_down_to_multiple(max_process_size, 8))

    if original_size < target_size:
        desired = min(float(target_size), original_size * max_upscale)
    else:
        desired = float(original_size)

    desired = min(desired, max_process_size)
    return max(64, min(max_process_size, _round_to_multiple(desired, 8)))


def _prepare_binary_mask(mask: ImageType, mask_expand: int, mask_feather: int) -> ImageType:
    mask_np = np.where(np.array(mask.convert("L"), dtype=np.uint8) > 0, 255, 0).astype(np.uint8)

    if mask_expand > 0:
        kernel_size = mask_expand * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    expanded = Image.fromarray(mask_np)
    if mask_feather > 0:
        expanded = expanded.filter(ImageFilter.GaussianBlur(mask_feather))

    return expanded


def _prepare_denoise_mask(mask: ImageType, mask_expand: int) -> ImageType:
    # create_gradient_mask expects black where detail should be applied, white where preserved.
    # Keep this mask hard-edged. The denoise gradient is applied later by create_gradient_mask.edge_radius.
    return ImageOps.invert(_prepare_binary_mask(mask=mask, mask_expand=mask_expand, mask_feather=0))


def _prepare_paste_alpha_mask(mask: ImageType, mask_expand: int, mask_feather: int) -> ImageType:
    # The detailer paste invocation uses normal alpha semantics: white pastes, black preserves.
    return _prepare_binary_mask(mask=mask, mask_expand=mask_expand, mask_feather=mask_feather)


def prepare_detailer_crop(
    image: ImageType,
    mask: ImageType,
    padding: int = 64,
    mask_expand: int = 8,
    mask_feather: int = 24,
    denoise_mask_expand: int | None = None,
    denoise_mask_feather: int | None = None,
    paste_mask_expand: int = 0,
    paste_mask_feather: int = 8,
    target_size: int = 768,
    max_upscale: float = 8.0,
    max_process_size: int = 768,
) -> DetailerCropResult:
    image = image.convert("RGBA")
    denoise_mask_expand = mask_expand if denoise_mask_expand is None else denoise_mask_expand
    _ = denoise_mask_feather
    mask_bbox = _get_mask_bbox(mask)

    if mask_bbox is None:
        return _make_noop_crop(image)

    crop_box = _get_square_crop_box(mask_bbox=mask_bbox, image_size=image.size, padding=padding)
    x, y, x_max, y_max = crop_box
    crop = image.crop(crop_box)
    crop_mask = mask.convert("L").crop(crop_box)
    denoise_mask = _prepare_denoise_mask(
        mask=crop_mask,
        mask_expand=denoise_mask_expand,
    )
    paste_alpha_mask = _prepare_paste_alpha_mask(
        mask=crop_mask,
        mask_expand=paste_mask_expand,
        mask_feather=paste_mask_feather,
    )

    process_size = _get_process_size(
        original_size=crop.width,
        target_size=target_size,
        max_upscale=max_upscale,
        max_process_size=max_process_size,
    )
    processed_image = crop.resize((process_size, process_size), resample=Image.Resampling.LANCZOS)
    processed_denoise_mask = denoise_mask.resize((process_size, process_size), resample=Image.Resampling.NEAREST)

    return DetailerCropResult(
        image=processed_image,
        denoise_mask=processed_denoise_mask,
        paste_alpha_mask=paste_alpha_mask,
        x=x,
        y=y,
        original_width=x_max - x,
        original_height=y_max - y,
        processed_width=process_size,
        processed_height=process_size,
        detected=True,
    )


@invocation(
    "detailer_crop_from_mask",
    title="Detailer Crop From Mask",
    tags=["image", "mask", "detailer"],
    category="image",
    version="1.0.0",
)
class DetailerCropFromMaskInvocation(BaseInvocation):
    """Prepare a padded, optionally upscaled crop and masks for a detailer pass."""

    image: ImageField = InputField(description="The original image.")
    mask: ImageField = InputField(description="The full-size object mask.")
    padding: int = InputField(default=64, ge=0, description="Padding around the detected mask bbox.")
    mask_expand: int = InputField(default=8, ge=0, description="Pixels to expand the mask before feathering.")
    mask_feather: int = InputField(default=24, ge=0, description="Gaussian feather radius for the mask.")
    denoise_mask_expand: int | None = InputField(
        default=None,
        ge=0,
        description="Pixels to expand the denoise mask before feathering. Defaults to mask_expand.",
    )
    denoise_mask_feather: int | None = InputField(
        default=None,
        ge=0,
        description="Gaussian feather radius for the denoise mask. Defaults to mask_feather.",
    )
    paste_mask_expand: int = InputField(
        default=0,
        ge=0,
        description="Pixels to expand the paste alpha mask before feathering.",
    )
    paste_mask_feather: int = InputField(
        default=8,
        ge=0,
        description="Gaussian feather radius for the paste alpha mask.",
    )
    target_size: int = InputField(default=768, ge=64, description="Preferred processing size for small crops.")
    max_upscale: float = InputField(default=8.0, ge=1.0, description="Maximum crop upscale factor.")
    max_process_size: int = InputField(default=768, ge=64, description="Maximum processing size.")

    def invoke(self, context: InvocationContext) -> DetailerCropOutput:
        image = context.images.get_pil(self.image.image_name, mode="RGBA")
        mask = context.images.get_pil(self.mask.image_name, mode="L")
        result = prepare_detailer_crop(
            image=image,
            mask=mask,
            padding=self.padding,
            mask_expand=self.mask_expand,
            mask_feather=self.mask_feather,
            denoise_mask_expand=self.denoise_mask_expand,
            denoise_mask_feather=self.denoise_mask_feather,
            paste_mask_expand=self.paste_mask_expand,
            paste_mask_feather=self.paste_mask_feather,
            target_size=self.target_size,
            max_upscale=self.max_upscale,
            max_process_size=self.max_process_size,
        )

        image_dto = context.images.save(image=result.image)
        denoise_mask_dto = context.images.save(image=result.denoise_mask, image_category=ImageCategory.MASK)
        paste_alpha_mask_dto = context.images.save(image=result.paste_alpha_mask, image_category=ImageCategory.MASK)

        return DetailerCropOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=result.processed_width,
            height=result.processed_height,
            denoise_mask=ImageField(image_name=denoise_mask_dto.image_name),
            paste_alpha_mask=ImageField(image_name=paste_alpha_mask_dto.image_name),
            x=result.x,
            y=result.y,
            original_width=result.original_width,
            original_height=result.original_height,
            processed_width=result.processed_width,
            processed_height=result.processed_height,
            detected=result.detected,
        )


def paste_detailer_crop(
    base_image: ImageType,
    image: ImageType,
    paste_alpha_mask: ImageType,
    x: int,
    y: int,
    original_width: int,
    original_height: int,
) -> ImageType:
    base = base_image.convert("RGBA")
    crop = image.convert("RGBA").resize((original_width, original_height), resample=Image.Resampling.LANCZOS)
    alpha = paste_alpha_mask.convert("L").resize((original_width, original_height), resample=Image.Resampling.BILINEAR)

    if alpha.getbbox() is None:
        return base

    result = base.copy()
    base_crop = result.crop((x, y, x + original_width, y + original_height))
    blended_crop = Image.composite(crop, base_crop, alpha)
    result.paste(blended_crop, (x, y))
    return result


@invocation(
    "detailer_paste_crop",
    title="Detailer Paste Crop",
    tags=["image", "mask", "detailer"],
    category="image",
    version="1.0.0",
)
class DetailerPasteCropInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Paste a detailed crop back onto the base image using normal alpha mask semantics."""

    base_image: ImageField = InputField(description="The original base image.")
    image: ImageField = InputField(description="The detailed crop image.")
    paste_alpha_mask: ImageField = InputField(description="White pastes the detailed crop, black preserves the base.")
    x: int = InputField(default=0, ge=0, description="The x coordinate of the original crop's left side.")
    y: int = InputField(default=0, ge=0, description="The y coordinate of the original crop's top side.")
    original_width: int = InputField(default=64, ge=1, description="The original crop width.")
    original_height: int = InputField(default=64, ge=1, description="The original crop height.")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        base_image = context.images.get_pil(self.base_image.image_name, mode="RGBA")
        image = context.images.get_pil(self.image.image_name, mode="RGBA")
        paste_alpha_mask = context.images.get_pil(self.paste_alpha_mask.image_name, mode="L")
        result = paste_detailer_crop(
            base_image=base_image,
            image=image,
            paste_alpha_mask=paste_alpha_mask,
            x=self.x,
            y=self.y,
            original_width=self.original_width,
            original_height=self.original_height,
        )
        image_dto = context.images.save(image=result)

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=result.width,
            height=result.height,
        )
