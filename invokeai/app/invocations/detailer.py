import math
import re
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
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
LABEL_PRIORITY_SPLIT_PATTERN = re.compile(r"[|,.;]+")

DEBUG_COLLAGE_PANEL_WIDTH = 420
DEBUG_COLLAGE_PANEL_HEIGHT = 330
DEBUG_COLLAGE_PANEL_PADDING = 14
DEBUG_COLLAGE_HEADER_HEIGHT = 30
DEBUG_COLLAGE_LINE_HEIGHT = 16
DEBUG_COLLAGE_COLUMNS = 3
DEBUG_COLLAGE_ROWS = 2


def _bbox_area(bbox: BoundingBoxField) -> int:
    return max(0, bbox.x_max - bbox.x_min) * max(0, bbox.y_max - bbox.y_min)


def _normalize_bbox_label(label: str) -> str:
    return label.strip().strip("|,.;").strip().lower()


def _parse_label_priority(label_priority: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    for raw_label in LABEL_PRIORITY_SPLIT_PATTERN.split(label_priority):
        label = _normalize_bbox_label(raw_label)
        if not label or label in seen:
            continue
        labels.append(label)
        seen.add(label)

    return labels


def _filter_bounding_boxes_by_label_priority(
    collection: list[BoundingBoxField], label_priority: str
) -> list[BoundingBoxField]:
    priority_labels = _parse_label_priority(label_priority)
    if len(priority_labels) == 0 or not any(bbox.label for bbox in collection):
        return collection

    for priority_label in priority_labels:
        matches = [bbox for bbox in collection if bbox.label and _normalize_bbox_label(bbox.label) == priority_label]
        if matches:
            return matches

    return collection


def select_bounding_box(
    collection: list[BoundingBoxField],
    selection_mode: DetailerFaceSelectionMode,
    index: int = 0,
    label_priority: str = "",
) -> list[BoundingBoxField]:
    collection = _filter_bounding_boxes_by_label_priority(collection=collection, label_priority=label_priority)

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
    label_priority: str = InputField(
        default="",
        description="Optional detector label priority. If matching labeled boxes exist, the first label with matches is selected before applying the selection mode.",
    )

    def invoke(self, context: InvocationContext) -> BoundingBoxCollectionOutput:
        return BoundingBoxCollectionOutput(
            collection=select_bounding_box(
                collection=self.collection,
                selection_mode=self.selection_mode,
                index=self.index,
                label_priority=self.label_priority,
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
    prevent_downscale: bool = False,
) -> int:
    target_size = max(64, _round_to_multiple(target_size, 8))
    max_process_size = max(64, _round_down_to_multiple(max_process_size, 8))

    if original_size < target_size:
        desired = min(float(target_size), original_size * max_upscale)
    else:
        desired = float(original_size)

    if prevent_downscale:
        desired = max(desired, float(original_size))

    desired = min(desired, max_process_size)
    return max(64, min(max_process_size, _round_to_multiple(desired, 8)))


def _contract_binary_mask(mask_np: np.ndarray, mask_contract: int) -> np.ndarray:
    if mask_contract <= 0:
        return mask_np

    kernel_size = mask_contract * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    contracted_mask_np = cv2.erode(mask_np, kernel, iterations=1)
    if np.any(contracted_mask_np):
        return contracted_mask_np

    return mask_np


def _prepare_binary_mask(mask: ImageType, mask_expand: int, mask_feather: int, mask_contract: int = 0) -> ImageType:
    mask_np = np.where(np.array(mask.convert("L"), dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    mask_np = _contract_binary_mask(mask_np=mask_np, mask_contract=mask_contract)

    if mask_expand > 0:
        kernel_size = mask_expand * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)

    expanded = Image.fromarray(mask_np)
    if mask_feather > 0:
        expanded = expanded.filter(ImageFilter.GaussianBlur(mask_feather))

    return expanded


def _prepare_denoise_mask(mask: ImageType, mask_expand: int, mask_contract: int = 0) -> ImageType:
    # create_gradient_mask expects black where detail should be applied, white where preserved.
    # Keep this mask hard-edged. The denoise gradient is applied later by create_gradient_mask.edge_radius.
    return ImageOps.invert(
        _prepare_binary_mask(mask=mask, mask_expand=mask_expand, mask_feather=0, mask_contract=mask_contract)
    )


def _prepare_paste_alpha_mask(
    mask: ImageType, mask_expand: int, mask_feather: int, mask_contract: int = 0
) -> ImageType:
    # The detailer paste invocation uses normal alpha semantics: white pastes, black preserves.
    return _prepare_binary_mask(
        mask=mask, mask_expand=mask_expand, mask_feather=mask_feather, mask_contract=mask_contract
    )


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
    prevent_downscale: bool = False,
    denoise_mask_contract: int = 0,
    paste_mask_contract: int = 0,
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
        mask_contract=denoise_mask_contract,
    )
    paste_alpha_mask = _prepare_paste_alpha_mask(
        mask=crop_mask,
        mask_expand=paste_mask_expand,
        mask_feather=paste_mask_feather,
        mask_contract=paste_mask_contract,
    )

    process_size = _get_process_size(
        original_size=crop.width,
        target_size=target_size,
        max_upscale=max_upscale,
        max_process_size=max_process_size,
        prevent_downscale=prevent_downscale,
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
    prevent_downscale: bool = InputField(
        default=False,
        description="Avoid downscaling unless the crop exceeds max_process_size.",
    )
    denoise_mask_contract: int = InputField(
        default=0,
        ge=0,
        description="Pixels to contract the denoise mask before expansion.",
    )
    paste_mask_contract: int = InputField(
        default=0,
        ge=0,
        description="Pixels to contract the paste alpha mask before expansion.",
    )

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
            prevent_downscale=self.prevent_downscale,
            denoise_mask_contract=self.denoise_mask_contract,
            paste_mask_contract=self.paste_mask_contract,
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


def _fit_into(image: ImageType, max_width: int, max_height: int) -> ImageType:
    image = image.convert("RGBA")
    scale = min(max_width / image.width, max_height / image.height)
    width = max(1, int(round(image.width * scale)))
    height = max(1, int(round(image.height * scale)))
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)


def _draw_text_lines(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], fill: tuple[int, int, int, int]
) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill=fill)
        y += DEBUG_COLLAGE_LINE_HEIGHT


def _clamped_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    x_min = _clamp(x, 0, image_width)
    y_min = _clamp(y, 0, image_height)
    x_max = _clamp(x + width, 0, image_width)
    y_max = _clamp(y + height, 0, image_height)

    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)


def _draw_rectangle(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int] | None,
    outline: tuple[int, int, int, int],
    width: int,
) -> None:
    if rect is None:
        return
    for offset in range(width):
        draw.rectangle(
            (
                rect[0] + offset,
                rect[1] + offset,
                rect[2] - 1 - offset,
                rect[3] - 1 - offset,
            ),
            outline=outline,
        )


def _make_mask_visual(mask: ImageType, color: tuple[int, int, int]) -> ImageType:
    mask_l = mask.convert("L")
    image = Image.new("RGBA", mask_l.size, (*color, 255))
    image.putalpha(mask_l)
    backdrop = Image.new("RGBA", mask_l.size, (18, 20, 25, 255))
    return Image.alpha_composite(backdrop, image)


def _make_base_overlay(
    base_image: ImageType,
    mask: ImageType,
    selected_bounding_boxes: list[BoundingBoxField],
    x: int,
    y: int,
    original_width: int,
    original_height: int,
) -> ImageType:
    base = base_image.convert("RGBA")
    mask_l = mask.convert("L").resize(base.size, resample=Image.Resampling.NEAREST)
    overlay_alpha = mask_l.point(lambda value: min(120, value // 2))
    overlay = Image.new("RGBA", base.size, (56, 189, 248, 0))
    overlay.putalpha(overlay_alpha)
    result = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(result)
    stroke = max(2, round(min(base.size) / 256))

    for bbox in selected_bounding_boxes:
        bbox_rect = _clamped_rect(
            x=bbox.x_min,
            y=bbox.y_min,
            width=bbox.x_max - bbox.x_min,
            height=bbox.y_max - bbox.y_min,
            image_width=base.width,
            image_height=base.height,
        )
        _draw_rectangle(draw=draw, rect=bbox_rect, outline=(255, 214, 102, 255), width=stroke)

    crop_rect = _clamped_rect(
        x=x,
        y=y,
        width=original_width,
        height=original_height,
        image_width=base.width,
        image_height=base.height,
    )
    _draw_rectangle(draw=draw, rect=crop_rect, outline=(45, 212, 191, 255), width=stroke)

    return result


def _make_debug_panel(title: str, image: ImageType, lines: list[str]) -> ImageType:
    panel = Image.new("RGBA", (DEBUG_COLLAGE_PANEL_WIDTH, DEBUG_COLLAGE_PANEL_HEIGHT), (25, 28, 34, 255))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, DEBUG_COLLAGE_PANEL_WIDTH - 1, DEBUG_COLLAGE_PANEL_HEIGHT - 1), outline=(63, 70, 82, 255))
    draw.text((DEBUG_COLLAGE_PANEL_PADDING, 9), title, fill=(229, 231, 235, 255))

    footer_height = DEBUG_COLLAGE_LINE_HEIGHT * len(lines)
    max_image_width = DEBUG_COLLAGE_PANEL_WIDTH - DEBUG_COLLAGE_PANEL_PADDING * 2
    max_image_height = (
        DEBUG_COLLAGE_PANEL_HEIGHT - DEBUG_COLLAGE_HEADER_HEIGHT - footer_height - DEBUG_COLLAGE_PANEL_PADDING * 2
    )
    fitted = _fit_into(image, max_image_width, max(1, max_image_height))
    paste_x = (DEBUG_COLLAGE_PANEL_WIDTH - fitted.width) // 2
    paste_y = DEBUG_COLLAGE_HEADER_HEIGHT + max(0, (max_image_height - fitted.height) // 2)
    panel.alpha_composite(fitted, (paste_x, paste_y))

    if lines:
        _draw_text_lines(
            draw=draw,
            xy=(DEBUG_COLLAGE_PANEL_PADDING, DEBUG_COLLAGE_PANEL_HEIGHT - footer_height - DEBUG_COLLAGE_PANEL_PADDING),
            lines=lines,
            fill=(180, 187, 199, 255),
        )

    return panel


def _format_selected_bbox(selected_bbox: BoundingBoxField | None) -> str:
    if selected_bbox is None:
        return "bbox none"

    width = selected_bbox.x_max - selected_bbox.x_min
    height = selected_bbox.y_max - selected_bbox.y_min
    score = f" score {selected_bbox.score:.2f}" if selected_bbox.score is not None else ""
    label = f" label {selected_bbox.label}" if selected_bbox.label else ""
    return f"bbox {selected_bbox.x_min},{selected_bbox.y_min} {width}x{height}{score}{label}"


def _get_scale_mode(processed_width: int, original_width: int) -> str:
    if processed_width > original_width:
        return "upscale"
    if processed_width < original_width:
        return "downscale"
    return "no-scale"


def _get_detail_delta_lines(processed_crop: ImageType, detailed_crop: ImageType, denoise_mask: ImageType) -> list[str]:
    crop = processed_crop.convert("RGB")
    detail = detailed_crop.convert("RGB").resize(crop.size, resample=Image.Resampling.LANCZOS)
    mask = denoise_mask.convert("L").resize(crop.size, resample=Image.Resampling.NEAREST)
    edit_region = np.array(mask, dtype=np.uint8) < 128

    if not np.any(edit_region):
        return ["luma delta n/a", "rgb delta n/a"]

    crop_np = np.array(crop, dtype=np.float32)
    detail_np = np.array(detail, dtype=np.float32)
    delta = detail_np - crop_np
    mean_rgb_delta = delta[edit_region].mean(axis=0)
    luma_delta = mean_rgb_delta[0] * 0.2126 + mean_rgb_delta[1] * 0.7152 + mean_rgb_delta[2] * 0.0722

    return [
        f"luma delta {luma_delta:+.1f}",
        f"rgb delta {mean_rgb_delta[0]:+.1f},{mean_rgb_delta[1]:+.1f},{mean_rgb_delta[2]:+.1f}",
    ]


def create_detailer_debug_collage(
    base_image: ImageType,
    selected_bounding_boxes: list[BoundingBoxField],
    mask: ImageType,
    processed_crop: ImageType,
    denoise_mask: ImageType,
    paste_alpha_mask: ImageType,
    detailed_crop: ImageType,
    final_image: ImageType,
    x: int,
    y: int,
    original_width: int,
    original_height: int,
    processed_width: int,
    processed_height: int,
    detected: bool,
    target_prompt: str,
    detector_prompt: str = "",
    sam_model: str = "",
) -> ImageType:
    effective_upscale = processed_width / original_width if original_width > 0 else 1
    selected_bbox = selected_bounding_boxes[0] if selected_bounding_boxes else None
    selected_bbox_text = _format_selected_bbox(selected_bbox)
    scale_mode = _get_scale_mode(processed_width=processed_width, original_width=original_width)
    detail_delta_lines = _get_detail_delta_lines(
        processed_crop=processed_crop,
        detailed_crop=detailed_crop,
        denoise_mask=denoise_mask,
    )
    base_overlay = _make_base_overlay(
        base_image=base_image,
        mask=mask,
        selected_bounding_boxes=selected_bounding_boxes,
        x=x,
        y=y,
        original_width=original_width,
        original_height=original_height,
    )

    panels = [
        _make_debug_panel(
            title="Base / bbox / mask / crop",
            image=base_overlay,
            lines=[
                f"target {target_prompt or 'face'}",
                f"detector {detector_prompt or target_prompt or 'face'}",
                *([f"sam {sam_model}"] if sam_model else []),
                f"detected {str(detected).lower()}",
                selected_bbox_text,
            ],
        ),
        _make_debug_panel(
            title="Processed crop",
            image=processed_crop,
            lines=[
                f"crop {x},{y} {original_width}x{original_height}",
                f"process {processed_width}x{processed_height}",
                f"scale {scale_mode} {effective_upscale:.2f}x",
            ],
        ),
        _make_debug_panel(
            title="Denoise mask",
            image=_make_mask_visual(denoise_mask, (239, 68, 68)),
            lines=["black edits", "white preserves"],
        ),
        _make_debug_panel(
            title="Paste alpha",
            image=_make_mask_visual(paste_alpha_mask, (59, 130, 246)),
            lines=["white pastes", "black preserves"],
        ),
        _make_debug_panel(
            title="Detailed crop before paste",
            image=detailed_crop,
            lines=[f"detail {detailed_crop.width}x{detailed_crop.height}", *detail_delta_lines],
        ),
        _make_debug_panel(
            title="Final pasted result",
            image=final_image,
            lines=[f"final {final_image.width}x{final_image.height}"],
        ),
    ]

    collage = Image.new(
        "RGBA",
        (
            DEBUG_COLLAGE_PANEL_WIDTH * DEBUG_COLLAGE_COLUMNS,
            DEBUG_COLLAGE_PANEL_HEIGHT * DEBUG_COLLAGE_ROWS,
        ),
        (18, 20, 25, 255),
    )
    for index, panel in enumerate(panels):
        column = index % DEBUG_COLLAGE_COLUMNS
        row = index // DEBUG_COLLAGE_COLUMNS
        collage.alpha_composite(panel, (column * DEBUG_COLLAGE_PANEL_WIDTH, row * DEBUG_COLLAGE_PANEL_HEIGHT))

    return collage


@invocation(
    "detailer_debug_collage",
    title="Detailer Debug Collage",
    tags=["image", "mask", "detailer", "debug"],
    category="image",
    version="1.0.0",
)
class DetailerDebugCollageInvocation(BaseInvocation):
    """Create a diagnostic collage for the DINO/SAM detailer path."""

    base_image: ImageField = InputField(description="The original base image.")
    selected_bounding_boxes: list[BoundingBoxField] = InputField(
        default=[],
        description="The selected bounding box collection.",
    )
    mask: ImageField = InputField(description="The full-size segmentation mask.")
    processed_crop: ImageField = InputField(description="The processing-sized crop image.")
    denoise_mask: ImageField = InputField(description="The processing-sized denoise mask.")
    paste_alpha_mask: ImageField = InputField(description="The original crop-sized paste alpha mask.")
    detailed_crop: ImageField = InputField(description="The detailed crop before paste-back.")
    final_image: ImageField = InputField(description="The final pasted detailer result.")
    x: int = InputField(default=0, ge=0, description="The x coordinate of the original crop's left side.")
    y: int = InputField(default=0, ge=0, description="The y coordinate of the original crop's top side.")
    original_width: int = InputField(default=64, ge=1, description="The original crop width.")
    original_height: int = InputField(default=64, ge=1, description="The original crop height.")
    processed_width: int = InputField(default=64, ge=1, description="The processing crop width.")
    processed_height: int = InputField(default=64, ge=1, description="The processing crop height.")
    detected: bool = InputField(default=False, description="Whether a usable target was detected.")
    target_prompt: str = InputField(default="face", description="The detector target prompt.")
    detector_prompt: str = InputField(default="", description="The actual detector prompt.")
    sam_model: str = InputField(default="", description="The SAM model used for segmentation.")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        base_image = context.images.get_pil(self.base_image.image_name, mode="RGBA")
        mask = context.images.get_pil(self.mask.image_name, mode="L")
        processed_crop = context.images.get_pil(self.processed_crop.image_name, mode="RGBA")
        denoise_mask = context.images.get_pil(self.denoise_mask.image_name, mode="L")
        paste_alpha_mask = context.images.get_pil(self.paste_alpha_mask.image_name, mode="L")
        detailed_crop = context.images.get_pil(self.detailed_crop.image_name, mode="RGBA")
        final_image = context.images.get_pil(self.final_image.image_name, mode="RGBA")
        collage = create_detailer_debug_collage(
            base_image=base_image,
            selected_bounding_boxes=self.selected_bounding_boxes,
            mask=mask,
            processed_crop=processed_crop,
            denoise_mask=denoise_mask,
            paste_alpha_mask=paste_alpha_mask,
            detailed_crop=detailed_crop,
            final_image=final_image,
            x=self.x,
            y=self.y,
            original_width=self.original_width,
            original_height=self.original_height,
            processed_width=self.processed_width,
            processed_height=self.processed_height,
            detected=self.detected,
            target_prompt=self.target_prompt,
            detector_prompt=self.detector_prompt,
            sam_model=self.sam_model,
        )
        image_dto = context.images.save(image=collage)

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=collage.width,
            height=collage.height,
        )


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
