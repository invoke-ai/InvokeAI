"""Exporting stored images: encoding to another format and size, and naming the exported files.

The single-image export route and the bulk download archive both use this module, so a preview is the exact bytes a
download holds and a file name shown in the dialog is the name the archive entry gets.
"""

import io
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Sequence, Union

from PIL import Image, features
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel, Field, StringConstraints

from invokeai.app.services.images.images_common import ImageDTO

if TYPE_CHECKING:
    from invokeai.app.services.invocation_services import InvocationServices

ImageExportFormat = Literal["png", "jpeg", "webp", "avif"]
ImageExportMetadata = Literal["all", "parameters", "none"]
ImageExportNamePattern = Annotated[str, StringConstraints(max_length=200)]

EXPORT_MEDIA_TYPES: dict[ImageExportFormat, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "avif": "image/avif",
}
EXPORT_EXTENSIONS: dict[ImageExportFormat, str] = {"png": ".png", "jpeg": ".jpg", "webp": ".webp", "avif": ".avif"}
_PIL_FORMATS: dict[ImageExportFormat, str] = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP", "avif": "AVIF"}
_PIL_FEATURES: dict[ImageExportFormat, str] = {"png": "zlib", "jpeg": "jpg", "webp": "webp", "avif": "avif"}

# Encodes are CPU-bound; a burst of preview requests must not take every core. The AVIF encoder is multithreaded
# and would otherwise use all of them per encode, so its threads share the same budget.
_ENCODE_CORES = max(1, (os.cpu_count() or 2) // 2)
_ENCODE_SLOT_COUNT = min(4, _ENCODE_CORES)
_ENCODE_SLOTS = threading.BoundedSemaphore(_ENCODE_SLOT_COUNT)
_AVIF_MAX_THREADS = max(1, _ENCODE_CORES // _ENCODE_SLOT_COUNT)


class ImageExportOptions(BaseModel):
    format: ImageExportFormat = Field(description="The format to encode the image in")
    quality: int = Field(default=90, ge=1, le=100, description="Encoder quality for lossy formats; ignored for PNG")
    long_edge: Optional[int] = Field(
        default=None, ge=1, description="Caps the longer side in pixels. The image is never enlarged."
    )
    metadata: ImageExportMetadata = Field(
        default="none",
        description="Which Invoke metadata a PNG keeps; other formats never carry any",
    )


class BulkImageExportOptions(ImageExportOptions):
    name_pattern: ImageExportNamePattern = Field(
        default="{name}", description="The file-name pattern for every exported image"
    )


class ImageExportFormatUnavailableError(Exception):
    """This server's Pillow build cannot encode the requested format."""


def assert_export_format_available(format: ImageExportFormat) -> None:
    if not features.check(_PIL_FEATURES[format]):
        raise ImageExportFormatUnavailableError(f"This server cannot encode {format.upper()} images")


def encode_image(path: Union[str, Path], options: ImageExportOptions) -> bytes:
    """Encode the stored image file at `path` with `options`.

    Raises FileNotFoundError for a missing file and ImageExportFormatUnavailableError for an unavailable codec.
    """
    assert_export_format_available(options.format)
    with _ENCODE_SLOTS, Image.open(path) as source:
        # PNG text chunks after IDAT reach .info only once the pixels are loaded.
        source.load()
        info = source.info
        icc_profile = info.get("icc_profile")
        buffer = io.BytesIO()
        if options.format == "png":
            image = _fit_png(source, options.long_edge)
            transparency = image.info.get("transparency") if image is source else None
            pnginfo = PngInfo()
            for key, value in info.items():
                if isinstance(value, str) and _keeps_png_text(key, options.metadata):
                    pnginfo.add_text(key, value)
            image.info = {}
            image.save(buffer, "PNG", pnginfo=pnginfo, icc_profile=icc_profile, transparency=transparency)
        else:
            image = _resize(_to_8bit_rgb(source), options.long_edge)
            if options.format == "jpeg" and image.mode == "RGBA":
                matte = Image.new("RGB", image.size, (255, 255, 255))
                matte.paste(image, mask=image.getchannel("A"))
                image = matte
            image.info = {}
            extra = {"max_threads": _AVIF_MAX_THREADS} if options.format == "avif" else {}
            image.save(buffer, _PIL_FORMATS[options.format], quality=options.quality, icc_profile=icc_profile, **extra)
        return buffer.getvalue()


def _keeps_png_text(key: str, metadata: ImageExportMetadata) -> bool:
    if metadata == "all":
        return key.startswith("invokeai_")
    return metadata == "parameters" and key == "invokeai_metadata"


def _has_alpha(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def _target_size(size: tuple[int, int], long_edge: Optional[int]) -> Optional[tuple[int, int]]:
    width, height = size
    longest = max(width, height)
    if long_edge is None or long_edge >= longest:
        return None
    scale = long_edge / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def _resize(image: Image.Image, long_edge: Optional[int]) -> Image.Image:
    size = _target_size(image.size, long_edge)
    return image if size is None else image.resize(size, Image.Resampling.LANCZOS)


def _fit_png(image: Image.Image, long_edge: Optional[int]) -> Image.Image:
    if _target_size(image.size, long_edge) is None:
        return image
    # Palette, bilevel and colour-key transparency cannot be resampled smoothly; Pillow would fall back to nearest.
    if image.mode in ("1", "P", "PA") or "transparency" in image.info:
        image = image.convert("RGBA" if _has_alpha(image) else "RGB")
    return _resize(image, long_edge)


def _to_8bit_rgb(image: Image.Image) -> Image.Image:
    if image.mode.startswith("I"):
        # Pillow clips 16-bit values into 8 bits on conversion, which turns most of a depth map white: scale instead.
        image = image.convert("I").point(lambda value: value / 256).convert("L")
    return image.convert("RGBA" if _has_alpha(image) else "RGB")


# File names

_TOKENS = re.compile(r"\{(name|board|seed|model|date|n)\}")
_UNSAFE_CHARACTERS = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f-\x9f]')
_SEPARATORS = "-_ ."
MAX_EXPORT_STEM_BYTES = 120


def resolve_export_file_names(
    services: "InvocationServices",
    images: Sequence[ImageDTO],
    pattern: str,
    format: ImageExportFormat,
) -> list[str]:
    """Resolve `pattern` into one unique, path-safe file name per image, in order."""
    parts = _TOKENS.split(pattern)
    tokens = set(parts[1::2])
    reads_metadata = bool(tokens & {"seed", "model"})
    board_names: dict[str, str] = {}
    extension = EXPORT_EXTENSIONS[format]
    position_width = max(3, len(str(len(images))))
    used: set[str] = set()
    file_names: list[str] = []

    for position, image in enumerate(images, start=1):
        name = os.path.splitext(image.image_name)[0]
        values = {"name": name, "n": str(position).zfill(position_width), "date": _date_token(image.created_at)}
        if "board" in tokens:
            if image.board_id is None:
                values["board"] = "uncategorized"
            else:
                if image.board_id not in board_names:
                    board_names[image.board_id] = services.board_records.get(image.board_id).board_name
                values["board"] = board_names[image.board_id]
        if reads_metadata:
            metadata = services.images.get_metadata(image.image_name)
            values.update(_metadata_tokens(metadata.root if metadata is not None else {}))

        stem = _cut_stem(_render(parts, values) or _sanitize(name))
        candidate, suffix = stem, 2
        while (candidate + extension).casefold() in used:
            candidate, suffix = f"{stem}-{suffix}", suffix + 1
        used.add((candidate + extension).casefold())
        file_names.append(candidate + extension)

    return file_names


def _render(parts: list[str], values: dict[str, str]) -> str:
    rendered = ""
    after_empty_token = False
    for index, part in enumerate(parts):
        text = _sanitize(values.get(part, "") if index % 2 else part)
        if index % 2 and not text:
            after_empty_token = True
            continue
        # An empty token leaves the separators on both of its sides: keep the run before it, drop the one after.
        if after_empty_token and (not rendered or rendered[-1] in _SEPARATORS):
            text = text.lstrip(_SEPARATORS)
        if text:
            after_empty_token = False
        rendered += text
    return rendered.strip(_SEPARATORS)


def _sanitize(text: str) -> str:
    return _UNSAFE_CHARACTERS.sub("-", text)


def _cut_stem(stem: str) -> str:
    cut = stem.encode("utf-8")[:MAX_EXPORT_STEM_BYTES].decode("utf-8", "ignore")
    return cut.rstrip(_SEPARATORS)


def _date_token(created_at: Union[datetime, str]) -> str:
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            return ""
    return created_at.strftime("%Y-%m-%d")


def _metadata_tokens(metadata: dict) -> dict[str, str]:
    seed = metadata.get("seed")
    model = metadata.get("model")
    model_name = model.get("name") if isinstance(model, dict) else None
    return {
        "seed": "" if seed is None or isinstance(seed, bool) else str(seed),
        "model": model_name if isinstance(model_name, str) else "",
    }
