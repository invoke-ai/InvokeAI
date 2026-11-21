import io
import os
import re
import threading
import unicodedata
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Sequence, Union

import anyio
from PIL import Image, UnidentifiedImageError, features
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel, Field, StringConstraints

from invokeai.app.services.board_access.board_access_common import BoardAction, BoardViewer
from invokeai.app.services.board_records.board_records_common import BoardRecordNotFoundException
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


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# Encodes get half the cores, AVIF threads included; archives have their own pool so they never queue previews.
_ENCODE_CORES = max(1, _available_cpus() // 2)
_REQUEST_SLOT_COUNT = min(4, _ENCODE_CORES)
_REQUEST_SLOTS = anyio.Semaphore(_REQUEST_SLOT_COUNT)
_ARCHIVE_SLOTS = threading.BoundedSemaphore(max(1, _REQUEST_SLOT_COUNT // 2))
_AVIF_MAX_THREADS = max(1, _ENCODE_CORES // _REQUEST_SLOT_COUNT)


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


class ImageExportError(Exception):
    """An export the server refuses; its message is shown to the user and must name no path."""


class ImageExportFormatUnavailableError(ImageExportError):
    pass


class ImageExportSourceError(ImageExportError):
    pass


def assert_export_format_available(format: ImageExportFormat) -> None:
    if not features.check(_PIL_FEATURES[format]):
        raise ImageExportFormatUnavailableError(f"This server cannot encode {format.upper()} images")


def encode_image(path: Union[str, Path], options: ImageExportOptions) -> bytes:
    stored = _stored_file_if_final(path, options)
    if stored is not None:
        return stored
    with _ARCHIVE_SLOTS:
        return _encode(path, options)


async def encode_image_for_request(
    path: Union[str, Path], options: ImageExportOptions, client_gone: Callable[[], Awaitable[None]]
) -> Optional[bytes]:
    stored = await anyio.to_thread.run_sync(_stored_file_if_final, path, options)
    if stored is not None:
        return stored
    if not await _acquire_request_slot(client_gone):
        return None
    try:
        return await anyio.to_thread.run_sync(_encode, path, options)
    finally:
        _REQUEST_SLOTS.release()


async def _acquire_request_slot(client_gone: Callable[[], Awaitable[None]]) -> bool:
    gone = granted = False

    async def watch_client() -> None:
        nonlocal gone
        await client_gone()
        gone = True
        task_group.cancel_scope.cancel()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(watch_client)
        await _REQUEST_SLOTS.acquire()
        granted = True
        task_group.cancel_scope.cancel()
    if granted and gone:
        _REQUEST_SLOTS.release()
    return granted and not gone


def _stored_file_if_final(path: Union[str, Path], options: ImageExportOptions) -> Optional[bytes]:
    if options.format != "png" or options.metadata != "all":
        return None
    try:
        with Image.open(path) as source:
            if source.format != "PNG" or _target_size(source.size, options.long_edge) is not None:
                return None
    except (UnidentifiedImageError, Image.DecompressionBombError):
        return None
    return Path(path).read_bytes()


def _decode(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        # PNG text chunks after IDAT reach .info only once the pixels are loaded.
        image.load()
    except Image.DecompressionBombError as e:
        raise ImageExportSourceError("This image is too large to export") from e
    except OSError as e:
        raise ImageExportSourceError("This image's stored file is damaged or is not an image") from e
    return image


def _encode(path: Union[str, Path], options: ImageExportOptions) -> bytes:
    assert_export_format_available(options.format)
    with _decode(Path(path).read_bytes()) as source:
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
    if image.mode in ("1", "P", "PA") or "transparency" in image.info:
        image = image.convert("RGBA" if _has_alpha(image) else "RGB")
    return _resize(image, long_edge)


def _to_8bit_rgb(image: Image.Image) -> Image.Image:
    if image.mode.startswith("I"):
        # Pillow clips 16-bit values on conversion, turning most of a depth map white: scale first.
        image = image.convert("I").point(lambda value: value / 256).convert("L")
    return image.convert("RGBA" if _has_alpha(image) else "RGB")


_TOKENS = re.compile(r"\{(name|board|seed|model|date|n)\}")
_UNSAFE_CHARACTERS = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f-\x9f]')
_SEPARATORS = "-_ ."
MAX_EXPORT_STEM_BYTES = 120
_RESERVED_STEMS = frozenset(
    ["CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))]
)


def resolve_export_file_names(
    services: "InvocationServices",
    images: Sequence[ImageDTO],
    pattern: str,
    format: ImageExportFormat,
    viewer: BoardViewer,
) -> list[str]:
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
                    board_names[image.board_id] = _readable_board_name(services, image.board_id, viewer)
                values["board"] = board_names[image.board_id]
        if reads_metadata:
            metadata = services.images.get_metadata(image.image_name)
            values.update(_metadata_tokens(metadata.root if metadata is not None else {}))

        stem = _avoid_reserved_stem(_cut_stem(_render(parts, values) or _sanitize(name)))
        candidate, suffix = stem, 2
        while (candidate + extension).casefold() in used:
            candidate, suffix = f"{stem}-{suffix}", suffix + 1
        used.add((candidate + extension).casefold())
        file_names.append(candidate + extension)

    return file_names


def _readable_board_name(services: "InvocationServices", board_id: str, viewer: BoardViewer) -> str:
    try:
        if not services.board_access.get(board_id, viewer.user_id, viewer.is_admin).allows(BoardAction.Read):
            return ""
        return services.board_records.get(board_id).board_name
    except BoardRecordNotFoundException:
        return ""


def _render(parts: list[str], values: dict[str, str]) -> str:
    rendered = ""
    after_empty_token = False
    for index, part in enumerate(parts):
        text = _sanitize(values.get(part, "") if index % 2 else part)
        if index % 2 and not text:
            after_empty_token = True
            continue
        if after_empty_token and (not rendered or rendered[-1] in _SEPARATORS):
            text = text.lstrip(_SEPARATORS)
        if text:
            after_empty_token = False
        rendered += text
    return rendered.strip(_SEPARATORS)


def _sanitize(text: str) -> str:
    text = _UNSAFE_CHARACTERS.sub("-", text)
    if text.isascii():
        return text
    return "".join("-" if unicodedata.category(character) == "Cf" else character for character in text)


def _avoid_reserved_stem(stem: str) -> str:
    head, dot, rest = stem.partition(".")
    if head.upper() not in _RESERVED_STEMS:
        return stem
    return _cut_stem(f"{head}_{dot}{rest}") if dot else f"{head}_"


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
