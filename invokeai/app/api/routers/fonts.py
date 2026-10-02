"""Authenticated custom-font catalog and file routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.fonts.fonts_common import FontAxis, FontInstance, FontRecord, FontScope
from invokeai.app.services.fonts.fonts_default import (
    FONT_MEDIA_TYPES,
    FontChangedError,
    FontDeleteForbiddenError,
    FontForbiddenError,
    FontInstanceError,
    FontNotFoundError,
    FontQuotaExceededError,
    FontService,
    FontStorageError,
    FontValidationError,
)
from invokeai.app.services.shared.pagination import MAX_PAGE_SIZE

fonts_router = APIRouter(prefix="/v1/fonts", tags=["fonts"])

# The request-body limiter runs before Starlette parses multipart uploads. Keep the framing
# allowance bounded so the configured font limit remains the dominant part of the request cap.
FONT_UPLOAD_MULTIPART_OVERHEAD = 1024 * 1024
MAX_CONCURRENT_FONT_UPLOADS = 2
MAX_CONCURRENT_FONT_UPLOADS_PER_USER = 1
FONT_UPLOAD_IDLE_TIMEOUT_SECONDS = 30.0
FONT_UPLOAD_MAX_DURATION_SECONDS = 120.0


class FontAxisDTO(BaseModel):
    tag: str
    label: str
    minimum: float
    default: float
    maximum: float
    hidden: bool


class FontInstanceDTO(BaseModel):
    name: str
    coordinates: dict[str, float]


class FontDTO(BaseModel):
    id: str
    family: str
    label: str
    style: str
    weight: int
    content_hash: str
    scope: Literal["private", "shared"]
    source: Literal["uploaded", "directory"]
    filename: str
    byte_size: int
    url: str
    axes: list[FontAxisDTO]
    instances: list[FontInstanceDTO]


class FontListResponse(BaseModel):
    offset: int
    limit: int
    total: int
    items: list[FontDTO]


class FontUploadResponse(BaseModel):
    font: FontDTO
    created: bool


class FontValidationResponse(BaseModel):
    filename: str
    family: str
    label: str
    style: str
    weight: int
    content_hash: str
    byte_size: int
    axes: list[FontAxisDTO]
    instances: list[FontInstanceDTO]


class FontInstanceRequest(BaseModel):
    content_hash: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    coordinates: dict[str, float] = Field(default_factory=dict)


class FontRescanResponse(BaseModel):
    indexed: int
    revision: int


def _get_font_service() -> FontService:
    service = getattr(ApiDependencies.invoker.services, "fonts", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Font service is unavailable")
    return cast(FontService, service)


def _font_url(font_id: str) -> str:
    return f"/api/v1/fonts/{quote(font_id, safe='')}/file"


def _axis_dto(axis: FontAxis) -> FontAxisDTO:
    return FontAxisDTO(
        tag=axis.tag,
        label=axis.label,
        minimum=axis.minimum,
        default=axis.default,
        maximum=axis.maximum,
        hidden=axis.hidden,
    )


def _instance_dto(instance: FontInstance) -> FontInstanceDTO:
    return FontInstanceDTO(name=instance.name, coordinates=instance.coordinates)


def _font_dto(font: FontRecord) -> FontDTO:
    return FontDTO(
        id=font.id,
        family=font.family,
        label=font.label,
        style=font.style,
        weight=font.weight,
        content_hash=font.content_hash,
        scope=font.scope.value,
        source=font.source.value,
        filename=font.filename,
        byte_size=font.byte_size,
        url=_font_url(font.id),
        axes=[_axis_dto(axis) for axis in font.axes],
        instances=[_instance_dto(instance) for instance in font.instances],
    )


def _inline_content_disposition(filename: str) -> str:
    safe_filename = filename.replace("\r", "").replace("\n", "")
    return f"inline; filename*=UTF-8''{quote(safe_filename, safe='')}"


def _get_font_cache_control(font: FontRecord | None = None) -> str:
    """Prevent cross-account browser reuse of font responses in multiuser mode."""
    if ApiDependencies.invoker.services.configuration.multiuser:
        return "private, no-store"
    if font is not None and font.source.value == "uploaded":
        return "private, max-age=31536000, immutable"
    return "private, no-cache"


def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = file.file.read(min(1024 * 1024, max_bytes - size + 1))
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise FontQuotaExceededError("Font file exceeds the upload size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _raise_font_http_error(error: Exception) -> None:
    if isinstance(error, FontNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Font not found") from error
    if isinstance(error, FontChangedError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    if isinstance(error, FontDeleteForbiddenError | FontForbiddenError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(error)) from error
    if isinstance(error, FontQuotaExceededError):
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(error)) from error
    if isinstance(error, FontValidationError | FontInstanceError):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    if isinstance(error, FontStorageError):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Font storage failed") from error
    raise error


@fonts_router.get("", response_model=FontListResponse, operation_id="list_fonts")
def list_fonts(
    current_user: CurrentUserOrDefault,
    response: Response,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=0, le=MAX_PAGE_SIZE),
    search: str | None = Query(default=None, max_length=200),
    scope: FontScope = Query(default=FontScope.ALL),
    content_hash: str | None = Query(default=None, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$"),
) -> FontListResponse:
    service = _get_font_service()
    try:
        fonts, total = service.list(
            user_id=current_user.user_id,
            offset=offset,
            limit=limit,
            search=search,
            scope=scope,
            content_hash=content_hash,
        )
    except Exception as e:
        _raise_font_http_error(e)
    response.headers["Cache-Control"] = _get_font_cache_control()
    response.headers["X-Font-Catalog-Revision"] = str(service.revision)
    return FontListResponse(offset=offset, limit=limit, total=total, items=[_font_dto(font) for font in fonts])


@fonts_router.post("/validate", response_model=FontValidationResponse, operation_id="validate_font")
def validate_font(
    current_user: CurrentUserOrDefault, file: UploadFile = File(description="The font file to validate")
) -> FontValidationResponse:
    del current_user
    service = _get_font_service()
    try:
        validation = service.validate(file.filename or "", _read_upload(file, service.max_upload_bytes))
    except Exception as e:
        _raise_font_http_error(e)
    return FontValidationResponse(
        filename=validation.filename,
        family=validation.family,
        label=validation.label,
        style=validation.style,
        weight=validation.weight,
        content_hash=validation.content_hash,
        byte_size=validation.byte_size,
        axes=[_axis_dto(axis) for axis in validation.axes],
        instances=[_instance_dto(instance) for instance in validation.instances],
    )


@fonts_router.post("", response_model=FontUploadResponse, operation_id="upload_font")
def upload_font(
    current_user: CurrentUserOrDefault,
    response: Response,
    file: UploadFile = File(description="The font file to upload"),
    scope: str = Form(default=FontScope.PRIVATE.value),
) -> FontUploadResponse:
    service = _get_font_service()
    try:
        requested_scope = FontScope(scope)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid font scope") from e
    if requested_scope == FontScope.ALL:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid font scope")
    if requested_scope == FontScope.SHARED and not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required for shared fonts")
    try:
        result = service.upload(
            user_id=current_user.user_id,
            filename=file.filename or "",
            data=_read_upload(file, service.max_upload_bytes),
            scope=requested_scope,
        )
    except Exception as e:
        _raise_font_http_error(e)
    response.headers["X-Font-Catalog-Revision"] = str(service.revision)
    return FontUploadResponse(font=_font_dto(result.font), created=result.created)


@fonts_router.post("/rescan", response_model=FontRescanResponse, operation_id="rescan_fonts")
def rescan_fonts(_admin_user: AdminUserOrDefault) -> FontRescanResponse:
    service = _get_font_service()
    return FontRescanResponse(indexed=service.rescan_directory(), revision=service.revision)


@fonts_router.get("/{font_id}/file", operation_id="get_font_file")
def get_font_file(
    current_user: CurrentUserOrDefault,
    font_id: str = PathParam(min_length=1),
    expected_hash: str | None = Query(default=None, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$"),
) -> Response:
    service = _get_font_service()
    try:
        font, data = service.read_file(user_id=current_user.user_id, font_id=font_id, expected_hash=expected_hash)
    except Exception as e:
        _raise_font_http_error(e)
    extension = Path(font.filename).suffix.lower()
    return Response(
        content=data,
        media_type=FONT_MEDIA_TYPES.get(extension, "application/octet-stream"),
        headers={
            "Cache-Control": _get_font_cache_control(font),
            "ETag": f'"{font.content_hash}"',
            "Content-Disposition": _inline_content_disposition(font.filename),
        },
    )


@fonts_router.get("/{font_id}", response_model=FontDTO, operation_id="get_font")
def get_font(current_user: CurrentUserOrDefault, response: Response, font_id: str = PathParam(min_length=1)) -> FontDTO:
    service = _get_font_service()
    try:
        font = service.get_accessible(user_id=current_user.user_id, font_id=font_id)
    except Exception as e:
        _raise_font_http_error(e)
    response.headers["Cache-Control"] = _get_font_cache_control()
    return _font_dto(font)


@fonts_router.post("/{font_id}/instance", operation_id="create_font_instance")
def create_font_instance(
    current_user: CurrentUserOrDefault,
    request: FontInstanceRequest,
    font_id: str = PathParam(min_length=1),
) -> Response:
    service = _get_font_service()
    try:
        font, data = service.instance(
            user_id=current_user.user_id,
            font_id=font_id,
            content_hash=request.content_hash,
            coordinates=request.coordinates,
        )
    except Exception as e:
        _raise_font_http_error(e)
    extension = Path(font.filename).suffix.lower()
    return Response(
        content=data,
        media_type=FONT_MEDIA_TYPES.get(extension, "application/octet-stream"),
        headers={
            "Cache-Control": _get_font_cache_control(font),
            "ETag": f'"{font.content_hash}-{hashlib.sha256(json.dumps(request.coordinates, sort_keys=True).encode()).hexdigest()[:16]}"',
            "Content-Disposition": _inline_content_disposition(f"{Path(font.filename).stem}-instance{extension}"),
        },
    )


@fonts_router.delete("/{font_id}", status_code=status.HTTP_204_NO_CONTENT, operation_id="delete_font")
def delete_font(current_user: CurrentUserOrDefault, font_id: str = PathParam(min_length=1)) -> Response:
    service = _get_font_service()
    try:
        service.delete(user_id=current_user.user_id, font_id=font_id, is_admin=current_user.is_admin)
    except Exception as e:
        _raise_font_http_error(e)
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"X-Font-Catalog-Revision": str(service.revision)})


__all__ = [
    "FontDTO",
    "FontListResponse",
    "FontUploadResponse",
    "fonts_router",
]
