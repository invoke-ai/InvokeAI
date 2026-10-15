"""The custom font library.

The service keeps the configured ``fonts_dir`` as a read-only, shared source and stores
uploads in an application-owned directory. Font files are addressed by opaque IDs and
content hashes, so a changed directory file cannot silently satisfy an old project
reference.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import pickle
import re
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any
from uuid import uuid4

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

from invokeai.app.services.fonts.fonts_common import (
    FontAxis,
    FontInstance,
    FontRecord,
    FontScope,
    FontSource,
    FontUploadResult,
    FontValidationResult,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.fonts import MAX_SOURCE_PATH_LENGTH, DirectoryFont, UploadedFont

SUPPORTED_FONT_EXTENSIONS = frozenset({".ttf", ".otf", ".woff", ".woff2"})
FONT_MEDIA_TYPES = {
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}

MAX_FONT_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_FONT_LIBRARY_BYTES = 1024 * 1024 * 1024
MAX_INDEXED_FONT_BYTES = 128 * 1024 * 1024
MAX_INSTANCE_JOBS = 2
MAX_INSTANCE_QUEUE = 4
MAX_INSTANCE_CACHE_BYTES = 64 * 1024 * 1024
INSTANCE_JOB_TIMEOUT_SECONDS = 30.0


class FontServiceError(RuntimeError):
    """Base class for errors raised by the font service."""


class FontNotFoundError(FontServiceError):
    """The font does not exist or is not visible to the requesting account."""


class FontForbiddenError(FontServiceError):
    """The caller is not permitted to perform the requested operation."""


class FontDeleteForbiddenError(FontForbiddenError):
    """A read-only or otherwise protected font cannot be deleted."""


class FontValidationError(FontServiceError):
    """The file is too large, unsupported, malformed, or missing font metadata."""


class FontQuotaExceededError(FontServiceError):
    """An upload would exceed the configured library quota."""


class FontChangedError(FontServiceError):
    """A font ID no longer points at the requested content hash."""


class FontInstanceError(FontServiceError):
    """A variable font instance could not be produced."""


class FontStorageError(FontServiceError):
    """A managed font file could not be read or safely stored."""


def _normalize_variant_text(value: str) -> str:
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def _infer_font_weight(style_name: str, filename: str, weight_class: int | None) -> int:
    if isinstance(weight_class, int) and 1 <= weight_class <= 1000:
        return weight_class

    combined = _normalize_variant_text(f"{style_name} {filename}")
    weight_keywords = [
        (("thin", "hairline"), 100),
        (("extra light", "ultra light", "extralight", "ultralight"), 200),
        (("light",), 300),
        (("normal", "regular", "roman", "book"), 400),
        (("medium",), 500),
        (("semi bold", "semibold", "demi bold", "demibold"), 600),
        (("extra bold", "ultra bold", "extrabold", "ultrabold"), 800),
        (("black", "heavy"), 900),
        (("bold",), 700),
    ]
    for keywords, weight in weight_keywords:
        if any(keyword in combined for keyword in keywords):
            return weight
    return 400


def _infer_font_style(style_name: str, filename: str, italic_flag: bool) -> str:
    if italic_flag:
        return "italic"
    combined = _normalize_variant_text(f"{style_name} {filename}")
    return "italic" if "italic" in combined or "oblique" in combined else "normal"


def _get_name_table_value(font: TTFont, name_ids: tuple[int, ...]) -> str | None:
    """Read a localized OpenType name, preferring English Windows records."""
    if "name" not in font:
        return None

    def sort_key(record: Any) -> tuple[int, int]:
        platform_id = getattr(record, "platformID", -1)
        language_id = getattr(record, "langID", -1)
        if platform_id == 3 and language_id in (0x0409, 0):
            return (0, 0)
        if platform_id == 3:
            return (1, 0)
        if platform_id == 0:
            return (2, 0)
        return (3, language_id)

    records = font["name"].names
    for name_id in name_ids:
        for record in sorted((record for record in records if record.nameID == name_id), key=sort_key):
            try:
                value = record.toUnicode().strip()
            except Exception:
                continue
            if value:
                return value
    return None


def _parse_font_bytes(filename: str, data: bytes, *, max_bytes: int) -> FontValidationResult:
    """Validate and extract metadata without persisting the supplied bytes."""
    if (
        not filename
        or Path(filename).name != filename
        or any(separator in filename for separator in ("/", "\\"))
        or any(ord(character) < 32 for character in filename)
        or filename in {".", ".."}
    ):
        raise FontValidationError("Font filename is invalid")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_FONT_EXTENSIONS:
        raise FontValidationError("Unsupported font format; use TTF, OTF, WOFF, or WOFF2")
    if len(data) == 0:
        raise FontValidationError("Font file is empty")
    if len(data) > max_bytes:
        raise FontValidationError(f"Font file exceeds the {max_bytes // (1024 * 1024)} MiB limit")

    try:
        with TTFont(file=io.BytesIO(data), lazy=False) as font:
            family = (_get_name_table_value(font, (16, 1, 4)) or "").strip()
            style_name = (_get_name_table_value(font, (17, 2)) or "").strip()
            label = (_get_name_table_value(font, (4,)) or family).strip()
            if not family:
                raise FontValidationError("Font is missing a family name")

            os2_table = font["OS/2"] if "OS/2" in font else None
            head_table = font["head"] if "head" in font else None
            post_table = font["post"] if "post" in font else None
            weight_class = getattr(os2_table, "usWeightClass", None)
            italic_flag = bool(getattr(os2_table, "fsSelection", 0) & 0x01) or bool(
                getattr(head_table, "macStyle", 0) & 0x02
            )
            if post_table is not None:
                italic_flag = italic_flag or bool(getattr(post_table, "italicAngle", 0))
            weight = _infer_font_weight(style_name, filename, weight_class)
            style = _infer_font_style(style_name, filename, italic_flag)

            axes: list[FontAxis] = []
            instances: list[FontInstance] = []
            if "fvar" in font:
                for axis in font["fvar"].axes:
                    axis_tag = str(axis.axisTag)
                    axis_label = _get_name_table_value(font, (int(axis.axisNameID),)) or axis_tag
                    axes.append(
                        FontAxis(
                            tag=axis_tag,
                            label=axis_label,
                            minimum=float(axis.minValue),
                            default=float(axis.defaultValue),
                            maximum=float(axis.maxValue),
                            hidden=bool(getattr(axis, "flags", 0) & 0x0001),
                        )
                    )
                for index, instance in enumerate(font["fvar"].instances):
                    instance_name = _get_name_table_value(font, (int(instance.subfamilyNameID),)) or (
                        f"Instance {index + 1}"
                    )
                    coordinates = {
                        axis.axisTag: float(instance.coordinates.get(axis.axisTag, axis.defaultValue))
                        for axis in font["fvar"].axes
                    }
                    instances.append(FontInstance(name=instance_name, coordinates=coordinates))
    except FontValidationError:
        raise
    except Exception as e:
        raise FontValidationError("The uploaded file is not a valid font") from e

    return FontValidationResult(
        filename=filename,
        family=family,
        label=label,
        style=style,
        weight=weight,
        content_hash=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        axes=tuple(axes),
        instances=tuple(instances),
    )


def _parse_font_worker(filename: str, data: bytes, max_bytes: int) -> FontValidationResult:
    """Pickle-friendly worker entry point for isolated metadata parsing."""
    return _parse_font_bytes(filename, data, max_bytes=max_bytes)


def _validation_from_record(record: FontRecord, filename: str) -> FontValidationResult:
    """Reuse indexed metadata when a directory file's content hash is unchanged."""
    return FontValidationResult(
        filename=filename,
        family=record.family,
        label=record.label,
        style=record.style,
        weight=record.weight,
        content_hash=record.content_hash,
        byte_size=record.byte_size,
        axes=record.axes,
        instances=record.instances,
    )


def _indexed_as(record: FontRecord | None, font: DirectoryFont) -> bool:
    """Whether the catalog already holds the directory font as found."""
    return (
        record is not None
        and record.filename == font.filename
        and record.source_path == font.source_path
        and record.family == font.family
        and record.label == font.label
        and record.style == font.style
        and record.weight == font.weight
        and record.content_hash == font.content_hash
        and record.byte_size == font.byte_size
        and record.axes == font.axes
        and record.instances == font.instances
    )


def _read_bounded_file(path: Path, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` and reject a file that grows during the read."""
    if path.stat().st_size > max_bytes:
        raise FontValidationError(f"Font file exceeds the {max_bytes // (1024 * 1024)} MiB limit")
    chunks: list[bytes] = []
    size = 0
    with path.open("rb") as file:
        while size <= max_bytes:
            chunk = file.read(min(1024 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise FontValidationError(f"Font file exceeds the {max_bytes // (1024 * 1024)} MiB limit")
    return b"".join(chunks)


def _safe_relative_path(path: Path, root: Path) -> Path:
    """Resolve a path under ``root`` while rejecting symlink components."""
    root_absolute = root.absolute()
    candidate = (root_absolute / path).absolute()
    try:
        candidate.relative_to(root_absolute)
    except ValueError as e:
        raise FontStorageError("Font path escapes its storage directory") from e

    current = candidate
    while True:
        if current.is_symlink():
            raise FontStorageError("Symlinked font paths are not supported")
        if current == root_absolute:
            break
        current = current.parent

    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError as e:
        raise FontStorageError("Font path escapes its storage directory") from e
    return candidate


class FontService:
    """Owns font metadata, managed files, access checks, and variable instances."""

    def __init__(
        self,
        database: Database,
        fonts_dir: Path,
        storage_dir: Path,
        logger: logging.Logger | None = None,
        *,
        max_upload_bytes: int = MAX_FONT_UPLOAD_BYTES,
        max_library_bytes: int = MAX_FONT_LIBRARY_BYTES,
        max_instance_cache_bytes: int = MAX_INSTANCE_CACHE_BYTES,
        instance_job_timeout_seconds: float = INSTANCE_JOB_TIMEOUT_SECONDS,
    ) -> None:
        fonts_absolute = fonts_dir.resolve()
        storage_absolute = storage_dir.resolve()
        if (
            fonts_absolute == storage_absolute
            or fonts_absolute.is_relative_to(storage_absolute)
            or storage_absolute.is_relative_to(fonts_absolute)
        ):
            raise ValueError("Font source and managed storage directories must not overlap")
        self._queries = database.queries
        self._fonts_dir = fonts_dir
        self._storage_dir = storage_dir
        self._logger = logger or logging.getLogger(__name__)
        self._max_upload_bytes = max_upload_bytes
        self._max_library_bytes = max_library_bytes
        self._max_instance_cache_bytes = max_instance_cache_bytes
        self._instance_job_timeout_seconds = instance_job_timeout_seconds
        # FontTools parses binary tables and performs variation instancing in native/Python
        # code that cannot be interrupted safely from a thread. Each operation runs in a
        # short-lived, lightweight worker subprocess so a timeout can terminate the work
        # without inheriting the server's CUDA or thread state.
        self._worker_admission = threading.BoundedSemaphore(MAX_INSTANCE_JOBS + MAX_INSTANCE_QUEUE)
        self._worker_slots = threading.BoundedSemaphore(MAX_INSTANCE_JOBS)
        self._active_workers: set[subprocess.Popen[bytes]] = set()
        self._active_workers_lock = threading.RLock()
        # Coordinate catalog snapshots with the managed-file write/commit sequence. A
        # cleanup sweep must not observe an upload between writing its file and committing
        # its row, or it could remove a live upload.
        self._managed_storage_lock = threading.RLock()
        # One scan at a time: two would race to insert the same new directory font.
        self._rescan_lock = threading.Lock()
        self._instance_cache: OrderedDict[tuple[str, tuple[tuple[str, float], ...]], bytes] = OrderedDict()
        self._instance_cache_bytes = 0
        self._cache_lock = threading.RLock()
        self._revision = 0
        self._revision_lock = threading.Lock()

    @property
    def revision(self) -> int:
        """Monotonic process-local revision for client catalog invalidation."""
        with self._revision_lock:
            return self._revision

    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted size for a single uploaded font."""
        return self._max_upload_bytes

    def _bump_revision(self) -> None:
        with self._revision_lock:
            self._revision += 1

    def start(self, _invoker: object | None = None) -> None:
        """Index configured directory fonts once the service is attached to the invoker."""
        self.cleanup_orphaned_files()
        self.rescan_directory()

    def stop(self, _invoker: object | None = None) -> None:
        """Terminate active font workers and release cached instances."""
        with self._active_workers_lock:
            workers = list(self._active_workers)
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
        self._invalidate_instance_cache()

    def validate(self, filename: str, data: bytes) -> FontValidationResult:
        """Validate a font upload and return metadata without changing the database."""
        if len(data) > self._max_upload_bytes:
            raise FontValidationError(f"Font file exceeds the {self._max_upload_bytes // (1024 * 1024)} MiB limit")
        return self._run_worker(
            "parse",
            (filename, data, self._max_upload_bytes),
            error_type=FontValidationError,
        )

    def upload(
        self, *, user_id: str, filename: str, data: bytes, scope: FontScope = FontScope.PRIVATE
    ) -> FontUploadResult:
        """Validate and atomically publish an uploaded font, deduplicating by content hash."""
        if scope not in (FontScope.PRIVATE, FontScope.SHARED):
            raise FontValidationError("Uploaded fonts must be private or shared")
        validation = self.validate(filename, data)
        owner_id = user_id if scope == FontScope.PRIVATE else None
        duplicate: FontRecord | None = None
        storage_path: str | None = None
        font_id = f"font_{uuid4().hex}"
        # The duplicate and quota checks, the file and the row are published together: every upload, and the
        # orphaned-file sweep, run under the managed-storage lock (one process serves the database). The file is
        # written outside any transaction, so that no database transaction waits on its fsync.
        with self._managed_storage_lock:
            duplicate, used_bytes = self._queries.fonts.upload_state(
                owner_id=owner_id, content_hash=validation.content_hash
            )
            if duplicate is None:
                if used_bytes + validation.byte_size > self._max_library_bytes:
                    raise FontQuotaExceededError("Font library storage quota exceeded")
                storage_path = f"{font_id}{Path(filename).suffix.lower()}"
                try:
                    self._write_atomic(_safe_relative_path(Path(storage_path), self._storage_dir), data)
                    self._queries.fonts.insert_upload(
                        UploadedFont(
                            id=font_id,
                            owner_id=owner_id,
                            scope=scope,
                            filename=validation.filename,
                            storage_path=storage_path,
                            family=validation.family,
                            label=validation.label,
                            style=validation.style,
                            weight=validation.weight,
                            content_hash=validation.content_hash,
                            byte_size=validation.byte_size,
                            axes=validation.axes,
                            instances=validation.instances,
                        )
                    )
                except Exception:
                    # A lost connection can fail a commit that took effect: keep the file of a row that exists, or
                    # that cannot be checked; the startup sweep removes it if no row names it.
                    try:
                        indexed = self._queries.fonts.get(font_id) is not None
                    except Exception:
                        indexed = True
                    if not indexed:
                        self._unlink_managed_file(storage_path)
                    raise

        if duplicate is not None:
            return FontUploadResult(font=duplicate, created=False)

        self._bump_revision()
        record = self.get(font_id)
        if record is None:
            raise FontStorageError("Uploaded font was not indexed")
        return FontUploadResult(font=record, created=True)

    def rescan_directory(self) -> int:
        """Index valid files in the configured directory and remove stale directory rows."""
        with self._rescan_lock:
            return self._rescan_directory()

    def _rescan_directory(self) -> int:
        root = self._fonts_dir
        if not root.exists() or not root.is_dir() or root.is_symlink():
            if root.is_symlink():
                self._logger.warning("Skipping custom fonts directory %s: symlinks are not supported", root)
            if self._queries.fonts.delete_directory_fonts():
                self._bump_revision()
            return 0

        existing_records = {
            record.source_path: record
            for record in self._queries.fonts.directory_fonts()
            if record.source_path is not None
        }

        discovered: list[DirectoryFont] = []
        for path in sorted(root.rglob("*")):
            if self._path_has_symlink_component(path, root):
                self._logger.warning("Skipping font path %s: symlinks are not supported", path)
                continue
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            if len(relative) > MAX_SOURCE_PATH_LENGTH:
                self._logger.warning("Skipping font file %s: its path is longer than the catalog holds", path)
                continue
            try:
                data = _read_bounded_file(path, MAX_INDEXED_FONT_BYTES)
                content_hash = hashlib.sha256(data).hexdigest()
                existing = existing_records.get(relative)
                if existing is not None and existing.byte_size == len(data) and existing.content_hash == content_hash:
                    metadata = _validation_from_record(existing, path.name)
                else:
                    metadata = self._run_worker(
                        "parse",
                        (path.name, data, MAX_INDEXED_FONT_BYTES),
                        error_type=FontValidationError,
                    )
            except (OSError, FontValidationError) as e:
                self._logger.warning("Skipping font file %s: %s", path, e)
                continue
            # The path-derived ID lets a project distinguish two same-named files and remains
            # stable when a directory file is replaced. Its content hash detects that replacement.
            discovered.append(
                DirectoryFont(
                    id=f"directory_{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:32]}",
                    filename=metadata.filename,
                    source_path=relative,
                    family=metadata.family,
                    label=metadata.label,
                    style=metadata.style,
                    weight=metadata.weight,
                    content_hash=metadata.content_hash,
                    byte_size=metadata.byte_size,
                    axes=metadata.axes,
                    instances=metadata.instances,
                )
            )

        existing_by_id = {record.id: record for record in existing_records.values()}
        changed_fonts = [font for font in discovered if not _indexed_as(existing_by_id.get(font.id), font)]
        discovered_ids = frozenset(font.id for font in discovered)
        if changed_fonts or set(existing_by_id) != discovered_ids:
            self._queries.fonts.replace_directory_fonts(changed_fonts, discovered_ids)
            self._invalidate_instance_cache()
            self._bump_revision()
        return len(discovered)

    def list(
        self,
        *,
        user_id: str,
        offset: int = 0,
        limit: int = 100,
        search: str | None = None,
        scope: FontScope = FontScope.ALL,
        content_hash: str | None = None,
    ) -> tuple[list[FontRecord], int]:
        """List fonts visible to an account with bounded offset pagination."""
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        if scope not in (FontScope.ALL, FontScope.PRIVATE, FontScope.SHARED):
            raise ValueError("Invalid font scope")

        if content_hash is not None and re.fullmatch(r"[a-f0-9]{64}", content_hash) is None:
            raise ValueError("content_hash must be a lowercase SHA-256 hash")
        return self._queries.fonts.page(
            user_id=user_id, scope=scope, content_hash=content_hash, search=search, offset=offset, limit=limit
        )

    def get(self, font_id: str) -> FontRecord | None:
        """Get an indexed font without applying account visibility."""
        return self._queries.fonts.get(font_id)

    def get_accessible(self, *, user_id: str, font_id: str) -> FontRecord:
        """Get a font visible to the account, hiding private-resource existence."""
        record = self.get(font_id)
        if record is None:
            raise FontNotFoundError("Font not found")
        if record.source == FontSource.DIRECTORY or record.scope == FontScope.SHARED or record.owner_id == user_id:
            return record
        raise FontNotFoundError("Font not found")

    def read_file(self, *, user_id: str, font_id: str, expected_hash: str | None = None) -> tuple[FontRecord, bytes]:
        """Read an accessible font and verify its immutable content hash."""
        record = self.get_accessible(user_id=user_id, font_id=font_id)
        if expected_hash is not None and expected_hash != record.content_hash:
            raise FontChangedError("Font content has changed")
        if record.source == FontSource.UPLOADED:
            if record.storage_path is None:
                raise FontStorageError("Managed font has no storage path")
            path = _safe_relative_path(Path(record.storage_path), self._storage_dir)
        else:
            if record.source_path is None:
                raise FontStorageError("Directory font has no source path")
            path = _safe_relative_path(Path(record.source_path), self._fonts_dir)
        try:
            data = _read_bounded_file(path, max(self._max_upload_bytes, MAX_INDEXED_FONT_BYTES))
        except OSError as e:
            raise FontNotFoundError("Font file not found") from e
        except FontValidationError as e:
            raise FontChangedError("Font content has changed; refresh the font library") from e
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != record.content_hash:
            raise FontChangedError("Font content has changed; refresh the font library")
        return record, data

    def delete(self, *, user_id: str, font_id: str, is_admin: bool = False) -> None:
        """Delete an owned upload and best-effort clean its managed file."""
        record = self.get_accessible(user_id=user_id, font_id=font_id)
        if record.source != FontSource.UPLOADED:
            raise FontDeleteForbiddenError("Configured directory fonts cannot be deleted")
        if record.scope == FontScope.PRIVATE and record.owner_id != user_id:
            raise FontForbiddenError("Only the font owner may delete a private font")
        if record.scope == FontScope.SHARED and not is_admin:
            raise FontForbiddenError("Only administrators may delete a shared font")

        if not self._queries.fonts.delete_upload(font_id):
            raise FontNotFoundError("Font not found")
        # A variable-font instance is derived from the source bytes. Drop every derived
        # value before unlinking the source so a later request cannot serve a deleted font
        # from memory while the filesystem cleanup is in progress.
        self._invalidate_instance_cache()
        if record.storage_path is not None:
            self._unlink_managed_file(record.storage_path)
        self._bump_revision()

    def cleanup_orphaned_files(self) -> None:
        """Remove interrupted-upload files that are no longer referenced by the catalog."""
        with self._managed_storage_lock:
            # Account deletion cascades font rows in SQLite, but it cannot remove files.
            # Clearing derived instances here also covers that deletion path and ensures
            # an in-flight worker cannot repopulate the cache without re-checking access.
            self._invalidate_instance_cache()
            if not self._storage_dir.exists() or not self._storage_dir.is_dir() or self._storage_dir.is_symlink():
                return
            referenced = self._queries.fonts.storage_paths()
            for path in self._storage_dir.iterdir():
                if not path.is_file() or path.is_symlink():
                    continue
                # Managed uploads are flat files whose names begin with the opaque `font_` ID;
                # leave unrelated operator files alone in case a shared storage directory is used.
                if (
                    re.fullmatch(r"font_[0-9a-f]{32}\.(?:ttf|otf|woff2?)", path.name) is None
                    and re.fullmatch(r"\.font_[0-9a-f]{32}\.(?:ttf|otf|woff2?)\.[0-9a-f]{32}\.tmp", path.name) is None
                ):
                    continue
                if path.name in referenced:
                    continue
                try:
                    path.unlink()
                except OSError:
                    self._logger.warning("Unable to clean up orphaned font file %s", path, exc_info=True)

    def prepare_user_cleanup(self, user_id: str) -> tuple[str, ...]:
        """Capture a user's private managed paths before its account row is deleted."""
        with self._managed_storage_lock:
            storage_paths = self._queries.fonts.private_storage_paths(user_id)
            # A caller may delete the account through a separate service, which cascades
            # rows without notifying this process. Purge derived values before that race.
            self._invalidate_instance_cache()
            return storage_paths

    def cleanup_user(self, user_id: str, storage_paths: tuple[str, ...]) -> None:
        """Remove only private files captured before an account deletion.

        Shared uploads have no account owner and are deliberately absent from ``storage_paths``.
        The startup orphan sweep remains responsible for files from interrupted uploads.
        """
        self._logger.debug("Cleaning up fonts after deleting user %s", user_id)
        self._invalidate_instance_cache()
        for storage_path in storage_paths:
            self._unlink_managed_file(storage_path)

    def instance(
        self,
        *,
        user_id: str,
        font_id: str,
        content_hash: str,
        coordinates: dict[str, float],
    ) -> tuple[FontRecord, bytes]:
        """Create a pinned static instance of an accessible variable font."""
        record, data = self.read_file(user_id=user_id, font_id=font_id, expected_hash=content_hash)
        normalized = self._normalize_coordinates(record, coordinates)
        if not record.axes:
            if normalized:
                raise FontInstanceError("Font does not contain variation axes")
            return record, data

        key = (record.content_hash, tuple(sorted(normalized.items())))
        with self._cache_lock:
            self._assert_accessible_instance_source(user_id=user_id, font_id=font_id, content_hash=record.content_hash)
            cached = self._instance_cache.get(key)
            if cached is not None:
                self._instance_cache.move_to_end(key)
                return record, cached

        instance_data = self._run_worker(
            "instance",
            (data, record.filename, normalized),
            error_type=FontInstanceError,
        )

        with self._cache_lock:
            # The source can be deleted while FontTools is running in the child process.
            # Hold the cache lock across this check and insertion so deletion cannot commit,
            # invalidate, and then race us into retaining a result for an inaccessible font.
            self._assert_accessible_instance_source(user_id=user_id, font_id=font_id, content_hash=record.content_hash)
            if len(instance_data) <= self._max_instance_cache_bytes:
                previous = self._instance_cache.pop(key, None)
                if previous is not None:
                    self._instance_cache_bytes -= len(previous)
                self._instance_cache[key] = instance_data
                self._instance_cache_bytes += len(instance_data)
                while self._instance_cache_bytes > self._max_instance_cache_bytes and self._instance_cache:
                    _old_key, old_data = self._instance_cache.popitem(last=False)
                    self._instance_cache_bytes -= len(old_data)
        return record, instance_data

    def _assert_accessible_instance_source(self, *, user_id: str, font_id: str, content_hash: str) -> None:
        """Reject a worker result when its source was deleted, replaced, or hidden."""
        current = self.get_accessible(user_id=user_id, font_id=font_id)
        if current.content_hash != content_hash:
            raise FontChangedError("Font content has changed; refresh the font library")

    def _invalidate_instance_cache(self) -> None:
        """Drop all derived instances after a source catalog mutation."""
        with self._cache_lock:
            self._instance_cache.clear()
            self._instance_cache_bytes = 0

    def _normalize_coordinates(self, record: FontRecord, coordinates: dict[str, float]) -> dict[str, float]:
        known_axes = {axis.tag: axis for axis in record.axes}
        unknown = set(coordinates) - set(known_axes)
        if unknown:
            raise FontInstanceError(f"Unknown variation axis: {sorted(unknown)[0]}")
        normalized: dict[str, float] = {}
        for axis in record.axes:
            raw = coordinates.get(axis.tag, axis.default)
            try:
                value = float(raw)
            except (TypeError, ValueError) as e:
                raise FontInstanceError(f"Invalid value for variation axis {axis.tag}") from e
            if not math.isfinite(value) or value < axis.minimum or value > axis.maximum:
                raise FontInstanceError(
                    f"Variation axis {axis.tag} must be between {axis.minimum:g} and {axis.maximum:g}"
                )
            normalized[axis.tag] = value
        return normalized

    def _run_worker(self, operation: str, args: tuple[Any, ...], *, error_type: type[FontServiceError]) -> Any:
        """Run a fontTools operation in a killable, bounded child process."""
        if not self._worker_admission.acquire(blocking=False):
            raise error_type("Too many font processing jobs are queued")
        if not self._worker_slots.acquire(timeout=self._instance_job_timeout_seconds):
            self._worker_admission.release()
            raise error_type("Too many font processing jobs are queued")
        if operation == "parse":
            filename, data, max_bytes = args
            command = [
                sys.executable,
                "-m",
                "invokeai.app.services.fonts.fonts_worker",
                "parse",
                str(filename),
                str(max_bytes),
            ]
            input_data = data
        elif operation == "instance":
            data, filename, coordinates = args
            command = [
                sys.executable,
                "-m",
                "invokeai.app.services.fonts.fonts_worker",
                "instance",
                str(filename),
                json.dumps(coordinates, separators=(",", ":"), sort_keys=True),
            ]
            input_data = data
        else:
            self._worker_slots.release()
            self._worker_admission.release()
            raise error_type("Unknown font processing operation")

        process: subprocess.Popen[bytes] | None = None
        try:
            with self._active_workers_lock:
                process = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                self._active_workers.add(process)
            try:
                stdout, _ = process.communicate(input=input_data, timeout=self._instance_job_timeout_seconds)
            except subprocess.TimeoutExpired as e:
                process.kill()
                process.communicate()
                raise error_type("Font processing timed out") from e
            if process.returncode != 0 or not stdout:
                raise error_type("Font processing failed")
            try:
                result = pickle.loads(stdout)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError("invalid worker response")
                result_kind, payload = result
            except (EOFError, ValueError, pickle.PickleError, TypeError, AttributeError, ImportError) as e:
                raise error_type("Font processing returned invalid data") from e
            if result_kind == "error":
                raise error_type(str(payload))
            if result_kind != "ok":
                raise error_type("Font processing returned an unknown result")
            return payload
        finally:
            if process is not None:
                with self._active_workers_lock:
                    self._active_workers.discard(process)
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            self._worker_slots.release()
            self._worker_admission.release()

    @staticmethod
    def _write_atomic(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
        except Exception:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise

    def _unlink_managed_file(self, storage_path: str) -> None:
        try:
            path = _safe_relative_path(Path(storage_path), self._storage_dir)
            path.unlink(missing_ok=True)
        except (OSError, FontStorageError):
            self._logger.warning("Unable to clean up managed font file %s", storage_path, exc_info=True)

    @staticmethod
    def _path_has_symlink_component(path: Path, boundary: Path) -> bool:
        current = path
        boundary = boundary.absolute()
        try:
            current.absolute().relative_to(boundary)
        except ValueError:
            return True
        while True:
            if current.is_symlink():
                return True
            if current == boundary:
                return False
            current = current.parent


def _instantiate_font_bytes(data: bytes, filename: str, coordinates: dict[str, float]) -> bytes:
    """Worker function for variable-font instancing."""
    source = io.BytesIO(data)
    with TTFont(file=source, lazy=False) as font:
        if "fvar" not in font:
            return data
        flavor = font.flavor
        # fontTools calls the coordinate mapping ``axisLimits``: values outside the
        # declared ranges are rejected by the service before this worker is submitted.
        instantiated = instantiateVariableFont(font, axisLimits=coordinates, inplace=False)
        instantiated.flavor = flavor
        output = io.BytesIO()
        try:
            instantiated.save(output)
            return output.getvalue()
        finally:
            instantiated.close()


__all__ = [
    "FontChangedError",
    "FontDeleteForbiddenError",
    "FontForbiddenError",
    "FontInstanceError",
    "FontNotFoundError",
    "FontQuotaExceededError",
    "FontService",
    "FontStorageError",
    "FontValidationError",
    "SUPPORTED_FONT_EXTENSIONS",
]
