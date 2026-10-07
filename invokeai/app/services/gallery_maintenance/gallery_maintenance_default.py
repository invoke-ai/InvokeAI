"""Application-owned image gallery maintenance operations."""

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from invokeai.app.services.gallery_maintenance.gallery_maintenance_common import (
    GalleryMaintenanceConflict,
    GalleryMaintenanceError,
    GalleryMaintenanceOperation,
    GalleryMaintenancePreview,
    GalleryMaintenancePreviewChanged,
    GalleryMaintenanceResult,
)
from invokeai.app.services.image_moves.image_moves_default import ImageMoveJobAlreadyRunning, ImageMoveQueueActive
from invokeai.app.util.thumbnails import get_thumbnail_name
from invokeai.backend.util.logging import InvokeAILogger

if TYPE_CHECKING:
    from invokeai.app.services.invoker import Invoker

_ENUMERATION_BATCH_SIZE = 250
_MAX_RESULT_ERRORS = 20
_MAX_ERROR_LENGTH = 300
_OperationStatus = Literal["completed", "partial", "no_op", "failed"]
_StatSignature = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class _Scan:
    fingerprint: str
    examined_count: int
    affected_count: int
    skipped_count: int


@dataclass
class _ResultCounts:
    examined_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    records_removed: int = 0
    images_archived: int = 0
    thumbnails_archived: int = 0
    thumbnails_regenerated: int = 0
    errors: list[str] = field(default_factory=list)
    archive_path: str | None = None
    backup_path: str | None = None

    def add_error(self, description: str) -> None:
        if len(self.errors) < _MAX_RESULT_ERRORS:
            self.errors.append(description[:_MAX_ERROR_LENGTH])

    def result(self, operation: GalleryMaintenanceOperation) -> GalleryMaintenanceResult:
        affected = self.records_removed + self.images_archived + self.thumbnails_archived + self.thumbnails_regenerated
        if self.failed_count == 0 and self.skipped_count == 0 and affected == 0:
            status: _OperationStatus = "no_op"
        elif self.failed_count or self.skipped_count:
            status = "partial"
        else:
            status = "completed"
        if self.failed_count and affected == 0:
            status = "failed"

        return GalleryMaintenanceResult(
            operation=operation,
            status=status,
            examined_count=self.examined_count,
            skipped_count=self.skipped_count,
            failed_count=self.failed_count,
            records_removed=self.records_removed,
            images_archived=self.images_archived,
            thumbnails_archived=self.thumbnails_archived,
            thumbnails_regenerated=self.thumbnails_regenerated,
            archive_path=self.archive_path,
            backup_path=self.backup_path,
            errors=self.errors,
        )


class GalleryMaintenanceService:
    """Coordinates installation-wide maintenance through the active application services."""

    def start(self, invoker: "Invoker") -> None:
        self._invoker = invoker
        self._logger = InvokeAILogger.get_logger(self.__class__.__name__)

    def preview(self, operation: GalleryMaintenanceOperation) -> GalleryMaintenancePreview:
        operation = self._normalize_operation(operation)
        with self._reserve():
            with self._inventory() as inventory:
                scan = self._scan(operation, inventory)
        return GalleryMaintenancePreview(
            operation=operation,
            fingerprint=scan.fingerprint,
            examined_count=scan.examined_count,
            affected_count=scan.affected_count,
            skipped_count=scan.skipped_count,
            error_count=0,
            errors=[],
            archive_path=str(self._archive_root) if self._archives(operation) else None,
        )

    def execute(self, operation: GalleryMaintenanceOperation, expected_fingerprint: str) -> GalleryMaintenanceResult:
        operation = self._normalize_operation(operation)
        if len(expected_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in expected_fingerprint
        ):
            raise GalleryMaintenanceError("The preview fingerprint is invalid; preview this operation again.")

        with self._reserve():
            with self._inventory() as inventory:
                scan = self._scan(operation, inventory)
                if scan.fingerprint != expected_fingerprint:
                    raise GalleryMaintenancePreviewChanged(
                        "The gallery changed after preview. Refresh the preview and confirm again."
                    )

                counts = _ResultCounts(
                    examined_count=scan.examined_count,
                    skipped_count=scan.skipped_count,
                    archive_path=str(self._archive_root) if self._archives(operation) else None,
                )
                if scan.affected_count == 0:
                    return counts.result(operation)

                counts.backup_path = self._create_backup()
                if self._archives(operation):
                    try:
                        counts.archive_path = str(self._create_archive_run(operation))
                    except GalleryMaintenanceError:
                        raise
                    except Exception as e:
                        self._logger.exception("Could not prepare gallery archive storage")
                        raise GalleryMaintenanceError("Could not safely prepare archive storage.") from e

                try:
                    if operation is GalleryMaintenanceOperation.REMOVE_MISSING:
                        self._remove_missing(inventory, counts)
                    elif operation is GalleryMaintenanceOperation.ARCHIVE_UNTRACKED:
                        self._archive_untracked(inventory, counts)
                    else:
                        self._regenerate_thumbnails(inventory, counts)
                except Exception:
                    self._logger.exception("Gallery maintenance stopped after partial changes")
                    counts.failed_count += 1
                    counts.add_error("Gallery maintenance stopped after a storage or database error.")
                return counts.result(operation)

    def _normalize_operation(self, operation: GalleryMaintenanceOperation) -> GalleryMaintenanceOperation:
        try:
            return GalleryMaintenanceOperation(operation)
        except ValueError as e:
            raise GalleryMaintenanceError("Unsupported gallery maintenance operation.") from e

    @contextmanager
    def _reserve(self) -> Iterator[None]:
        image_moves = getattr(self._invoker.services, "image_moves", None)
        if image_moves is None:
            raise GalleryMaintenanceError("Image storage maintenance is unavailable.")
        try:
            with image_moves.reserve_gallery_maintenance():
                yield
        except (ImageMoveJobAlreadyRunning, ImageMoveQueueActive) as e:
            raise GalleryMaintenanceConflict("Image storage or queue work is active.") from e

    @property
    def _services(self):
        return self._invoker.services

    @property
    def _archive_root(self) -> Path:
        return self._services.image_files.image_root.parent / "images-archive"

    @staticmethod
    def _archives(operation: GalleryMaintenanceOperation) -> bool:
        return operation in {
            GalleryMaintenanceOperation.REMOVE_MISSING,
            GalleryMaintenanceOperation.ARCHIVE_UNTRACKED,
        }

    @contextmanager
    def _inventory(self) -> Iterator[sqlite3.Connection]:
        """Spools the confirmed inventory to disk so execution never materializes the gallery in RAM."""
        with tempfile.TemporaryDirectory(prefix="invokeai-gallery-maintenance-") as directory:
            connection = sqlite3.connect(Path(directory) / "inventory.sqlite")
            try:
                connection.executescript(
                    """
                    CREATE TABLE records (
                        image_name TEXT PRIMARY KEY,
                        subfolder TEXT NOT NULL,
                        original_signature TEXT,
                        thumbnail_signature TEXT,
                        action INTEGER NOT NULL DEFAULT 0,
                        duplicate INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE TABLE files (
                        kind TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        base_name TEXT NOT NULL,
                        subfolder TEXT NOT NULL,
                        signature TEXT,
                        action INTEGER NOT NULL,
                        PRIMARY KEY(kind, relative_path)
                    );
                    CREATE INDEX files_by_name ON files(kind, base_name);
                    """
                )
                yield connection
            finally:
                connection.close()

    def _scan(self, operation: GalleryMaintenanceOperation, inventory: sqlite3.Connection) -> _Scan:
        digest = hashlib.sha256()
        try:
            root_signatures = self._validate_storage_roots()
            roots = self._fingerprint_roots(operation)
            self._feed(digest, "gallery-maintenance-v2", operation.value, *roots, *root_signatures)
            if operation is GalleryMaintenanceOperation.REMOVE_MISSING:
                return self._scan_remove_missing(digest, inventory)
            if operation is GalleryMaintenanceOperation.ARCHIVE_UNTRACKED:
                return self._scan_archive_untracked(digest, inventory)
            return self._scan_regenerate_thumbnails(digest, inventory)
        except GalleryMaintenanceError:
            raise
        except Exception as e:
            self._logger.exception("Gallery maintenance scan failed for %s", operation.value)
            raise GalleryMaintenanceError("Could not safely inspect gallery storage.") from e

    def _validate_storage_roots(self) -> tuple[str, str]:
        signatures: list[str] = []
        for root in (self._services.image_files.image_root, self._services.image_files.thumbnail_root):
            try:
                result = root.lstat()
                if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
                    raise GalleryMaintenanceError("A required gallery storage root is unavailable.")
                # Opening the directory catches unreadable roots even when the operation has no
                # file paths to enumerate (for example, an empty gallery).
                with os.scandir(root) as entries:
                    next(entries, None)
            except GalleryMaintenanceError:
                raise
            except OSError as e:
                self._logger.warning("Could not inspect gallery storage root (%s)", e.errno)
                raise GalleryMaintenanceError("A required gallery storage root is unavailable.") from e
            signatures.append(json.dumps(self._signature(result), separators=(",", ":")))
        return signatures[0], signatures[1]

    def _fingerprint_roots(self, operation: GalleryMaintenanceOperation) -> tuple[str, ...]:
        files = self._services.image_files
        roots = [str(files.image_root.resolve()), str(files.thumbnail_root.resolve())]
        if self._archives(operation):
            roots.append(str(self._archive_root.resolve()))
        return tuple(roots)

    @staticmethod
    def _feed(digest, *parts: object) -> None:
        digest.update(json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")

    def _scan_remove_missing(self, digest, inventory: sqlite3.Connection) -> _Scan:
        examined = affected = skipped = 0
        for image_name, subfolder in self._services.image_records.iter_all_image_locations(_ENUMERATION_BATCH_SIZE):
            examined += 1
            self._feed(digest, "record", image_name, subfolder)
            original = self._record_path(image_name, subfolder)
            thumbnail = self._record_path(image_name, subfolder, thumbnail=True)
            original_stat = self._stat(original)
            thumbnail_stat = self._stat(thumbnail)
            self._feed_stat(digest, "original", image_name, subfolder, original_stat)
            self._feed_stat(digest, "thumbnail", image_name, subfolder, thumbnail_stat)
            inventory.execute(
                "INSERT INTO records(image_name, subfolder, original_signature, thumbnail_signature, action) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    image_name,
                    subfolder,
                    self._encode_signature(original_stat),
                    self._encode_signature(thumbnail_stat),
                    int(original_stat is None),
                ),
            )

        for paths in self._batches(self._services.image_files.iter_image_paths()):
            for path in paths:
                relative = self._relative(path, self._services.image_files.image_root)
                subfolder = self._relative_parent(path, self._services.image_files.image_root)
                signature = self._stat(path)
                self._feed_stat(digest, "image-file", relative, signature)
                inventory.execute(
                    "INSERT INTO files(kind, relative_path, base_name, subfolder, signature, action) "
                    "VALUES ('image', ?, ?, ?, ?, 0)",
                    (relative, path.name, subfolder, self._encode_signature(signature)),
                )
                inventory.execute(
                    "UPDATE records SET duplicate = 1 WHERE image_name = ? AND subfolder != ?",
                    (path.name, subfolder),
                )

        inventory.commit()
        for (is_duplicate,) in inventory.execute("SELECT duplicate FROM records WHERE action = 1 ORDER BY image_name"):
            if is_duplicate:
                skipped += 1
            else:
                affected += 1
        inventory.execute("UPDATE records SET action = 0 WHERE duplicate = 1")
        return _Scan(digest.hexdigest(), examined, affected, skipped)

    @staticmethod
    def _encode_signature(signature: _StatSignature | None) -> str | None:
        return None if signature is None else json.dumps(signature, separators=(",", ":"))

    @staticmethod
    def _decode_signature(signature: str | None) -> _StatSignature | None:
        return None if signature is None else tuple(json.loads(signature))  # type: ignore[return-value]

    def _scan_archive_untracked(self, digest, inventory: sqlite3.Connection) -> _Scan:
        examined = affected = skipped = 0
        self._feed_records(digest, inventory)

        for paths in self._batches(self._services.image_files.iter_image_paths()):
            locations = self._services.image_records.get_subfolders([path.name for path in paths])
            for path in paths:
                examined += 1
                subfolder = self._relative_parent(path, self._services.image_files.image_root)
                signature = self._stat(path)
                self._feed_stat(
                    digest, "image-file", self._relative(path, self._services.image_files.image_root), signature
                )
                stored_subfolder = locations.get(path.name)
                if stored_subfolder is None:
                    affected += 1
                elif stored_subfolder != subfolder:
                    skipped += 1
                inventory.execute(
                    "INSERT INTO files(kind, relative_path, base_name, subfolder, signature, action) "
                    "VALUES ('image', ?, ?, ?, ?, ?)",
                    (
                        self._relative(path, self._services.image_files.image_root),
                        path.name,
                        subfolder,
                        self._encode_signature(signature),
                        int(stored_subfolder is None),
                    ),
                )

        for paths in self._batches(self._services.image_files.iter_thumbnail_paths()):
            names = [f"{path.stem}.png" for path in paths]
            locations = self._services.image_records.get_subfolders(names)
            for path in paths:
                examined += 1
                subfolder = self._relative_parent(path, self._services.image_files.thumbnail_root)
                signature = self._stat(path)
                self._feed_stat(
                    digest,
                    "thumbnail-file",
                    self._relative(path, self._services.image_files.thumbnail_root),
                    signature,
                )
                source_name = f"{path.stem}.png"
                stored_subfolder = locations.get(source_name)
                action = False
                if stored_subfolder is not None:
                    if stored_subfolder != subfolder:
                        skipped += 1
                    # A thumbnail still associated with a record is recoverable even if its
                    # original is missing; remove-missing owns that record and thumbnail pair.
                    continue
                original = self._record_path(source_name, subfolder)
                if self._stat(original) is None:
                    affected += 1
                    action = True
                else:
                    original_relative = (Path(subfolder) / source_name).as_posix()
                    row = inventory.execute(
                        "SELECT action FROM files WHERE kind = 'image' AND relative_path = ?",
                        (original_relative,),
                    ).fetchone()
                    action = row is not None and bool(row[0])
                inventory.execute(
                    "INSERT INTO files(kind, relative_path, base_name, subfolder, signature, action) "
                    "VALUES ('thumbnail', ?, ?, ?, ?, ?)",
                    (
                        self._relative(path, self._services.image_files.thumbnail_root),
                        path.name,
                        subfolder,
                        self._encode_signature(signature),
                        int(action),
                    ),
                )

        inventory.commit()
        return _Scan(digest.hexdigest(), examined, affected, skipped)

    def _scan_regenerate_thumbnails(self, digest, inventory: sqlite3.Connection) -> _Scan:
        examined = affected = skipped = 0
        for image_name, subfolder in self._services.image_records.iter_all_image_locations(_ENUMERATION_BATCH_SIZE):
            examined += 1
            self._feed(digest, "record", image_name, subfolder)
            original = self._record_path(image_name, subfolder)
            thumbnail = self._record_path(image_name, subfolder, thumbnail=True)
            original_stat = self._stat(original)
            thumbnail_stat = self._stat(thumbnail)
            self._feed_stat(digest, "original", image_name, subfolder, original_stat)
            self._feed_stat(digest, "thumbnail", image_name, subfolder, thumbnail_stat)
            if original_stat is None:
                skipped += 1
            elif thumbnail_stat is None:
                affected += 1
            inventory.execute(
                "INSERT INTO records(image_name, subfolder, original_signature, thumbnail_signature, action) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    image_name,
                    subfolder,
                    self._encode_signature(original_stat),
                    self._encode_signature(thumbnail_stat),
                    int(original_stat is not None and thumbnail_stat is None),
                ),
            )
        inventory.commit()
        return _Scan(digest.hexdigest(), examined, affected, skipped)

    def _feed_records(self, digest, inventory: sqlite3.Connection) -> None:
        for image_name, subfolder in self._services.image_records.iter_all_image_locations(_ENUMERATION_BATCH_SIZE):
            self._feed(digest, "record", image_name, subfolder)
            inventory.execute(
                "INSERT INTO records(image_name, subfolder, action) VALUES (?, ?, 0)", (image_name, subfolder)
            )

    @staticmethod
    def _batches(paths: Iterator[Path]) -> Iterator[list[Path]]:
        batch: list[Path] = []
        for path in paths:
            batch.append(path)
            if len(batch) == _ENUMERATION_BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch

    def _record_path(self, image_name: str, subfolder: str, thumbnail: bool = False) -> Path:
        files = self._services.image_files
        root = files.thumbnail_root if thumbnail else files.image_root
        filename = get_thumbnail_name(image_name) if thumbnail else image_name
        relative = Path(subfolder) / filename
        if relative.is_absolute() or ".." in relative.parts:
            raise GalleryMaintenanceError("An image record contains an unsafe storage path.")

        current = root
        for part in relative.parts:
            current /= part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                break
            except OSError as e:
                raise GalleryMaintenanceError("Could not inspect an image storage path.") from e
            if stat.S_ISLNK(mode):
                raise GalleryMaintenanceError("Gallery maintenance does not follow symbolic links.")

        try:
            path = files.get_path(image_name, thumbnail=thumbnail, image_subfolder=subfolder)
        except (ValueError, OSError) as e:
            self._logger.error("Invalid recorded image path for %s", image_name)
            raise GalleryMaintenanceError("An image record contains an unsafe storage path.") from e
        return path

    def _stat(self, path: Path) -> _StatSignature | None:
        try:
            result = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as e:
            self._logger.warning("Could not inspect gallery path (%s)", e.errno)
            raise GalleryMaintenanceError("Could not safely inspect gallery storage.") from e
        if not stat.S_ISREG(result.st_mode):
            raise GalleryMaintenanceError("Gallery storage contains a non-file at an image path.")
        return self._signature(result)

    @staticmethod
    def _signature(result: os.stat_result) -> _StatSignature:
        return (
            result.st_dev,
            result.st_ino,
            result.st_mode,
            result.st_size,
            result.st_mtime_ns,
            result.st_ctime_ns,
        )

    def _feed_stat(self, digest, *parts: object) -> None:
        self._feed(digest, *parts)

    @staticmethod
    def _relative(path: Path, root: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError as e:
            raise GalleryMaintenanceError("Gallery file escaped its configured storage root.") from e

    def _relative_parent(self, path: Path, root: Path) -> str:
        parent = path.parent
        relative = self._relative(parent, root)
        return "" if relative == "." else relative

    def _create_backup(self) -> str:
        backup_dir = self._services.configuration.db_path.parent / "backup"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = backup_dir / f"backup-{timestamp}-gallery-maintenance-{uuid.uuid4().hex}.db"
        try:
            self._services.database.backup_to(destination)
        except Exception as e:
            self._logger.exception("Gallery maintenance database backup failed")
            raise GalleryMaintenanceError("The database backup failed; no gallery changes were made.") from e
        return str(destination)

    def _create_archive_run(self, operation: GalleryMaintenanceOperation) -> Path:
        root = self._archive_root
        self._assert_archive_root(root)
        run_parent = root / "gallery-maintenance"
        self._assert_archive_root(run_parent)
        run_parent.mkdir(parents=True, exist_ok=True)
        self._assert_archive_root(run_parent)
        run_path = run_parent / f"{operation.value}-{uuid.uuid4().hex}"
        run_path.mkdir(exist_ok=False)
        self._assert_archive_root(run_path)
        return run_path

    def _assert_archive_root(self, path: Path) -> None:
        output_root = self._services.image_files.image_root.parent.resolve()
        current = output_root
        try:
            relative = path.relative_to(output_root)
        except ValueError as e:
            raise GalleryMaintenanceError("Archive path escaped the configured output directory.") from e
        for part in relative.parts:
            current /= part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                continue
            except OSError as e:
                raise GalleryMaintenanceError("Could not inspect the configured archive directory.") from e
            if stat.S_ISLNK(mode):
                raise GalleryMaintenanceError("Gallery maintenance does not follow archive symlinks.")

    def _remove_missing(self, inventory: sqlite3.Connection, counts: _ResultCounts) -> None:
        archive_run = Path(counts.archive_path) if counts.archive_path else None
        cursor = inventory.execute(
            "SELECT image_name, subfolder, thumbnail_signature FROM records WHERE action = 1 ORDER BY image_name"
        )
        while page := cursor.fetchmany(_ENUMERATION_BATCH_SIZE):
            self._remove_missing_page(page, archive_run, counts)

    def _remove_missing_page(
        self, page: list[tuple[str, str, str | None]], archive_run: Path | None, counts: _ResultCounts
    ) -> None:
        files = self._services.image_files
        records = self._services.image_records
        locations = records.get_subfolders([image_name for image_name, _subfolder, _thumb in page])
        delete_items: list[tuple[str, str, Path, Path, _StatSignature | None, Path | None, _StatSignature | None]] = []
        for image_name, subfolder, encoded_thumbnail_signature in page:
            if locations.get(image_name) != subfolder:
                counts.skipped_count += 1
                continue
            original = self._record_path(image_name, subfolder)
            thumbnail = self._record_path(image_name, subfolder, thumbnail=True)
            thumbnail_signature = self._decode_signature(encoded_thumbnail_signature)
            archived_thumbnail: Path | None = None
            archive_counted = False
            try:
                if self._stat(original) is not None or self._stat(thumbnail) != thumbnail_signature:
                    counts.skipped_count += 1
                    continue
                archived_signature: _StatSignature | None = None
                if thumbnail_signature is not None:
                    if archive_run is None:
                        raise GalleryMaintenanceError("Archive destination was not prepared.")
                    archived_thumbnail = archive_run / "thumbnails" / Path(subfolder) / thumbnail.name
                    self._archive_file(thumbnail, archived_thumbnail, thumbnail_signature)
                    archived_signature = self._path_signature(archived_thumbnail)
                    if archived_signature is None:
                        raise OSError("Archived thumbnail disappeared before record removal")
                    counts.thumbnails_archived += 1
                    archive_counted = True
                    files.evict_cache_paths([thumbnail])
                delete_items.append(
                    (
                        image_name,
                        subfolder,
                        original,
                        thumbnail,
                        thumbnail_signature,
                        archived_thumbnail,
                        archived_signature,
                    )
                )
            except Exception as e:
                if archived_thumbnail is not None:
                    try:
                        archived_signature = self._path_signature(archived_thumbnail)
                        source_signature = self._path_signature(thumbnail)
                    except OSError:
                        archived_signature = None
                        source_signature = None
                        counts.add_error("Could not confirm recovery of an archived thumbnail.")
                    if archived_signature is not None and source_signature is None:
                        restored = self._restore_archived_file(archived_thumbnail, thumbnail, counts)
                        if restored:
                            if archive_counted:
                                counts.thumbnails_archived -= 1
                        elif not archive_counted:
                            counts.thumbnails_archived += 1
                    elif archived_signature is None and archive_counted:
                        counts.thumbnails_archived -= 1
                self._record_item_failure(counts, "record", image_name, e)

        if not delete_items:
            return

        still_current: list[tuple[str, str, Path, Path, _StatSignature | None, Path | None, _StatSignature | None]] = []
        for item in delete_items:
            image_name, subfolder, original, thumbnail, thumbnail_signature, archived_thumbnail, archived_signature = (
                item
            )
            try:
                thumbnail_unchanged = (
                    self._path_signature(archived_thumbnail) == archived_signature
                    if archived_thumbnail is not None
                    else self._stat(thumbnail) is None
                )
                if self._stat(original) is None and thumbnail_unchanged:
                    still_current.append(item)
                else:
                    counts.skipped_count += 1
                    self._restore_archived_file(archived_thumbnail, thumbnail, counts)
                    if archived_thumbnail is not None and thumbnail.exists():
                        counts.thumbnails_archived -= 1
            except Exception as e:
                self._record_item_failure(counts, "record", image_name, e)
                self._restore_archived_file(archived_thumbnail, thumbnail, counts)
                if archived_thumbnail is not None and thumbnail.exists():
                    counts.thumbnails_archived -= 1

        if not still_current:
            return
        delete_names = [item[0] for item in still_current]
        try:
            records.delete_many(delete_names)
        except Exception:
            for _name, _subfolder, _original, source, _signature, archived, _archived_signature in reversed(
                still_current
            ):
                self._restore_archived_file(archived, source, counts)
                if archived is not None and source.exists():
                    counts.thumbnails_archived -= 1
            self._logger.exception("Failed to remove missing image records")
            counts.failed_count += len(delete_names)
            counts.add_error("Database record removal failed; remaining records were kept.")
            return

        # delete_many owns one transaction and returns only after its commit; count committed
        # removals before cache, existence, or notification work can fail.
        counts.records_removed += len(delete_names)
        for image_name, _subfolder, original, thumbnail, _signature, _archived, _archived_signature in still_current:
            try:
                files.evict_cache_paths([original, thumbnail])
            except Exception:
                self._logger.exception("Could not evict deleted image cache entries")
                counts.failed_count += 1
                counts.add_error("An image was removed, but its cached file could not be cleared.")
            try:
                still_exists = records.exists(image_name)
                if still_exists:
                    counts.failed_count += 1
                    counts.add_error("An image record remained after the removal transaction.")
            except Exception:
                self._logger.exception("Could not verify image record removal")
                counts.failed_count += 1
                counts.add_error("Could not confirm an image record removal.")
            try:
                self._services.images.notify_deleted(image_name)
            except Exception:
                self._logger.exception("Image deletion notification failed after gallery maintenance")
                counts.failed_count += 1
                counts.add_error("An image was removed, but a dependent service was not notified.")

    def _archive_untracked(self, inventory: sqlite3.Connection, counts: _ResultCounts) -> None:
        archive_run = Path(counts.archive_path) if counts.archive_path else None
        if archive_run is None:
            return
        self._archive_snapshot_files(inventory, "image", archive_run, counts)
        self._archive_snapshot_files(inventory, "thumbnail", archive_run, counts)

    def _archive_snapshot_files(
        self, inventory: sqlite3.Connection, kind: str, archive_run: Path, counts: _ResultCounts
    ) -> None:
        files = self._services.image_files
        records = self._services.image_records
        root = files.image_root if kind == "image" else files.thumbnail_root
        cursor = inventory.execute(
            "SELECT relative_path, base_name, subfolder, signature FROM files "
            "WHERE kind = ? AND action = 1 ORDER BY relative_path",
            (kind,),
        )
        while page := cursor.fetchmany(_ENUMERATION_BATCH_SIZE):
            locations = records.get_subfolders(
                [base_name if kind == "image" else f"{Path(base_name).stem}.png" for _, base_name, _, _ in page]
            )
            for relative_path, base_name, subfolder, encoded_signature in page:
                source_name = base_name if kind == "image" else f"{Path(base_name).stem}.png"
                path = root / Path(relative_path)
                if locations.get(source_name) is not None:
                    counts.skipped_count += 1
                    continue
                try:
                    if kind == "thumbnail":
                        original = self._record_path(source_name, subfolder)
                        if self._stat(original) is not None:
                            # Keep a thumbnail paired with an original that changed or failed to move.
                            counts.skipped_count += 1
                            continue
                    signature = self._decode_signature(encoded_signature)
                    current_signature = self._stat(path)
                    if signature is None or current_signature != signature:
                        counts.skipped_count += 1
                        continue
                    destination = archive_run / ("images" if kind == "image" else "thumbnails")
                    destination = destination / Path(subfolder) / path.name
                    self._archive_file(path, destination, signature)
                    if kind == "image":
                        counts.images_archived += 1
                    else:
                        counts.thumbnails_archived += 1
                    files.evict_cache_paths([path])
                except Exception as e:
                    expected = self._decode_signature(encoded_signature)
                    if (
                        self._path_signature(path) is None
                        and self._path_signature(
                            archive_run / ("images" if kind == "image" else "thumbnails") / Path(subfolder) / path.name
                        )
                        == expected
                    ):
                        if kind == "image":
                            counts.images_archived += 1
                        else:
                            counts.thumbnails_archived += 1
                    self._record_item_failure(counts, kind, path.name, e)

    def _regenerate_thumbnails(self, inventory: sqlite3.Connection, counts: _ResultCounts) -> None:
        files = self._services.image_files
        records = self._services.image_records
        cursor = inventory.execute(
            "SELECT image_name, subfolder, original_signature FROM records WHERE action = 1 ORDER BY image_name"
        )
        while page := cursor.fetchmany(_ENUMERATION_BATCH_SIZE):
            locations = records.get_subfolders([image_name for image_name, _subfolder, _signature in page])
            for image_name, subfolder, encoded_original_signature in page:
                original = self._record_path(image_name, subfolder)
                thumbnail = self._record_path(image_name, subfolder, thumbnail=True)
                generated_thumbnail_signature: _StatSignature | None = None
                try:
                    expected_original = self._decode_signature(encoded_original_signature)
                    if locations.get(image_name) != subfolder:
                        counts.skipped_count += 1
                        continue
                    if self._stat(original) != expected_original or self._stat(thumbnail) is not None:
                        counts.skipped_count += 1
                        continue
                    if expected_original is None:
                        counts.skipped_count += 1
                        continue
                    created = files.generate_thumbnail_if_missing(image_name, image_subfolder=subfolder)
                    if not created:
                        counts.skipped_count += 1
                        continue
                    generated_thumbnail_signature = self._path_signature(thumbnail)
                    if generated_thumbnail_signature is None:
                        raise OSError("Generated thumbnail disappeared before validation")
                    if self._stat(original) != expected_original:
                        raise OSError("An image changed while its thumbnail was regenerated")
                    size = files.get_file_size_bytes(image_name, image_subfolder=subfolder)
                    records.set_file_size_bytes(image_name, size)
                    counts.thumbnails_regenerated += 1
                except Exception:
                    self._logger.exception("Failed to regenerate a gallery thumbnail")
                    if generated_thumbnail_signature is not None:
                        try:
                            self._remove_generated_thumbnail(thumbnail, generated_thumbnail_signature)
                            files.evict_cache_paths([thumbnail])
                        except Exception:
                            self._logger.exception("Could not remove an incomplete gallery thumbnail")
                            counts.add_error("Could not remove an incomplete image thumbnail.")
                    counts.failed_count += 1
                    counts.add_error("Could not regenerate an image thumbnail.")

    @staticmethod
    def _archive_file(source: Path, destination: Path, expected: _StatSignature) -> None:
        """Copy to an exclusive file, publish without overwrite, then identity-check source removal."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        temporary_fd: int | None = None
        quarantine_directory: Path | None = None
        quarantined_source: Path | None = None
        source_quarantined = False
        try:
            temporary_fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with open(source, "rb") as source_file, os.fdopen(temporary_fd, "wb") as destination_file:
                temporary_fd = None
                before = os.fstat(source_file.fileno())
                # Windows may report ctime differently through fstat() and path-based stat().
                # Device, inode, type, size, and mtime remain stable identity/content checks;
                # the full path signature (including ctime) is checked before source removal.
                if GalleryMaintenanceService._signature(before)[:5] != expected[:5]:
                    raise OSError("Source changed after preview validation")
                shutil.copyfileobj(source_file, destination_file, length=1024 * 1024)
                destination_file.flush()
                os.fsync(destination_file.fileno())
                after = os.fstat(source_file.fileno())
                if GalleryMaintenanceService._signature(after)[:5] != expected[:5]:
                    raise OSError("Source changed while archiving")
            os.link(temporary, destination)
            temporary.unlink()
            GalleryMaintenanceService.__fsync_directory(destination.parent)
            if GalleryMaintenanceService._path_signature(source) != expected:
                raise OSError("Source changed before archive completion")
            quarantine_directory = Path(tempfile.mkdtemp(prefix=".gallery-archive-", dir=source.parent))
            GalleryMaintenanceService.__fsync_directory(source.parent)
            quarantined_source = quarantine_directory / source.name
            source.replace(quarantined_source)
            source_quarantined = True
            GalleryMaintenanceService.__fsync_directory(quarantine_directory)
            GalleryMaintenanceService.__fsync_directory(source.parent)
            moved_stat = quarantined_source.lstat()
            moved_signature = GalleryMaintenanceService._signature(moved_stat)
            # Rename may update ctime, so compare identity, type, size, and mtime. The full
            # preview signature was checked immediately before the atomic same-directory move.
            if moved_signature[:5] != expected[:5]:
                raise OSError("Source changed before archive completion")
            quarantined_source.unlink()
            source_quarantined = False
            GalleryMaintenanceService.__fsync_directory(quarantine_directory)
            GalleryMaintenanceService.__fsync_directory(source.parent)
        except Exception:
            if source_quarantined and quarantined_source is not None:
                try:
                    # Link is atomic and refuses to replace a source created by another writer.
                    # If restoration cannot safely claim the original name, leave the quarantined
                    # file in its hidden directory for recovery instead of deleting user data.
                    os.link(quarantined_source, source)
                    quarantined_source.unlink()
                    source_quarantined = False
                    GalleryMaintenanceService.__fsync_directory(quarantine_directory)
                    GalleryMaintenanceService.__fsync_directory(source.parent)
                except OSError:
                    pass
            raise
        finally:
            if temporary_fd is not None:
                os.close(temporary_fd)
            temporary.unlink(missing_ok=True)
            if quarantine_directory is not None:
                try:
                    quarantine_directory.rmdir()
                except OSError:
                    pass

    def _remove_generated_thumbnail(self, thumbnail: Path, expected: _StatSignature) -> None:
        if self._path_signature(thumbnail) != expected:
            return

        quarantine_directory = Path(tempfile.mkdtemp(prefix=".thumbnail-rollback-", dir=thumbnail.parent))
        quarantined_thumbnail = quarantine_directory / thumbnail.name
        thumbnail_quarantined = False
        try:
            self.__fsync_directory(thumbnail.parent)
            thumbnail.replace(quarantined_thumbnail)
            thumbnail_quarantined = True
            self.__fsync_directory(quarantine_directory)
            self.__fsync_directory(thumbnail.parent)
            moved_signature = self._path_signature(quarantined_thumbnail)
            if moved_signature is None or moved_signature[:5] != expected[:5]:
                self._restore_quarantined_thumbnail(quarantined_thumbnail, thumbnail)
                thumbnail_quarantined = quarantined_thumbnail.exists()
                return
            quarantined_thumbnail.unlink()
            thumbnail_quarantined = False
            self.__fsync_directory(quarantine_directory)
            self.__fsync_directory(thumbnail.parent)
        except Exception:
            if thumbnail_quarantined:
                self._restore_quarantined_thumbnail(quarantined_thumbnail, thumbnail)
            raise
        finally:
            try:
                quarantine_directory.rmdir()
            except OSError:
                pass

    def _restore_quarantined_thumbnail(self, quarantined: Path, thumbnail: Path) -> None:
        try:
            os.link(quarantined, thumbnail)
            quarantined.unlink()
            self.__fsync_directory(quarantined.parent)
            self.__fsync_directory(thumbnail.parent)
        except OSError:
            # Keep the quarantined file for recovery if the canonical path was recreated or
            # filesystem operations failed. Never overwrite a concurrent replacement.
            pass

    @staticmethod
    def _path_signature(path: Path) -> _StatSignature | None:
        try:
            result = path.lstat()
        except FileNotFoundError:
            return None
        return GalleryMaintenanceService._signature(result) if stat.S_ISREG(result.st_mode) else None

    @staticmethod
    def __fsync_directory(directory: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _restore_archived_file(self, archived: Path | None, original: Path, counts: _ResultCounts) -> bool:
        if archived is None:
            return True
        try:
            signature = self._stat(archived)
            if signature is not None:
                self._archive_file(archived, original, signature)
            return original.exists()
        except Exception:
            self._logger.exception("Failed to restore an archived thumbnail after database failure")
            counts.failed_count += 1
            counts.add_error(f"Could not restore thumbnail {original.name}; it remains in the archive.")
            return False

    def _record_item_failure(self, counts: _ResultCounts, kind: str, relative_name: str, error: Exception) -> None:
        self._logger.warning("Gallery maintenance could not archive %s %s: %s", kind, relative_name, error)
        counts.failed_count += 1
        counts.add_error(f"Could not archive a gallery {kind}.")
