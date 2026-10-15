import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import ColumnElement, Table, func, select

from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_common import BoardChanges
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.gallery_maintenance.gallery_maintenance_common import (
    GalleryMaintenanceError,
    GalleryMaintenanceOperation,
    GalleryMaintenancePreviewChanged,
)
from invokeai.app.services.gallery_maintenance.gallery_maintenance_default import GalleryMaintenanceService
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_index.image_index_common import IndexedItem
from invokeai.app.services.image_index.image_index_records_default import ImageIndexRecords
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecordNotFoundException,
    ResourceOrigin,
)
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import board_images
from invokeai.app.services.shared.database.schema.image_index import image_embeddings, video_embeddings
from invokeai.app.services.shared.database.schema.image_moves import image_subfolder_move_items
from invokeai.app.services.video_records.video_records_default import VideoRecordStorage


class _ImageMutationGate:
    @contextmanager
    def reserve_gallery_maintenance(self):
        yield


@pytest.fixture
def maintenance(
    tmp_path: Path, database: Database
) -> tuple[GalleryMaintenanceService, Database, ImageRecordStorage, DiskImageFileStorage]:
    config = InvokeAIAppConfig(use_memory_db=True)
    config._root = tmp_path
    db = database
    records = ImageRecordStorage(db)
    files = DiskImageFileStorage(config.outputs_path / "images")
    invoker = MagicMock()
    invoker.services.configuration = config
    invoker.services.logger = MagicMock()
    invoker.services.database = db
    invoker.services.image_records = records
    invoker.services.image_files = files
    invoker.services.image_index_records = ImageIndexRecords(db)
    invoker.services.video_records = VideoRecordStorage(db)
    invoker.services.board_records = BoardRecordStorage(db)
    invoker.services.board_image_records = BoardImageRecordStorage(db)
    invoker.services.images = MagicMock()
    invoker.services.image_moves = _ImageMutationGate()
    files.start(invoker)
    service = GalleryMaintenanceService()
    service.start(invoker)
    return service, db, records, files


def _count(database: Database, table: Table, where: ColumnElement[bool] | None = None) -> int:
    statement = select(func.count()).select_from(table)
    if where is not None:
        statement = statement.where(where)
    with database.begin(write=False) as conn:
        return conn.execute(statement).scalar_one()


def _save_image_record(
    records: ImageRecordStorage,
    image_name: str,
    image_subfolder: str = "",
    *,
    is_intermediate: bool = False,
    user_id: str | None = None,
) -> None:
    records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=8,
        height=8,
        has_workflow=False,
        is_intermediate=is_intermediate,
        image_subfolder=image_subfolder,
        user_id=user_id,
    )


def _write_png(files: DiskImageFileStorage, image_name: str, subfolder: str = "") -> bytes:
    path = files.get_path(image_name, image_subfolder=subfolder)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (8, 8), (17, 23, 42, 101))
    image.save(path, format="PNG")
    image.close()
    return path.read_bytes()


def _embedding(invoker_services, kind: str, name: str, model_id: str = "model-a") -> None:
    invoker_services.image_index_records.upsert_embedding(
        IndexedItem(kind=kind, name=name), model_id, np.array([0.25, 0.75], dtype=np.float32)
    )


def _embedding_bytes(invoker_services, kind: str, name: str, model_id: str = "model-a") -> bytes:
    _items, embeddings = invoker_services.image_index_records.get_embeddings(
        [IndexedItem(kind=kind, name=name)], model_id
    )
    return embeddings.tobytes()


def test_remove_missing_archives_thumbnail_and_cascades_all_image_relations(maintenance) -> None:
    service, db, records, files = maintenance
    invoker = service._invoker
    _save_image_record(records, "missing.png", "archived/board", is_intermediate=True, user_id="user-a")
    _save_image_record(records, "kept.png", "live", user_id="user-b")
    _save_image_record(records, "archived-board-kept.png", "archived/board", user_id="user-a")
    _write_png(files, "kept.png", "live")
    _write_png(files, "archived-board-kept.png", "archived/board")
    thumbnail_path = files.get_path("missing.png", thumbnail=True, image_subfolder="archived/board")
    thumbnail_path.parent.mkdir(parents=True)
    thumbnail_path.write_bytes(b"recoverable thumbnail")

    board = invoker.services.board_records.save("Archived board", "user-a")
    invoker.services.board_records.update(board.board_id, BoardChanges(archived=True))
    invoker.services.board_image_records.add_image_to_board(board.board_id, "missing.png")
    invoker.services.board_image_records.add_image_to_board(board.board_id, "archived-board-kept.png")
    _embedding(invoker.services, "image", "missing.png", "model-a")
    _embedding(invoker.services, "image", "missing.png", "model-b")
    _embedding(invoker.services, "image", "kept.png", "model-a")
    _embedding(invoker.services, "image", "archived-board-kept.png", "model-a")
    invoker.services.video_records.save(
        video_name="kept.mp4",
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=8,
        height=8,
        duration=1.0,
        fps=24.0,
        has_workflow=False,
    )
    _embedding(invoker.services, "video", "kept.mp4", "model-a")
    kept_image_embedding = _embedding_bytes(invoker.services, "image", "kept.png")
    kept_video_embedding = _embedding_bytes(invoker.services, "video", "kept.mp4")
    archived_board_embedding = _embedding_bytes(invoker.services, "image", "archived-board-kept.png")

    preview = service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)
    assert preview.affected_count == 1
    assert records.exists("missing.png")
    assert thumbnail_path.read_bytes() == b"recoverable thumbnail"

    result = service.execute(GalleryMaintenanceOperation.REMOVE_MISSING, preview.fingerprint)

    assert result.records_removed == 1
    assert result.status == "completed"
    with pytest.raises(ImageRecordNotFoundException):
        records.get("missing.png")
    assert not thumbnail_path.exists()
    archived_thumbnail = Path(result.archive_path) / "thumbnails" / "archived" / "board" / "missing.webp"
    assert archived_thumbnail.read_bytes() == b"recoverable thumbnail"
    assert _count(db, board_images, board_images.c.image_name == "missing.png") == 0
    assert _count(db, image_embeddings, image_embeddings.c.image_name == "missing.png") == 0
    assert _count(db, image_subfolder_move_items, image_subfolder_move_items.c.image_name == "missing.png") == 0
    assert _count(db, image_embeddings, image_embeddings.c.image_name == "kept.png") == 1
    assert _count(db, video_embeddings, video_embeddings.c.video_name == "kept.mp4") == 1
    assert IndexedItem(
        kind="image", name="missing.png"
    ) not in invoker.services.image_index_records.list_accessible_embedded_items(None, "model-a")
    assert _embedding_bytes(invoker.services, "image", "kept.png") == kept_image_embedding
    assert _embedding_bytes(invoker.services, "video", "kept.mp4") == kept_video_embedding
    assert _embedding_bytes(invoker.services, "image", "archived-board-kept.png") == archived_board_embedding
    invoker.services.images.notify_deleted.assert_called_once_with("missing.png")
    if db.dialect_name == "sqlite":
        assert result.backup_path is not None and Path(result.backup_path).is_file()
    else:
        # A server database is backed up by its operator, not by gallery maintenance.
        assert result.backup_path is None


def test_archive_untracked_preserves_nested_paths_and_never_archives_tracked_or_video_files(maintenance) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "tracked.png", "nested/tracked")
    _embedding(service._invoker.services, "image", "tracked.png")
    embedding_before = _embedding_bytes(service._invoker.services, "image", "tracked.png")
    tracked_original = _write_png(files, "tracked.png", "nested/tracked")
    tracked_thumbnail = files.get_path("tracked.png", thumbnail=True, image_subfolder="nested/tracked")
    tracked_thumbnail.parent.mkdir(parents=True, exist_ok=True)
    tracked_thumbnail.write_bytes(b"tracked thumbnail")
    untracked_original = _write_png(files, "orphan.png", "nested/orphan")
    untracked_thumbnail = files.get_path("orphan.png", thumbnail=True, image_subfolder="nested/orphan")
    untracked_thumbnail.parent.mkdir(parents=True, exist_ok=True)
    untracked_thumbnail.write_bytes(b"matching thumbnail")
    orphan_thumbnail = files.get_path("thumbnail-only.png", thumbnail=True, image_subfolder="nested/only")
    orphan_thumbnail.parent.mkdir(parents=True, exist_ok=True)
    orphan_thumbnail.write_bytes(b"thumbnail only")
    (files.image_root / "nested" / "orphan" / "not-a-video.png.mp4").write_bytes(b"video")

    preview = service.preview(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED)
    assert preview.affected_count == 2

    result = service.execute(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED, preview.fingerprint)

    archive_root = Path(result.archive_path)
    assert result.images_archived == 1
    assert result.thumbnails_archived == 2
    assert not files.get_path("orphan.png", image_subfolder="nested/orphan").exists()
    assert not untracked_thumbnail.exists()
    assert not orphan_thumbnail.exists()
    assert files.get_path("tracked.png", image_subfolder="nested/tracked").read_bytes() == tracked_original
    assert tracked_thumbnail.read_bytes() == b"tracked thumbnail"
    assert (archive_root / "images" / "nested" / "orphan" / "orphan.png").read_bytes() == untracked_original
    assert (archive_root / "thumbnails" / "nested" / "orphan" / "orphan.webp").read_bytes() == b"matching thumbnail"
    assert (archive_root / "thumbnails" / "nested" / "only" / "thumbnail-only.webp").read_bytes() == b"thumbnail only"
    assert _embedding_bytes(service._invoker.services, "image", "tracked.png") == embedding_before


def test_regenerate_missing_thumbnail_is_record_scoped_and_refreshes_cached_size(maintenance, monkeypatch) -> None:
    service, db, records, files = maintenance
    _save_image_record(records, "missing-thumb.png", "nested")
    original_bytes = _write_png(files, "missing-thumb.png", "nested")
    _embedding(service._invoker.services, "image", "missing-thumb.png")
    embedding_before = _embedding_bytes(service._invoker.services, "image", "missing-thumb.png")
    records.set_file_size_bytes("missing-thumb.png", 1)
    _write_png(files, "untracked.png", "nested")
    _save_image_record(records, "missing-source.png", "nested")
    existing_thumbnail = files.get_path("missing-source.png", thumbnail=True, image_subfolder="nested")
    existing_thumbnail.parent.mkdir(parents=True, exist_ok=True)
    existing_thumbnail.write_bytes(b"existing")
    _save_image_record(records, "raced-thumb.png", "nested")
    _write_png(files, "raced-thumb.png", "nested")

    preview = service.preview(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS)
    assert preview.affected_count == 2
    assert preview.skipped_count == 1

    original_backup = service._create_backup

    def create_thumbnail_during_backup() -> str | None:
        backup_path = original_backup()
        raced_thumbnail = files.get_path("raced-thumb.png", thumbnail=True, image_subfolder="nested")
        raced_thumbnail.parent.mkdir(parents=True, exist_ok=True)
        raced_thumbnail.write_bytes(b"created after preview scan")
        return backup_path

    monkeypatch.setattr(service, "_create_backup", create_thumbnail_during_backup)

    result = service.execute(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS, preview.fingerprint)

    assert result.thumbnails_regenerated == 1
    assert result.skipped_count == 2
    assert result.status == "partial"
    assert files.get_path("missing-thumb.png", image_subfolder="nested").read_bytes() == original_bytes
    assert files.get_path("missing-thumb.png", thumbnail=True, image_subfolder="nested").is_file()
    assert files.get_path("untracked.png", thumbnail=True, image_subfolder="nested").exists() is False
    assert existing_thumbnail.read_bytes() == b"existing"
    assert files.get_path("raced-thumb.png", thumbnail=True, image_subfolder="nested").read_bytes() == (
        b"created after preview scan"
    )
    assert records.get("missing-thumb.png").file_size_bytes == files.get_file_size_bytes(
        "missing-thumb.png", image_subfolder="nested"
    )
    assert _embedding_bytes(service._invoker.services, "image", "missing-thumb.png") == embedding_before
    assert _count(db, image_embeddings) == 1


def test_execute_rejects_changed_preview_without_mutating_newly_untracked_file(maintenance) -> None:
    service, _db, _records, files = maintenance
    preview = service.preview(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED)
    added_bytes = _write_png(files, "arrived-after-preview.png")

    with pytest.raises(GalleryMaintenancePreviewChanged):
        service.execute(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED, preview.fingerprint)

    assert files.get_path("arrived-after-preview.png").read_bytes() == added_bytes


@pytest.mark.sqlite_only  # Only a SQLite database is backed up by gallery maintenance.
def test_backup_failure_prevents_record_removal_and_thumbnail_archival(maintenance, monkeypatch) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "missing.png", "recoverable")
    thumbnail = files.get_path("missing.png", thumbnail=True, image_subfolder="recoverable")
    thumbnail.parent.mkdir(parents=True)
    thumbnail.write_bytes(b"recoverable thumbnail")
    preview = service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)
    monkeypatch.setattr(service._invoker.services.database, "backup", MagicMock(side_effect=OSError("disk full")))

    with pytest.raises(GalleryMaintenanceError, match="database backup failed"):
        service.execute(GalleryMaintenanceOperation.REMOVE_MISSING, preview.fingerprint)

    assert records.exists("missing.png")
    assert thumbnail.read_bytes() == b"recoverable thumbnail"
    service._invoker.services.images.notify_deleted.assert_not_called()


def test_delete_failure_restores_archived_thumbnail_and_reports_failure(maintenance) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "missing.png", "recoverable")
    thumbnail = files.get_path("missing.png", thumbnail=True, image_subfolder="recoverable")
    thumbnail.parent.mkdir(parents=True)
    thumbnail_bytes = b"recoverable thumbnail"
    thumbnail.write_bytes(thumbnail_bytes)
    preview = service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)
    records.delete_many = MagicMock(side_effect=OSError("database unavailable"))

    result = service.execute(GalleryMaintenanceOperation.REMOVE_MISSING, preview.fingerprint)

    assert records.get("missing.png").image_name == "missing.png"
    assert thumbnail.read_bytes() == thumbnail_bytes
    assert not (Path(result.archive_path) / "thumbnails" / "recoverable" / "missing.webp").exists()
    assert result.records_removed == 0
    assert result.thumbnails_archived == 0
    assert result.failed_count == 1
    assert result.status == "failed"


def test_thumbnail_eviction_failure_restores_archived_thumbnail(maintenance, monkeypatch) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "missing.png", "recoverable")
    thumbnail = files.get_path("missing.png", thumbnail=True, image_subfolder="recoverable")
    thumbnail.parent.mkdir(parents=True)
    thumbnail_bytes = b"recoverable thumbnail"
    thumbnail.write_bytes(thumbnail_bytes)
    preview = service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)

    def fail_cache_eviction(paths: list[Path]) -> None:
        assert thumbnail in paths
        raise OSError("cache eviction failed")

    monkeypatch.setattr(files, "evict_cache_paths", fail_cache_eviction)

    result = service.execute(GalleryMaintenanceOperation.REMOVE_MISSING, preview.fingerprint)

    archived_thumbnail = Path(result.archive_path) / "thumbnails" / "recoverable" / "missing.webp"
    assert records.exists("missing.png")
    assert thumbnail.read_bytes() == thumbnail_bytes
    assert not archived_thumbnail.exists()
    assert result.records_removed == 0
    assert result.thumbnails_archived == 0
    assert result.failed_count == 1
    assert result.status == "failed"


@pytest.mark.parametrize("root_name", ["image_root", "thumbnail_root"])
def test_preview_fails_when_required_storage_root_is_missing(maintenance, root_name: str) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "missing.png", "missing")
    root = getattr(files, root_name)
    shutil.rmtree(root) if root_name == "image_root" else root.rmdir()

    with pytest.raises(GalleryMaintenanceError, match="storage"):
        service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)

    assert records.exists("missing.png")


def test_preview_fails_when_storage_root_cannot_be_scanned(maintenance, monkeypatch: pytest.MonkeyPatch) -> None:
    service, _db, _records, files = maintenance
    real_scandir = os.scandir

    def unavailable(path):
        if not isinstance(path, int) and Path(path) == files.thumbnail_root:
            raise PermissionError("unreadable thumbnail root")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", unavailable)

    with pytest.raises(GalleryMaintenanceError, match="storage"):
        service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)


def test_remove_missing_preserves_record_when_same_basename_image_exists_elsewhere(maintenance) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "duplicate.png", "recorded")
    elsewhere = _write_png(files, "duplicate.png", "actual")

    preview = service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)

    assert preview.affected_count == 0
    assert preview.skipped_count == 1
    result = service.execute(GalleryMaintenanceOperation.REMOVE_MISSING, preview.fingerprint)

    assert result.records_removed == 0
    assert records.exists("duplicate.png")
    assert files.get_path("duplicate.png", image_subfolder="actual").read_bytes() == elsewhere


def test_archive_untracked_executes_only_confirmed_files_with_unchanged_stats(maintenance, monkeypatch) -> None:
    service, _db, _records, files = maintenance
    confirmed = _write_png(files, "confirmed.png")
    changed_path = files.get_path("changed.png")
    changed_path.parent.mkdir(parents=True, exist_ok=True)
    changed_path.write_bytes(b"before preview")
    preview = service.preview(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED)
    original_backup = service._create_backup

    def change_storage() -> str | None:
        backup_path = original_backup()
        changed_path.write_bytes(b"changed after preview")
        _write_png(files, "arrived.png")
        return backup_path

    monkeypatch.setattr(service, "_create_backup", change_storage)

    result = service.execute(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED, preview.fingerprint)

    assert result.images_archived == 1
    assert not files.get_path("confirmed.png").exists()
    assert changed_path.read_bytes() == b"changed after preview"
    assert files.get_path("arrived.png").is_file()
    archived = Path(result.archive_path) / "images" / "confirmed.png"
    assert archived.read_bytes() == confirmed


def test_remove_missing_counts_committed_deletion_when_followup_check_fails(maintenance) -> None:
    service, _db, records, _files = maintenance
    _save_image_record(records, "missing.png")
    preview = service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)
    records.exists = MagicMock(side_effect=OSError("verification unavailable"))
    service._invoker.services.images.notify_deleted.side_effect = OSError("notification unavailable")

    result = service.execute(GalleryMaintenanceOperation.REMOVE_MISSING, preview.fingerprint)

    assert result.records_removed == 1
    assert result.failed_count == 2
    assert result.status == "partial"


def test_remove_missing_keeps_record_when_original_appears_during_backup(maintenance, monkeypatch) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "appeared.png", "during-backup")
    preview = service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)
    original_backup = service._create_backup

    def create_original() -> str | None:
        backup_path = original_backup()
        _write_png(files, "appeared.png", "during-backup")
        return backup_path

    monkeypatch.setattr(service, "_create_backup", create_original)

    result = service.execute(GalleryMaintenanceOperation.REMOVE_MISSING, preview.fingerprint)

    assert result.records_removed == 0
    assert result.skipped_count == 1
    assert records.exists("appeared.png")
    assert files.get_path("appeared.png", image_subfolder="during-backup").is_file()
    service._invoker.services.images.notify_deleted.assert_not_called()


def test_archive_iterator_failure_returns_partial_result_after_prior_archive(maintenance, monkeypatch) -> None:
    service, _db, _records, files = maintenance
    first = _write_png(files, "first.png")
    second = _write_png(files, "second.png")
    preview = service.preview(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED)
    original_stat = service._stat
    second_stat_calls = 0

    def fail_on_second_execution_stat(path: Path):
        nonlocal second_stat_calls
        if path == files.get_path("second.png"):
            second_stat_calls += 1
            if second_stat_calls == 2:
                raise OSError("stat failed during execution")
        return original_stat(path)

    monkeypatch.setattr(service, "_stat", fail_on_second_execution_stat)

    result = service.execute(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED, preview.fingerprint)

    assert result.images_archived == 1
    assert result.failed_count == 1
    assert result.status == "partial"
    archive_root = Path(result.archive_path) / "images"
    assert (archive_root / "first.png").read_bytes() == first
    assert files.get_path("second.png").read_bytes() == second


def test_archive_thumbnail_failure_keeps_original_recoverable(maintenance, monkeypatch) -> None:
    service, _db, _records, files = maintenance
    original_bytes = _write_png(files, "paired.png", "nested")
    thumbnail = files.get_path("paired.png", thumbnail=True, image_subfolder="nested")
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_bytes = b"recoverable thumbnail"
    thumbnail.write_bytes(thumbnail_bytes)
    preview = service.preview(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED)
    archive_file = service._archive_file

    def fail_thumbnail(source: Path, destination: Path, expected) -> None:
        if "thumbnails" in destination.parts:
            raise OSError("injected thumbnail archive failure")
        archive_file(source, destination, expected)

    monkeypatch.setattr(service, "_archive_file", fail_thumbnail)

    result = service.execute(GalleryMaintenanceOperation.ARCHIVE_UNTRACKED, preview.fingerprint)

    archive_root = Path(result.archive_path)
    assert (archive_root / "images" / "nested" / "paired.png").read_bytes() == original_bytes
    assert thumbnail.read_bytes() == thumbnail_bytes
    assert result.images_archived == 1
    assert result.thumbnails_archived == 0
    assert result.failed_count == 1
    assert result.status == "partial"


def test_archive_collision_never_overwrites_destination(maintenance, tmp_path: Path) -> None:
    service, _db, _records, _files = maintenance
    source = tmp_path / "source.png"
    destination = tmp_path / "archive" / "source.png"
    source.write_bytes(b"source data")
    destination.parent.mkdir()
    destination.write_bytes(b"prior archive data")
    signature = service._signature(source.lstat())

    with pytest.raises(FileExistsError):
        service._archive_file(source, destination, signature)

    assert source.read_bytes() == b"source data"
    assert destination.read_bytes() == b"prior archive data"


def test_archive_file_preserves_replacement_after_signature_check(maintenance, tmp_path: Path, monkeypatch) -> None:
    service, _db, _records, _files = maintenance
    source = tmp_path / "source.png"
    replacement = tmp_path / "replacement.png"
    destination = tmp_path / "archive" / "source.png"
    original_bytes = b"confirmed source bytes"
    replacement_bytes = b"concurrent replacement bytes"
    source.write_bytes(original_bytes)
    replacement.write_bytes(replacement_bytes)
    destination.parent.mkdir()
    expected = service._signature(source.lstat())
    path_signature = service._path_signature
    replaced = False

    def replace_after_signature(path: Path):
        nonlocal replaced
        signature = path_signature(path)
        if path == source and not replaced:
            source.unlink()
            replacement.replace(source)
            replaced = True
        return signature

    monkeypatch.setattr(GalleryMaintenanceService, "_path_signature", staticmethod(replace_after_signature))

    with pytest.raises(OSError, match="changed"):
        service._archive_file(source, destination, expected)

    assert source.read_bytes() == replacement_bytes
    assert destination.read_bytes() == original_bytes


def test_archive_file_tolerates_ctime_difference_between_path_and_open_file(
    maintenance, tmp_path: Path, monkeypatch
) -> None:
    service, _db, _records, _files = maintenance
    source = tmp_path / "source.png"
    destination = tmp_path / "archive" / "source.png"
    source_bytes = b"confirmed source bytes"
    source.write_bytes(source_bytes)
    expected = service._signature(source.lstat())
    real_fstat = os.fstat

    def fstat_with_different_ctime(descriptor: int):
        result = real_fstat(descriptor)
        return SimpleNamespace(
            st_dev=result.st_dev,
            st_ino=result.st_ino,
            st_mode=result.st_mode,
            st_size=result.st_size,
            st_mtime_ns=result.st_mtime_ns,
            st_ctime_ns=result.st_ctime_ns + 1,
        )

    monkeypatch.setattr(
        "invokeai.app.services.gallery_maintenance.gallery_maintenance_default.os.fstat", fstat_with_different_ctime
    )

    service._archive_file(source, destination, expected)

    assert destination.read_bytes() == source_bytes
    assert not source.exists()


def test_preview_rejects_symlinked_record_subfolder(maintenance, tmp_path: Path) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "outside.png", "linked")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_image = outside / "outside.png"
    outside_image.write_bytes(b"outside data")
    try:
        (files.image_root / "linked").symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as e:
        pytest.skip(f"directory symlinks are unavailable: {e}")

    with pytest.raises(GalleryMaintenanceError, match="symbolic links"):
        service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)

    assert outside_image.read_bytes() == b"outside data"


def test_repeated_thumbnail_regeneration_becomes_no_op(maintenance) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "image.png")
    _write_png(files, "image.png")
    first_preview = service.preview(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS)
    service.execute(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS, first_preview.fingerprint)

    second_preview = service.preview(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS)
    second_result = service.execute(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS, second_preview.fingerprint)

    assert second_preview.affected_count == 0
    assert second_result.status == "no_op"


def test_regeneration_removes_thumbnail_if_source_changes_during_generation(maintenance, monkeypatch) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "changing.png", "nested")
    original_path = files.get_path("changing.png", image_subfolder="nested")
    original_bytes = _write_png(files, "changing.png", "nested")
    thumbnail = files.get_path("changing.png", thumbnail=True, image_subfolder="nested")
    preview = service.preview(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS)
    generate = files.generate_thumbnail_if_missing

    def generate_then_change_source(image_name: str, image_subfolder: str = "") -> bool:
        created = generate(image_name, image_subfolder=image_subfolder)
        original_path.write_bytes(b"changed while thumbnail was generated")
        return created

    monkeypatch.setattr(files, "generate_thumbnail_if_missing", generate_then_change_source)

    result = service.execute(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS, preview.fingerprint)

    assert original_bytes != original_path.read_bytes()
    assert not thumbnail.exists()
    assert result.thumbnails_regenerated == 0
    assert result.failed_count == 1
    assert result.status == "failed"


def test_regeneration_removes_thumbnail_if_size_persistence_fails(maintenance, monkeypatch) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "persistence-failure.png", "nested")
    _write_png(files, "persistence-failure.png", "nested")
    thumbnail = files.get_path("persistence-failure.png", thumbnail=True, image_subfolder="nested")
    preview = service.preview(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS)
    monkeypatch.setattr(records, "set_file_size_bytes", MagicMock(side_effect=OSError("database write failed")))

    result = service.execute(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS, preview.fingerprint)

    assert not thumbnail.exists()
    assert result.thumbnails_regenerated == 0
    assert result.failed_count == 1
    assert result.status == "failed"


def test_regeneration_cleanup_preserves_replacement_thumbnail(maintenance, monkeypatch) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "replacement.png", "nested")
    _write_png(files, "replacement.png", "nested")
    thumbnail = files.get_path("replacement.png", thumbnail=True, image_subfolder="nested")
    replacement = thumbnail.with_name("replacement-copy.webp")
    replacement_bytes = b"concurrent thumbnail replacement"
    preview = service.preview(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS)
    monkeypatch.setattr(records, "set_file_size_bytes", MagicMock(side_effect=OSError("database write failed")))
    path_signature = service._path_signature
    thumbnail_signature_checks = 0

    def replace_after_cleanup_signature(path: Path):
        nonlocal thumbnail_signature_checks
        signature = path_signature(path)
        if path == thumbnail:
            thumbnail_signature_checks += 1
            if thumbnail_signature_checks == 2:
                replacement.write_bytes(replacement_bytes)
                thumbnail.unlink()
                replacement.replace(thumbnail)
        return signature

    monkeypatch.setattr(GalleryMaintenanceService, "_path_signature", staticmethod(replace_after_cleanup_signature))

    result = service.execute(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS, preview.fingerprint)

    assert thumbnail_signature_checks == 2
    assert thumbnail.read_bytes() == replacement_bytes
    assert result.thumbnails_regenerated == 0
    assert result.failed_count == 1
    assert result.status == "failed"


def test_result_errors_are_bounded(maintenance) -> None:
    service, _db, records, files = maintenance
    for index in range(25):
        name = f"broken-{index:02}.png"
        _save_image_record(records, name)
        path = files.get_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a png")
    preview = service.preview(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS)

    result = service.execute(GalleryMaintenanceOperation.REGENERATE_THUMBNAILS, preview.fingerprint)

    assert result.failed_count == 25
    assert len(result.errors) == 20


def test_archive_directory_entries_are_synced_before_quarantining_thumbnail(maintenance, monkeypatch) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "missing.png", "recoverable/deep")
    thumbnail = files.get_path("missing.png", thumbnail=True, image_subfolder="recoverable/deep")
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_bytes = b"recoverable thumbnail"
    thumbnail.write_bytes(thumbnail_bytes)
    preview = service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)
    archive_root = service._archive_root
    output_root = archive_root.parent
    events: list[tuple[str, Path]] = []
    real_mkdir = os.mkdir
    real_fsync_directory = GalleryMaintenanceService._GalleryMaintenanceService__fsync_directory
    real_replace = Path.replace
    durability_failure_injected = False

    def track_mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        created_path = Path(os.fsdecode(path))
        if created_path == archive_root or archive_root in created_path.parents:
            events.append(("mkdir", created_path))
        return result

    def fail_archive_parent_sync(directory: Path) -> None:
        nonlocal durability_failure_injected
        directory = Path(directory)
        if directory == output_root or directory == archive_root or archive_root in directory.parents:
            events.append(("fsync", directory))
            if directory.name == "recoverable" and directory.parent.name == "thumbnails":
                durability_failure_injected = True
                raise OSError("injected archive directory-entry sync failure")
        real_fsync_directory(directory)

    def track_source_quarantine(path: Path, target: Path) -> Path:
        if path == thumbnail and Path(target).parent.name.startswith(".gallery-archive-"):
            events.append(("quarantine", path))
        return real_replace(path, target)

    monkeypatch.setattr(os, "mkdir", track_mkdir)
    monkeypatch.setattr(
        GalleryMaintenanceService,
        "_GalleryMaintenanceService__fsync_directory",
        staticmethod(fail_archive_parent_sync),
    )
    monkeypatch.setattr(Path, "replace", track_source_quarantine)

    result = service.execute(GalleryMaintenanceOperation.REMOVE_MISSING, preview.fingerprint)

    assert durability_failure_injected
    assert not any(kind == "quarantine" for kind, _path in events)
    assert records.exists("missing.png")
    assert thumbnail.read_bytes() == thumbnail_bytes
    assert result.records_removed == 0
    assert result.thumbnails_archived == 0

    for index, (kind, directory) in enumerate(events):
        if kind != "mkdir":
            continue
        parent_syncs = [
            event_index
            for event_index, (event_kind, synced_directory) in enumerate(events)
            if event_kind == "fsync" and synced_directory == directory.parent
        ]
        assert parent_syncs and parent_syncs[0] > index


@pytest.mark.sqlite_only  # Only a SQLite database is backed up by gallery maintenance.
def test_backup_directory_sync_failure_prevents_gallery_deletion(maintenance, monkeypatch) -> None:
    service, _db, records, files = maintenance
    _save_image_record(records, "missing.png", "recoverable")
    thumbnail = files.get_path("missing.png", thumbnail=True, image_subfolder="recoverable")
    thumbnail.parent.mkdir(parents=True)
    thumbnail_bytes = b"recoverable thumbnail"
    thumbnail.write_bytes(thumbnail_bytes)
    _embedding(service._invoker.services, "image", "missing.png")
    embedding_before = _embedding_bytes(service._invoker.services, "image", "missing.png")
    preview = service.preview(GalleryMaintenanceOperation.REMOVE_MISSING)
    backup_directory = service._invoker.services.configuration.db_path.parent / "backup"
    database_directory = backup_directory.parent
    backup = service._invoker.services.database.backup
    directory_fsync = GalleryMaintenanceService._GalleryMaintenanceService__fsync_directory
    real_mkdir = os.mkdir
    events: list[tuple[str, Path]] = []
    completed_backup: Path | None = None
    sync_failure_injected = False

    assert not database_directory.exists()
    assert not backup_directory.exists()

    def track_mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        created_path = Path(os.fsdecode(path))
        if created_path in {database_directory, backup_directory}:
            events.append(("mkdir", created_path))
        return result

    def create_backup(destination: Path) -> None:
        nonlocal completed_backup
        backup(destination)
        completed_backup = destination
        events.append(("backup", destination))

    def fail_backup_directory_sync(directory: Path) -> None:
        nonlocal sync_failure_injected
        directory = Path(directory)
        events.append(("fsync", directory))
        if directory == backup_directory:
            sync_failure_injected = True
            raise OSError("injected backup directory-entry sync failure")
        directory_fsync(directory)

    monkeypatch.setattr(service._invoker.services.database, "backup", create_backup)
    monkeypatch.setattr(os, "mkdir", track_mkdir)
    monkeypatch.setattr(
        GalleryMaintenanceService,
        "_GalleryMaintenanceService__fsync_directory",
        staticmethod(fail_backup_directory_sync),
    )

    with pytest.raises(GalleryMaintenanceError, match="database backup"):
        service.execute(GalleryMaintenanceOperation.REMOVE_MISSING, preview.fingerprint)

    assert sync_failure_injected
    assert completed_backup is not None and completed_backup.is_file()
    assert records.exists("missing.png")
    assert _embedding_bytes(service._invoker.services, "image", "missing.png") == embedding_before
    assert thumbnail.read_bytes() == thumbnail_bytes
    assert not service._archive_root.exists()
    service._invoker.services.images.notify_deleted.assert_not_called()

    backup_index = next(index for index, (kind, _path) in enumerate(events) if kind == "backup")
    for created_directory in (database_directory, backup_directory):
        mkdir_index = next(
            index
            for index, (kind, directory) in enumerate(events)
            if kind == "mkdir" and directory == created_directory
        )
        parent_sync_index = next(
            index
            for index, (kind, directory) in enumerate(events)
            if kind == "fsync" and directory == created_directory.parent
        )
        assert mkdir_index < backup_index < parent_sync_index
