import os
import threading
from pathlib import Path
from shutil import copy2
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from sqlalchemy import insert, select, update

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_moves.image_moves_default import (
    ImageMoveJob,
    ImageMoveJobAlreadyRunning,
    ImageMoveQueueActive,
    ImageMoveService,
    UnreadableImageError,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.image_moves import (
    image_subfolder_move_items,
    image_subfolder_move_jobs,
)
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.backend.util.logging import InvokeAILogger


def _save_record(
    database: Database,
    records: ImageRecordStorage,
    image_name: str,
    subfolder: str,
    created_at: str,
    is_intermediate: bool = False,
) -> None:
    records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=16,
        height=16,
        has_workflow=False,
        is_intermediate=is_intermediate,
        image_subfolder=subfolder,
    )
    _set_record(database, image_name, created_at=created_at)


def _set_record(database: Database, image_name: str, **values: str) -> None:
    with database.begin(write=True) as conn:
        conn.execute(update(images).where(images.c.image_name == image_name).values(**values))


def _save_image(
    database: Database,
    service: ImageMoveService,
    records: ImageRecordStorage,
    image_name: str,
    subfolder: str,
    created_at: str,
    color: str,
    is_intermediate: bool = False,
) -> None:
    _save_record(
        database,
        records,
        image_name=image_name,
        subfolder=subfolder,
        created_at=created_at,
        is_intermediate=is_intermediate,
    )
    service.image_files.save(Image.new("RGB", (16, 16), color), image_name=image_name, image_subfolder=subfolder)


def _corrupt_png_idat(path: Path) -> None:
    data = bytearray(path.read_bytes())
    idat_data_offset = data.index(b"IDAT") + 4
    data[idat_data_offset + 1] ^= 0xFF
    path.write_bytes(data)


def _service(tmp_path: Path, database: Database, strategy: str = "date") -> tuple[ImageMoveService, ImageRecordStorage]:
    records = ImageRecordStorage(database)
    storage = DiskImageFileStorage(tmp_path / "images")
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 6
    storage.start(invoker)
    config = InvokeAIAppConfig(use_memory_db=True, image_subfolder_strategy=strategy)
    config._root = tmp_path
    service = ImageMoveService(database, image_files=storage, config=config, logger=InvokeAILogger.get_logger())
    return service, records


def _job_item_states(database: Database, job_id: int) -> dict[str, str]:
    items = image_subfolder_move_items.c
    statement = select(items.image_name, items.state).where(items.job_id == job_id).order_by(items.image_name)
    with database.begin(write=False) as conn:
        return dict(conn.execute(statement).all())


def _job_states(database: Database) -> dict[int, str]:
    jobs = image_subfolder_move_jobs.c
    with database.begin(write=False) as conn:
        return dict(conn.execute(select(jobs.id, jobs.state).order_by(jobs.id)).all())


def test_move_all_images_uses_created_at_for_date_strategy(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-a.png"
    _save_record(database, records, image_name=image_name, subfolder="", created_at="2024-02-03 04:05:06.000")
    service.image_files.save(Image.new("RGB", (16, 16), "red"), image_name=image_name)

    result = service.move_all_images()

    assert result.planned == 1
    assert result.committed == 1
    record = records.get(image_name)
    assert record.image_subfolder == "2024/02/03"
    assert service.image_files.get_path(image_name, image_subfolder="2024/02/03").exists()
    assert not service.image_files.get_path(image_name, image_subfolder="").exists()


def test_missing_intermediate_source_file_is_treated_as_success(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "missing-intermediate.png"
    _save_record(
        database,
        records,
        image_name=image_name,
        subfolder="",
        created_at="2024-02-04 04:05:06.000",
        is_intermediate=True,
    )

    result = service.move_all_images()

    assert result.planned == 1
    assert result.committed == 1
    assert result.errors == 0
    record = records.get(image_name)
    assert record.image_subfolder == "2024/02/04"
    assert service.get_latest_job().state == "committed"


def test_missing_intermediate_source_file_removes_orphaned_thumbnail(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "missing-intermediate-with-thumbnail.png"
    old_subfolder = "old/intermediate"
    _save_image(
        database,
        service,
        records,
        image_name=image_name,
        subfolder=old_subfolder,
        created_at="2024-02-04 04:05:06.000",
        color="red",
        is_intermediate=True,
    )
    old_path = service.image_files.get_path(image_name, image_subfolder=old_subfolder)
    old_thumbnail_path = service.image_files.get_path(image_name, thumbnail=True, image_subfolder=old_subfolder)
    assert old_thumbnail_path.exists()
    old_path.unlink()

    result = service.move_all_images()

    assert result.committed == 1
    assert not old_thumbnail_path.exists()
    assert records.get(image_name).image_subfolder == "2024/02/04"


def test_missing_non_intermediate_source_file_still_fails(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "missing-general.png"
    _save_record(
        database,
        records,
        image_name=image_name,
        subfolder="",
        created_at="2024-02-04 04:05:06.000",
        is_intermediate=False,
    )

    with pytest.raises(FileNotFoundError, match="Source image does not exist"):
        service.plan_batch(last_image_name="", limit=100)


def test_move_all_images_continues_after_missing_non_intermediate_source_file(
    tmp_path: Path, database: Database
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    missing_image_name = "missing-general.png"
    valid_image_name = "valid-general.png"
    _save_record(
        database,
        records,
        image_name=missing_image_name,
        subfolder="",
        created_at="2024-02-04 04:05:06.000",
        is_intermediate=False,
    )
    _save_image(database, service, records, valid_image_name, "", "2024-02-05 04:05:06.000", "blue")

    result = service.move_all_images()

    assert result.errors == 1
    assert result.committed == 1
    assert records.get(missing_image_name).image_subfolder == ""
    assert records.get(valid_image_name).image_subfolder == "2024/02/05"
    assert "error" in _job_states(database).values()
    assert "committed" in _job_states(database).values()


def test_move_recovers_existing_16_bit_destination_without_thumbnail(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "large-16-bit.png"
    _save_record(database, records, image_name=image_name, subfolder="", created_at="2024-02-04 04:05:06.000")
    old_path = service.image_files.get_path(image_name)
    old_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("I;16", (1024, 1024), 32768)

    try:
        image.save(old_path, format="PNG")
        move = service.plan_batch(last_image_name="", limit=100)[0]
        job_id = service.create_move_job([move])
        move.new_path.parent.mkdir(parents=True, exist_ok=True)
        move.old_path.replace(move.new_path)

        recovered = service.startup_recovery()

        assert recovered.committed == 1
        assert recovered.errors == 0
        assert records.get(image_name).image_subfolder == "2024/02/04"
        assert move.new_path.exists()
        assert move.new_thumbnail_path.exists()
        assert service.get_job(job_id).state == "committed"
    finally:
        image.close()


def test_regenerate_thumbnail_closes_temp_file_before_writing(tmp_path: Path, database: Database) -> None:
    service, _records = _service(tmp_path, database, strategy="date")

    class TrackingTempFile:
        name = str(tmp_path / "thumbnail.tmp")

        def __init__(self) -> None:
            self.closed = False

        def __enter__(self) -> "TrackingTempFile":
            return self

        def __exit__(self, *_args) -> None:
            self.closed = True

    temp_file = TrackingTempFile()
    thumbnail = MagicMock()

    def save(_path: Path, format: str) -> None:
        assert temp_file.closed

    thumbnail.save.side_effect = save
    source_image = MagicMock()
    source_image.__enter__.return_value = source_image

    with (
        patch("invokeai.app.services.image_moves.image_moves_default.Image.open", return_value=source_image),
        patch("invokeai.app.services.image_moves.image_moves_default.make_thumbnail", return_value=thumbnail),
        patch(
            "invokeai.app.services.image_moves.image_moves_default.tempfile.NamedTemporaryFile", return_value=temp_file
        ),
        patch.object(service, "_fsync_file"),
        patch.object(service, "_fsync_dir"),
        patch("invokeai.app.services.image_moves.image_moves_default.os.replace"),
    ):
        service._regenerate_thumbnail(tmp_path / "source.png", tmp_path / "thumbnail.webp")

    assert temp_file.closed


def test_move_does_not_relocate_source_when_thumbnail_generation_fails(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "thumbnail-retry.png"
    _save_record(database, records, image_name=image_name, subfolder="", created_at="2024-02-05 04:05:06.000")
    old_path = service.image_files.get_path(image_name)
    old_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "red").save(old_path, format="PNG")
    move = service.plan_batch(last_image_name="", limit=100)[0]
    job_id = service.create_move_job([move])

    with patch.object(service, "_regenerate_thumbnail", side_effect=OSError("thumbnail unavailable")):
        with pytest.raises(OSError, match="thumbnail unavailable"):
            service.perform_filesystem_moves(job_id)

    assert move.old_path.exists()
    assert not move.new_path.exists()
    assert records.get(image_name).image_subfolder == ""
    assert service.get_job(job_id).state == "moving"

    recovered = service.startup_recovery()

    assert recovered.committed == 1
    assert recovered.errors == 0
    assert records.get(image_name).image_subfolder == "2024/02/05"
    assert move.new_path.exists()
    assert move.new_thumbnail_path.exists()
    assert service.get_job(job_id).state == "committed"


def test_recovery_treats_missing_intermediate_source_file_as_success(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "missing-intermediate-recovery.png"
    _save_record(
        database,
        records,
        image_name=image_name,
        subfolder="",
        created_at="2024-02-05 04:05:06.000",
        is_intermediate=True,
    )
    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)

    recovered = service.startup_recovery()

    assert recovered.committed == 1
    assert recovered.errors == 0
    assert records.get(image_name).image_subfolder == "2024/02/05"
    assert service.get_job(job_id).state == "committed"
    assert _job_item_states(database, job_id) == {image_name: "committed"}


def test_startup_recovery_commits_after_files_moved_but_db_not_updated(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-b.png"
    _save_record(database, records, image_name=image_name, subfolder="", created_at="2025-06-07 08:09:10.000")
    service.image_files.save(Image.new("RGB", (16, 16), "blue"), image_name=image_name)

    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    service.perform_filesystem_moves(job_id)

    assert records.get(image_name).image_subfolder == ""

    recovered = service.startup_recovery()

    assert recovered.committed == 1
    assert records.get(image_name).image_subfolder == "2025/06/07"
    assert service.get_job(job_id).state == "committed"


def test_status_reports_unplanned_images_after_recovery(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    _save_image(database, service, records, "image-recovered.png", "", "2024-02-03 04:05:06.000", "red")
    _save_image(database, service, records, "image-unplanned.png", "", "2024-02-04 04:05:06.000", "blue")
    moves = service.plan_batch(last_image_name="", limit=1)
    job_id = service.create_move_job(moves)
    service.perform_filesystem_moves(job_id)

    recovered = service.startup_recovery()
    status = service.get_background_status()

    assert recovered.committed == 1
    assert records.get("image-recovered.png").image_subfolder == "2024/02/03"
    assert records.get("image-unplanned.png").image_subfolder == ""
    assert status.active_job_id is None
    assert status.latest_job is not None
    assert status.latest_job.state == "committed"
    assert status.needs_move_count == 1


def test_cleanup_empty_source_directories_after_move(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-c.png"
    old_subfolder = "old/nested"
    _save_record(
        database, records, image_name=image_name, subfolder=old_subfolder, created_at="2024-11-12 01:02:03.000"
    )
    service.image_files.save(Image.new("RGB", (16, 16), "green"), image_name=image_name, image_subfolder=old_subfolder)
    old_parent = service.image_files.get_path(image_name, image_subfolder=old_subfolder).parent
    old_thumb_parent = service.image_files.get_path(image_name, thumbnail=True, image_subfolder=old_subfolder).parent

    service.move_all_images()

    assert not old_parent.exists()
    assert not old_thumb_parent.exists()
    assert service.image_files.image_root.exists()
    assert service.image_files.thumbnail_root.exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are not supported on this platform")
def test_cleanup_empty_source_directories_stays_within_symlinked_root(tmp_path: Path, database: Database) -> None:
    service, _records = _service(tmp_path, database, strategy="date")
    real_root = tmp_path / "real-root"
    linked_root = tmp_path / "linked-root"
    sibling = tmp_path / "sibling"
    real_root.mkdir()
    sibling.mkdir()
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError as e:
        pytest.skip(f"symlink creation is not available: {e}")
    nested = linked_root / "old" / "nested"
    nested.mkdir(parents=True)

    service._remove_empty_parents(nested, linked_root)

    assert real_root.exists()
    assert linked_root.exists()
    assert sibling.exists()
    assert not (real_root / "old").exists()


def test_startup_recovery_cleans_empty_source_directories(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-recovery-cleanup.png"
    old_subfolder = "old/recovery"
    _save_image(database, service, records, image_name, old_subfolder, "2024-11-13 01:02:03.000", "green")
    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    move = moves[0]
    old_parent = service.image_files.get_path(image_name, image_subfolder=old_subfolder).parent
    old_thumb_parent = service.image_files.get_path(image_name, thumbnail=True, image_subfolder=old_subfolder).parent
    move.new_path.parent.mkdir(parents=True, exist_ok=True)
    move.new_thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    move.old_path.replace(move.new_path)
    move.old_thumbnail_path.replace(move.new_thumbnail_path)

    recovered = service.startup_recovery()

    assert recovered.committed == 1
    assert recovered.errors == 0
    assert not old_parent.exists()
    assert not old_thumb_parent.exists()
    assert service.get_job(job_id).state == "committed"


def test_preflight_rejects_active_uncommitted_job_for_same_image(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-d.png"
    _save_record(database, records, image_name=image_name, subfolder="", created_at="2024-01-02 03:04:05.000")
    service.image_files.save(Image.new("RGB", (16, 16), "yellow"), image_name=image_name)

    moves = service.plan_batch(last_image_name="", limit=100)
    service.create_move_job(moves)

    with pytest.raises(ValueError, match="active image move job"):
        service.plan_batch(last_image_name="", limit=100)


def test_create_move_job_rejects_second_active_job_from_stale_plan(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-active-race.png"
    _save_image(database, service, records, image_name, "", "2024-01-03 03:04:05.000", "yellow")

    stale_plan_a = service.plan_batch(last_image_name="", limit=100)
    stale_plan_b = service.plan_batch(last_image_name="", limit=100)
    service.create_move_job(stale_plan_a)

    with pytest.raises(ValueError, match="active image move job"):
        service.create_move_job(stale_plan_b)


def test_startup_recovery_completes_planned_job_before_any_file_move(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-e.png"
    _save_image(database, service, records, image_name, "", "2024-03-04 05:06:07.000", "purple")

    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)

    recovered_once = service.startup_recovery()
    recovered_twice = service.startup_recovery()

    assert recovered_once.committed == 1
    assert recovered_once.errors == 0
    assert recovered_twice.committed == 0
    assert recovered_twice.errors == 0
    assert records.get(image_name).image_subfolder == "2024/03/04"
    assert service.get_job(job_id).state == "committed"
    assert _job_item_states(database, job_id) == {image_name: "committed"}


def test_background_recovery_can_start_when_journal_job_is_active(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-background-recovery.png"
    _save_image(database, service, records, image_name, "", "2024-03-05 05:06:07.000", "purple")
    job_id = service.create_move_job(service.plan_batch(last_image_name="", limit=100))

    status = service.start_background_recovery()
    # Not `is_running`: that samples whether the worker happens to still be going, and on a loaded
    # machine the recovery can finish before the status is read. What the active journal job must
    # not do is prevent recovery from being started at all, which the rest of this test observes.
    assert status.operation == "recovery"

    assert service._future is not None
    service._future.result(timeout=5)

    assert records.get(image_name).image_subfolder == "2024/03/05"
    assert service.get_job(job_id).state == "committed"


def test_start_runs_recovery_before_normal_operation(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-startup-recovery.png"
    _save_image(database, service, records, image_name, "", "2024-03-05 05:06:07.000", "purple")
    job_id = service.create_move_job(service.plan_batch(last_image_name="", limit=100))
    service.perform_filesystem_moves(job_id)

    service.start(MagicMock())

    assert records.get(image_name).image_subfolder == "2024/03/05"
    assert service.get_job(job_id).state == "committed"
    assert service.is_maintenance_active() is False


def test_start_leaves_maintenance_active_when_recovery_remains_incomplete(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-startup-recovery-retry.png"
    _save_image(database, service, records, image_name, "", "2024-03-05 05:06:07.000", "purple")
    service.create_move_job(service.plan_batch(last_image_name="", limit=100))

    with patch.object(service, "complete_partial_filesystem_moves", side_effect=OSError("temporary failure")):
        service.start(MagicMock())

    assert records.get(image_name).image_subfolder == ""
    assert service.is_maintenance_active() is True


@pytest.mark.parametrize(("pending", "in_progress"), [(1, 0), (0, 1)])
def test_background_move_rejects_active_queue_work(
    tmp_path: Path, database: Database, pending: int, in_progress: int
) -> None:
    service, _records = _service(tmp_path, database, strategy="date")
    invoker = MagicMock()
    invoker.services.session_queue.has_active_queue_work.return_value = pending > 0 or in_progress > 0
    service.start(invoker)

    with pytest.raises(ImageMoveQueueActive, match="queue work is active"):
        service.start_background_move_all()


def test_background_move_is_reserved_before_queue_check(tmp_path: Path, database: Database) -> None:
    service, _records = _service(tmp_path, database, strategy="date")
    invoker = MagicMock()

    def has_active_queue_work() -> bool:
        assert service.is_maintenance_active() is True
        return True

    invoker.services.session_queue.has_active_queue_work.side_effect = has_active_queue_work
    service.start(invoker)

    with pytest.raises(ImageMoveQueueActive, match="queue work is active"):
        service.start_background_move_all()

    assert service.is_maintenance_active() is False


def test_gallery_maintenance_reservation_blocks_moves_and_releases_after_failure(
    tmp_path: Path, database: Database
) -> None:
    service, _records = _service(tmp_path, database, strategy="date")

    try:
        with service.reserve_gallery_maintenance():
            assert service.is_maintenance_active() is True
            with pytest.raises(ImageMoveJobAlreadyRunning, match="maintenance"):
                service.start_background_recovery()

        assert service.is_maintenance_active() is False

        with pytest.raises(RuntimeError, match="operation failed"):
            with service.reserve_gallery_maintenance():
                raise RuntimeError("operation failed")

        assert service.is_maintenance_active() is False
    finally:
        service.stop()


def test_gallery_maintenance_reservation_refuses_active_queue_and_releases_flag(
    tmp_path: Path, database: Database
) -> None:
    service, _records = _service(tmp_path, database, strategy="date")
    invoker = MagicMock()

    def has_active_queue_work() -> bool:
        # The operation reservation must be visible before it reads queue state.
        assert service.is_maintenance_active() is True
        return True

    invoker.services.session_queue.has_active_queue_work.side_effect = has_active_queue_work
    service.start(invoker)

    try:
        with pytest.raises(ImageMoveQueueActive, match="queue work is active"):
            with service.reserve_gallery_maintenance():
                pytest.fail("maintenance must not begin with pending queue work")

        assert service.is_maintenance_active() is False
    finally:
        service.stop()


@pytest.mark.parametrize("status", ["pending", "in_progress"])
def test_gallery_maintenance_reservation_refuses_active_work_on_custom_queue(
    tmp_path: Path, database: Database, status: str
) -> None:
    service, _records = _service(tmp_path, database, strategy="date")
    with database.begin(write=True) as conn:
        conn.execute(
            insert(session_queue).values(
                batch_id="custom-batch",
                queue_id="custom-queue",
                session_id="custom-session",
                session="{}",
                status=status,
            )
        )
    service.set_session_queue(SessionQueue(database))

    try:
        with pytest.raises(ImageMoveQueueActive, match="queue work is active"):
            with service.reserve_gallery_maintenance():
                pytest.fail("maintenance must not begin while a custom queue has pending work")

        assert service.is_maintenance_active() is False
    finally:
        service.stop()


def test_maintenance_is_active_while_background_job_or_uncommitted_journal_exists(
    tmp_path: Path, database: Database
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-maintenance-active.png"
    _save_image(database, service, records, image_name, "", "2024-03-05 05:06:07.000", "purple")
    service.create_move_job(service.plan_batch(last_image_name="", limit=100))

    assert service.is_maintenance_active() is True

    release_worker = threading.Event()

    def wait_for_release() -> None:
        release_worker.wait(timeout=5)

    service._start_background_operation("recovery", wait_for_release)
    try:
        assert service.is_maintenance_active() is True
    finally:
        release_worker.set()
        assert service._future is not None
        service._future.result(timeout=5)


def test_background_worker_error_is_exposed_in_status(tmp_path: Path, database: Database) -> None:
    service, _records = _service(tmp_path, database, strategy="date")
    started_worker = threading.Event()
    release_worker = threading.Event()

    def raise_error() -> None:
        started_worker.set()
        release_worker.wait(timeout=5)
        raise RuntimeError("background failed")

    status = service._start_background_operation("move_all", raise_error)
    assert started_worker.wait(timeout=5) is True
    assert status.is_running is True

    assert service._future is not None
    release_worker.set()
    service._future.result(timeout=5)

    status = service.get_background_status()
    assert status.is_running is False
    assert status.operation is None
    assert status.last_error == "background failed"


def test_stop_waits_for_active_background_job_without_recording_error(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-background-stop.png"
    _save_image(database, service, records, image_name, "", "2024-03-05 05:06:07.000", "purple")
    job_id = service.create_move_job(service.plan_batch(last_image_name="", limit=100))
    release_worker = threading.Event()

    def wait_for_shutdown() -> None:
        release_worker.wait(timeout=5)

    service._start_background_operation("recovery", wait_for_shutdown)

    stop_thread = threading.Thread(target=service.stop)
    stop_thread.start()
    assert stop_thread.is_alive()

    release_worker.set()
    stop_thread.join(timeout=5)

    assert not stop_thread.is_alive()
    assert service.get_job(job_id).error_message is None
    assert service.get_background_status().last_error is None


def test_startup_recovery_completes_partial_multi_image_move(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    _save_image(database, service, records, "image-f.png", "", "2024-04-05 06:07:08.000", "orange")
    _save_image(database, service, records, "image-g.png", "", "2024-04-06 06:07:08.000", "cyan")

    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    first_move = moves[0]
    first_move.new_path.parent.mkdir(parents=True, exist_ok=True)
    first_move.new_thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    first_move.old_path.replace(first_move.new_path)
    first_move.old_thumbnail_path.replace(first_move.new_thumbnail_path)

    recovered_once = service.startup_recovery()
    recovered_twice = service.startup_recovery()

    assert recovered_once.committed == 2
    assert recovered_once.errors == 0
    assert recovered_twice.committed == 0
    assert recovered_twice.errors == 0
    assert records.get("image-f.png").image_subfolder == "2024/04/05"
    assert records.get("image-g.png").image_subfolder == "2024/04/06"
    assert service.get_job(job_id).state == "committed"
    assert _job_item_states(database, job_id) == {"image-f.png": "committed", "image-g.png": "committed"}


def test_startup_recovery_marks_committed_after_db_update_but_before_journal_commit(
    tmp_path: Path, database: Database
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-h.png"
    _save_image(database, service, records, image_name, "", "2024-05-06 07:08:09.000", "pink")
    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    service.perform_filesystem_moves(job_id)

    _set_record(database, image_name, image_subfolder="2024/05/06")

    recovered_once = service.startup_recovery()
    recovered_twice = service.startup_recovery()

    assert recovered_once.committed == 1
    assert recovered_once.errors == 0
    assert recovered_twice.committed == 0
    assert recovered_twice.errors == 0
    assert records.get(image_name).image_subfolder == "2024/05/06"
    assert service.get_job(job_id).state == "committed"
    assert _job_item_states(database, job_id) == {image_name: "committed"}


def test_startup_recovery_marks_error_when_both_old_and_new_full_size_files_exist(
    tmp_path: Path, database: Database
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-i.png"
    _save_image(database, service, records, image_name, "", "2024-07-08 09:10:11.000", "red")
    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    move = moves[0]
    move.new_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(move.old_path, move.new_path)

    recovered = service.startup_recovery()

    assert recovered.committed == 0
    assert recovered.errors == 1
    assert records.get(image_name).image_subfolder == ""
    assert service.get_job(job_id).state == "error"
    assert _job_item_states(database, job_id) == {image_name: "error"}


def test_startup_recovery_marks_error_when_neither_old_nor_new_full_size_file_exists(
    tmp_path: Path, database: Database
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-j.png"
    _save_image(database, service, records, image_name, "", "2024-08-09 10:11:12.000", "blue")
    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    moves[0].old_path.unlink()

    recovered = service.startup_recovery()

    assert recovered.committed == 0
    assert recovered.errors == 1
    assert records.get(image_name).image_subfolder == ""
    assert service.get_job(job_id).state == "error"
    assert _job_item_states(database, job_id) == {image_name: "error"}


def test_startup_recovery_isolates_corrupt_image_and_commits_other_items(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    good_image_name = "image-a-good.png"
    corrupt_image_name = "image-b-corrupt.png"
    _save_image(database, service, records, good_image_name, "", "2024-09-10 11:12:13.000", "white")
    _save_image(database, service, records, corrupt_image_name, "", "2024-09-11 11:12:13.000", "black")

    moves = service.plan_batch(last_image_name="", limit=100)
    corrupt_move = next(move for move in moves if move.image_name == corrupt_image_name)
    corrupt_move.old_thumbnail_path.unlink()
    corrupt_move.old_path.write_bytes(b"")
    job_id = service.create_move_job(moves)

    recovered = service.startup_recovery()

    assert recovered.committed == 1
    assert recovered.errors == 1
    assert service.is_maintenance_active() is False
    assert service.get_job(job_id).state == "error"
    assert corrupt_image_name in (service.get_job(job_id).error_message or "")
    assert _job_item_states(database, job_id) == {
        good_image_name: "committed",
        corrupt_image_name: "error",
    }
    assert records.get(good_image_name).image_subfolder == "2024/09/10"
    assert records.get(corrupt_image_name).image_subfolder == ""
    assert service.image_files.get_path(good_image_name, image_subfolder="2024/09/10").exists()
    assert not service.image_files.get_path(good_image_name, image_subfolder="").exists()
    assert corrupt_move.old_path.exists()
    assert not corrupt_move.new_path.exists()


def test_startup_recovery_marks_truncated_image_error_without_retrying_forever(
    tmp_path: Path, database: Database
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-truncated.png"
    _save_image(database, service, records, image_name, "", "2024-09-12 11:12:13.000", "white")
    move = service.plan_batch(last_image_name="", limit=100)[0]
    move.old_thumbnail_path.unlink()
    job_id = service.create_move_job([move])

    with patch.object(
        service,
        "_regenerate_thumbnail",
        side_effect=UnreadableImageError("image file is truncated (0 bytes not processed)"),
    ):
        recovered = service.startup_recovery()

    assert recovered.committed == 0
    assert recovered.errors == 1
    assert service.is_maintenance_active() is False
    assert service.get_job(job_id).state == "error"
    assert _job_item_states(database, job_id) == {image_name: "error"}
    assert records.get(image_name).image_subfolder == ""
    assert move.old_path.exists()
    assert not move.new_path.exists()


def test_move_all_images_marks_corrupt_idat_as_unrecoverable(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-corrupt-idat.png"
    _save_image(database, service, records, image_name, "", "2024-09-16 11:12:13.000", "white")
    move = service.plan_batch(last_image_name="", limit=100)[0]
    move.old_thumbnail_path.unlink()
    _corrupt_png_idat(move.old_path)

    result = service.move_all_images()

    assert result.committed == 0
    assert result.errors == 1
    assert service.is_maintenance_active() is False
    assert service.get_latest_job().state == "error"
    assert records.get(image_name).image_subfolder == ""
    assert move.old_path.exists()
    assert not move.new_path.exists()


def test_move_all_images_counts_and_reports_each_unrecoverable_item(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_names = ["image-corrupt-a.png", "image-corrupt-b.png"]
    for index, image_name in enumerate(image_names):
        _save_image(
            database,
            service,
            records,
            image_name,
            "",
            f"2024-09-{17 + index:02d} 11:12:13.000",
            "white",
        )

    moves = service.plan_batch(last_image_name="", limit=100)
    for move in moves:
        move.old_thumbnail_path.unlink()
        _corrupt_png_idat(move.old_path)

    result = service.move_all_images()

    assert result.committed == 0
    assert result.errors == len(image_names)
    error_message = service.get_latest_job().error_message or ""
    assert all(image_name in error_message for image_name in image_names)
    assert all(records.get(image_name).image_subfolder == "" for image_name in image_names)


def test_startup_recovery_keeps_db_consistent_when_thumbnail_regeneration_fails(
    tmp_path: Path, database: Database
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-thumbnail-recovery.png"
    _save_image(database, service, records, image_name, "", "2024-09-13 11:12:13.000", "white")
    move = service.plan_batch(last_image_name="", limit=100)[0]
    move.new_thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(move.old_thumbnail_path, move.new_thumbnail_path)
    job_id = service.create_move_job([move])

    with patch.object(
        service,
        "_regenerate_thumbnail",
        side_effect=UnreadableImageError("image file is truncated (0 bytes not processed)"),
    ):
        recovered = service.startup_recovery()

    assert recovered.committed == 0
    assert recovered.errors == 1
    assert service.get_job(job_id).state == "error"
    assert records.get(image_name).image_subfolder == ""
    assert move.old_path.exists()
    assert not move.new_path.exists()


@pytest.mark.parametrize(("repointed_meanwhile", "expected_subfolder"), [(False, "2024/09/15"), (True, "elsewhere")])
def test_startup_recovery_reconciles_db_after_destination_thumbnail_failure(
    tmp_path: Path, database: Database, repointed_meanwhile: bool, expected_subfolder: str
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-destination-recovery.png"
    _save_image(database, service, records, image_name, "", "2024-09-15 11:12:13.000", "white")
    move = service.plan_batch(last_image_name="", limit=100)[0]
    move.old_thumbnail_path.unlink()
    move.new_path.parent.mkdir(parents=True, exist_ok=True)
    move.old_path.replace(move.new_path)
    job_id = service.create_move_job([move])
    if repointed_meanwhile:
        # Reconciling repoints a record only while it still names the old subfolder.
        _set_record(database, image_name, image_subfolder="elsewhere")

    with patch.object(
        service,
        "_regenerate_thumbnail",
        side_effect=UnreadableImageError("image file is truncated (0 bytes not processed)"),
    ):
        recovered = service.startup_recovery()

    assert recovered.committed == 0
    assert recovered.errors == 1
    assert service.is_maintenance_active() is False
    assert service.get_job(job_id).state == "error"
    assert _job_item_states(database, job_id) == {image_name: "error"}
    assert records.get(image_name).image_subfolder == expected_subfolder
    assert not move.old_path.exists()
    assert move.new_path.exists()


def test_startup_recovery_finalizes_job_after_item_error_was_recorded(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-recorded-error.png"
    _save_image(database, service, records, image_name, "", "2024-09-14 11:12:13.000", "white")
    move = service.plan_batch(last_image_name="", limit=100)[0]
    job_id = service.create_move_job([move])
    service.mark_item_unrecoverable(job_id, image_name, "image is corrupt")

    recovered = service.startup_recovery()

    assert recovered.committed == 0
    assert recovered.errors == 1
    assert service.is_maintenance_active() is False
    assert service.get_job(job_id).state == "error"
    assert _job_item_states(database, job_id) == {image_name: "error"}


def test_startup_recovery_keeps_job_recoverable_after_ordinary_exception(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-k.png"
    _save_image(database, service, records, image_name, "", "2024-09-10 11:12:13.000", "white")
    job_id = service.create_move_job(service.plan_batch(last_image_name="", limit=100))

    with patch.object(service, "complete_partial_filesystem_moves", side_effect=OSError("temporary failure")):
        recovered = service.startup_recovery()

    assert recovered.committed == 0
    assert recovered.errors == 1
    job = service.get_job(job_id)
    assert job.state == "planned"
    assert job.error_message == "temporary failure"

    recovered_retry = service.startup_recovery()

    assert recovered_retry.committed == 1
    assert recovered_retry.errors == 0
    assert records.get(image_name).image_subfolder == "2024/09/10"
    assert service.get_job(job_id).state == "committed"


def test_startup_recovery_regenerates_thumbnail_when_old_and_new_thumbnails_exist(
    tmp_path: Path, database: Database
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-l.png"
    _save_image(database, service, records, image_name, "", "2024-10-11 12:13:14.000", "black")
    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    move = moves[0]
    move.new_path.parent.mkdir(parents=True, exist_ok=True)
    move.new_thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    move.old_path.replace(move.new_path)
    copy2(move.old_thumbnail_path, move.new_thumbnail_path)

    recovered = service.startup_recovery()

    assert recovered.committed == 1
    assert recovered.errors == 0
    assert records.get(image_name).image_subfolder == "2024/10/11"
    assert move.new_thumbnail_path.exists()
    assert not move.old_thumbnail_path.exists()
    assert service.get_job(job_id).state == "committed"


def test_preflight_rejects_duplicate_thumbnail_destination_paths(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    _save_image(database, service, records, "same-name.jpg", "", "2024-12-13 14:15:16.000", "red")
    _save_image(database, service, records, "same-name.png", "", "2024-12-13 14:15:16.000", "green")

    with pytest.raises(ValueError, match="Duplicate destination thumbnail path"):
        service.plan_batch(last_image_name="", limit=100)


def test_successful_filesystem_move_fsyncs_files_and_directories(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    image_name = "image-m.png"
    _save_image(database, service, records, image_name, "", "2025-01-02 03:04:05.000", "blue")
    job_id = service.create_move_job(service.plan_batch(last_image_name="", limit=100))

    with (
        patch.object(service, "_fsync_file") as fsync_file,
        patch.object(service, "_fsync_dir") as fsync_dir,
    ):
        service.perform_filesystem_moves(job_id)

    moved = service._get_items(job_id)[0]
    fsync_file.assert_any_call(moved.new_path)
    fsync_file.assert_any_call(moved.new_thumbnail_path)
    fsync_dir.assert_any_call(moved.new_path.parent)
    fsync_dir.assert_any_call(moved.old_path.parent)
    fsync_dir.assert_any_call(moved.new_thumbnail_path.parent)
    fsync_dir.assert_any_call(moved.old_thumbnail_path.parent)


def test_fsync_dir_ignores_platform_close_failures(tmp_path: Path, database: Database) -> None:
    service, _records = _service(tmp_path, database, strategy="date")

    with (
        patch("invokeai.app.services.image_moves.image_moves_default.os.open", return_value=123),
        patch(
            "invokeai.app.services.image_moves.image_moves_default.os.fsync",
            side_effect=OSError(9, "Bad file descriptor"),
        ),
        patch(
            "invokeai.app.services.image_moves.image_moves_default.os.close",
            side_effect=OSError(9, "Bad file descriptor"),
        ),
    ):
        service._fsync_dir(tmp_path)


def test_fsync_file_ignores_platform_fsync_failures(tmp_path: Path, database: Database) -> None:
    service, _records = _service(tmp_path, database, strategy="date")
    path = tmp_path / "image.png"
    path.write_bytes(b"test")

    with patch(
        "invokeai.app.services.image_moves.image_moves_default.os.fsync",
        side_effect=OSError(9, "Bad file descriptor"),
    ):
        service._fsync_file(path)


def test_commit_rolls_back_every_repoint_when_a_record_was_repointed_meanwhile(
    tmp_path: Path, database: Database
) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    # The same day, so that one statement repoints both.
    _save_image(database, service, records, "image-kept.png", "", "2024-03-01 01:02:03.000", "red")
    _save_image(database, service, records, "image-raced.png", "", "2024-03-01 04:05:06.000", "blue")
    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    service.perform_filesystem_moves(job_id)
    _set_record(database, "image-raced.png", image_subfolder="elsewhere")

    with pytest.raises(RuntimeError, match="failed commit validation"):
        service.commit_database_updates(job_id)

    # The commit repoints a record only while it still names the old subfolder, and validates the job as a whole.
    assert records.get("image-raced.png").image_subfolder == "elsewhere"
    assert records.get("image-kept.png").image_subfolder == ""
    assert _job_item_states(database, job_id) == {"image-kept.png": "moved", "image-raced.png": "moved"}
    # Records and files disagree until the job is finished, so it stays active.
    assert service.get_job(job_id).state == "moved"
    assert service.is_maintenance_active()
    with pytest.raises(ValueError, match="active image move job"):
        service.create_move_job(moves)


def test_commit_repoints_each_image_to_its_own_subfolder(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    _save_image(database, service, records, "image-1.png", "", "2024-03-01 01:02:03.000", "red")
    _save_image(database, service, records, "image-2.png", "", "2024-03-01 04:05:06.000", "blue")
    _save_image(database, service, records, "image-3.png", "old", "2024-03-01 07:08:09.000", "green")
    _save_image(database, service, records, "image-4.png", "", "2024-03-02 01:02:03.000", "pink")
    _save_image(database, service, records, "image-5.png", "2024/03/03", "2024-03-03 01:02:03.000", "gray")
    job_id = service.create_move_job(service.plan_batch(last_image_name="", limit=100))
    service.perform_filesystem_moves(job_id)

    assert service.commit_database_updates(job_id) == 4

    names = [f"image-{i}.png" for i in range(1, 6)]
    assert {name: records.get(name).image_subfolder for name in names} == {
        "image-1.png": "2024/03/01",
        "image-2.png": "2024/03/01",
        "image-3.png": "2024/03/01",
        "image-4.png": "2024/03/02",
        "image-5.png": "2024/03/03",
    }


def test_commit_rejects_a_moved_image_whose_record_was_deleted(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    _save_image(database, service, records, "image-deleted.png", "", "2024-03-03 01:02:03.000", "red")
    job_id = service.create_move_job(service.plan_batch(last_image_name="", limit=100))
    service.perform_filesystem_moves(job_id)
    _set_record(database, "image-deleted.png", deleted_at="2024-03-04 00:00:00.000")

    with pytest.raises(RuntimeError, match="failed commit validation"):
        service.commit_database_updates(job_id)

    assert service.get_job(job_id).state == "moved"


def test_deleted_records_are_neither_counted_nor_planned(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    _save_image(database, service, records, "a-deleted.png", "", "2024-03-05 01:02:03.000", "red")
    _save_image(database, service, records, "b-live.png", "", "2024-03-06 01:02:03.000", "blue")
    _set_record(database, "a-deleted.png", deleted_at="2024-03-07 00:00:00.000")

    assert service.count_images_needing_move() == 1
    assert [move.image_name for move in service.plan_batch(last_image_name="", limit=100)] == ["b-live.png"]
    assert database.queries.image_moves.next_image_name("") == "b-live.png"


def test_finished_jobs_are_not_recovered(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    _save_image(database, service, records, "image-f.png", "", "2024-03-08 01:02:03.000", "red")
    move = service.plan_batch(last_image_name="", limit=100)[0]
    failed_job_id = service.create_error_move_job(move, "source vanished")

    assert database.queries.image_moves.recoverable_job_ids() == []
    job_id = service.create_move_job([move])
    assert database.queries.image_moves.recoverable_job_ids() == [job_id]
    assert service.get_latest_job() == ImageMoveJob(id=job_id, state="planned", error_message=None)

    service.record_job_error_message(job_id, "temporary failure")
    service.move_all_images()

    assert database.queries.image_moves.recoverable_job_ids() == []
    assert service.get_job(failed_job_id).error_message == "source vanished"
    # A committed job no longer reports what went wrong before.
    assert service.get_job(job_id) == ImageMoveJob(id=job_id, state="committed", error_message=None)


def test_unrecoverable_job_fails_every_item_and_releases_its_images(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    _save_image(database, service, records, "image-g.png", "", "2024-03-09 01:02:03.000", "red")
    _save_image(database, service, records, "image-h.png", "", "2024-03-10 01:02:03.000", "blue")
    job_id = service.create_move_job(service.plan_batch(last_image_name="", limit=100))

    service.mark_job_unrecoverable(job_id, "disk gone")

    assert service.get_job(job_id) == ImageMoveJob(id=job_id, state="error", error_message="disk gone")
    assert _job_item_states(database, job_id) == {"image-g.png": "error", "image-h.png": "error"}
    assert not service.is_maintenance_active()
    assert len(service.plan_batch(last_image_name="", limit=100)) == 2


def test_commit_reports_item_errors_in_image_name_order(tmp_path: Path, database: Database) -> None:
    service, records = _service(tmp_path, database, strategy="date")
    _save_image(database, service, records, "image-i.png", "", "2024-03-11 01:02:03.000", "red")
    _save_image(database, service, records, "image-j.png", "", "2024-03-12 01:02:03.000", "blue")
    # Journaled out of name order, so that only the commit's ordering puts the messages in it.
    job_id = service.create_move_job(list(reversed(service.plan_batch(last_image_name="", limit=100))))
    service.mark_item_unrecoverable(job_id, "image-j.png", "second")
    service.mark_item_unrecoverable(job_id, "image-i.png", "first")

    assert service.commit_database_updates(job_id) == 0

    assert service.get_job(job_id) == ImageMoveJob(id=job_id, state="error", error_message="first\nsecond")
    # Failed items moved nothing, so their records stay where they were.
    assert records.get("image-i.png").image_subfolder == ""
    assert records.get("image-j.png").image_subfolder == ""
