from pathlib import Path
from shutil import copy2
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_moves.image_moves_default import ImageMoveService
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.util.logging import InvokeAILogger


def _build_db(tmp_path: Path) -> SqliteDatabase:
    logger = InvokeAILogger.get_logger()
    config = InvokeAIAppConfig(use_memory_db=False)
    config._root = tmp_path
    image_files = DiskImageFileStorage(tmp_path / "images")
    return init_db(config=config, logger=logger, image_files=image_files)


def _save_record(records: SqliteImageRecordStorage, image_name: str, subfolder: str, created_at: str) -> None:
    records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=16,
        height=16,
        has_workflow=False,
        image_subfolder=subfolder,
    )
    with records._db.transaction() as cursor:
        cursor.execute("UPDATE images SET created_at = ? WHERE image_name = ?;", (created_at, image_name))


def _save_image(
    service: ImageMoveService,
    records: SqliteImageRecordStorage,
    image_name: str,
    subfolder: str,
    created_at: str,
    color: str,
) -> None:
    _save_record(records, image_name=image_name, subfolder=subfolder, created_at=created_at)
    service.image_files.save(Image.new("RGB", (16, 16), color), image_name=image_name, image_subfolder=subfolder)


def _service(tmp_path: Path, strategy: str = "date") -> tuple[ImageMoveService, SqliteImageRecordStorage]:
    db = _build_db(tmp_path)
    records = SqliteImageRecordStorage(db=db)
    storage = DiskImageFileStorage(tmp_path / "images")
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 6
    storage.start(invoker)
    config = InvokeAIAppConfig(use_memory_db=True, image_subfolder_strategy=strategy)
    config._root = tmp_path
    service = ImageMoveService(db=db, image_files=storage, config=config, logger=InvokeAILogger.get_logger())
    return service, records


def _job_item_states(service: ImageMoveService, job_id: int) -> dict[str, str]:
    with service._db.transaction() as cursor:
        cursor.execute(
            "SELECT image_name, state FROM image_subfolder_move_items WHERE job_id = ? ORDER BY image_name;",
            (job_id,),
        )
        return {row["image_name"]: row["state"] for row in cursor.fetchall()}


def test_move_all_images_uses_created_at_for_date_strategy(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-a.png"
    _save_record(records, image_name=image_name, subfolder="", created_at="2024-02-03 04:05:06.000")
    service.image_files.save(Image.new("RGB", (16, 16), "red"), image_name=image_name)

    result = service.move_all_images()

    assert result.planned == 1
    assert result.committed == 1
    record = records.get(image_name)
    assert record.image_subfolder == "2024/02/03"
    assert service.image_files.get_path(image_name, image_subfolder="2024/02/03").exists()
    assert not service.image_files.get_path(image_name, image_subfolder="").exists()


def test_startup_recovery_commits_after_files_moved_but_db_not_updated(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-b.png"
    _save_record(records, image_name=image_name, subfolder="", created_at="2025-06-07 08:09:10.000")
    service.image_files.save(Image.new("RGB", (16, 16), "blue"), image_name=image_name)

    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    service.perform_filesystem_moves(job_id)

    assert records.get(image_name).image_subfolder == ""

    recovered = service.startup_recovery()

    assert recovered.committed == 1
    assert records.get(image_name).image_subfolder == "2025/06/07"
    assert service.get_job(job_id).state == "committed"


def test_cleanup_empty_source_directories_after_move(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-c.png"
    old_subfolder = "old/nested"
    _save_record(records, image_name=image_name, subfolder=old_subfolder, created_at="2024-11-12 01:02:03.000")
    service.image_files.save(Image.new("RGB", (16, 16), "green"), image_name=image_name, image_subfolder=old_subfolder)
    old_parent = service.image_files.get_path(image_name, image_subfolder=old_subfolder).parent
    old_thumb_parent = service.image_files.get_path(image_name, thumbnail=True, image_subfolder=old_subfolder).parent

    service.move_all_images()

    assert not old_parent.exists()
    assert not old_thumb_parent.exists()
    assert service.image_files.image_root.exists()
    assert service.image_files.thumbnail_root.exists()


def test_preflight_rejects_active_uncommitted_job_for_same_image(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-d.png"
    _save_record(records, image_name=image_name, subfolder="", created_at="2024-01-02 03:04:05.000")
    service.image_files.save(Image.new("RGB", (16, 16), "yellow"), image_name=image_name)

    moves = service.plan_batch(last_image_name="", limit=100)
    service.create_move_job(moves)

    with pytest.raises(ValueError, match="active image move job"):
        service.plan_batch(last_image_name="", limit=100)


def test_create_move_job_rejects_second_active_job_from_stale_plan(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-active-race.png"
    _save_image(service, records, image_name, "", "2024-01-03 03:04:05.000", "yellow")

    stale_plan_a = service.plan_batch(last_image_name="", limit=100)
    stale_plan_b = service.plan_batch(last_image_name="", limit=100)
    service.create_move_job(stale_plan_a)

    with pytest.raises(ValueError, match="active image move job"):
        service.create_move_job(stale_plan_b)


def test_startup_recovery_completes_planned_job_before_any_file_move(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-e.png"
    _save_image(service, records, image_name, "", "2024-03-04 05:06:07.000", "purple")

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
    assert _job_item_states(service, job_id) == {image_name: "committed"}


def test_startup_recovery_completes_partial_multi_image_move(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    _save_image(service, records, "image-f.png", "", "2024-04-05 06:07:08.000", "orange")
    _save_image(service, records, "image-g.png", "", "2024-04-06 06:07:08.000", "cyan")

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
    assert _job_item_states(service, job_id) == {"image-f.png": "committed", "image-g.png": "committed"}


def test_startup_recovery_marks_committed_after_db_update_but_before_journal_commit(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-h.png"
    _save_image(service, records, image_name, "", "2024-05-06 07:08:09.000", "pink")
    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    service.perform_filesystem_moves(job_id)

    with service._db.transaction() as cursor:
        cursor.execute(
            "UPDATE images SET image_subfolder = ? WHERE image_name = ?;",
            ("2024/05/06", image_name),
        )

    recovered_once = service.startup_recovery()
    recovered_twice = service.startup_recovery()

    assert recovered_once.committed == 1
    assert recovered_once.errors == 0
    assert recovered_twice.committed == 0
    assert recovered_twice.errors == 0
    assert records.get(image_name).image_subfolder == "2024/05/06"
    assert service.get_job(job_id).state == "committed"
    assert _job_item_states(service, job_id) == {image_name: "committed"}


def test_startup_recovery_marks_error_when_both_old_and_new_full_size_files_exist(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-i.png"
    _save_image(service, records, image_name, "", "2024-07-08 09:10:11.000", "red")
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
    assert _job_item_states(service, job_id) == {image_name: "error"}


def test_startup_recovery_marks_error_when_neither_old_nor_new_full_size_file_exists(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-j.png"
    _save_image(service, records, image_name, "", "2024-08-09 10:11:12.000", "blue")
    moves = service.plan_batch(last_image_name="", limit=100)
    job_id = service.create_move_job(moves)
    moves[0].old_path.unlink()

    recovered = service.startup_recovery()

    assert recovered.committed == 0
    assert recovered.errors == 1
    assert records.get(image_name).image_subfolder == ""
    assert service.get_job(job_id).state == "error"
    assert _job_item_states(service, job_id) == {image_name: "error"}


def test_startup_recovery_keeps_job_recoverable_after_ordinary_exception(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-k.png"
    _save_image(service, records, image_name, "", "2024-09-10 11:12:13.000", "white")
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


def test_startup_recovery_regenerates_thumbnail_when_old_and_new_thumbnails_exist(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-l.png"
    _save_image(service, records, image_name, "", "2024-10-11 12:13:14.000", "black")
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


def test_preflight_rejects_duplicate_thumbnail_destination_paths(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    _save_image(service, records, "same-name.jpg", "", "2024-12-13 14:15:16.000", "red")
    _save_image(service, records, "same-name.png", "", "2024-12-13 14:15:16.000", "green")

    with pytest.raises(ValueError, match="Duplicate destination thumbnail path"):
        service.plan_batch(last_image_name="", limit=100)


def test_successful_filesystem_move_fsyncs_files_and_directories(tmp_path: Path) -> None:
    service, records = _service(tmp_path, strategy="date")
    image_name = "image-m.png"
    _save_image(service, records, image_name, "", "2025-01-02 03:04:05.000", "blue")
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
