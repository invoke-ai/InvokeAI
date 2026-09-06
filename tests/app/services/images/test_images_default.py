"""Tests for ImageService (images_default.py).

Covers subfolder forwarding for all strategies, the delete_images_on_board
silent-failure contract (Points 2 & 3 from PR review), and the transactional
staged-deletion contracts of delete() and delete_intermediates().
"""

import errno
import os
import stat
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileSaveException,
)
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_moves.image_moves_default import ImageMoveResult, ImageMoveService
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ResourceOrigin,
)
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.app.util.misc import get_iso_timestamp
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def image_service() -> ImageService:
    svc = ImageService()
    invoker = MagicMock()

    # Wire up service references
    invoker.services.names.create_image_name.return_value = "abc12345-test.png"
    invoker.services.image_records.get.return_value = _make_record(image_subfolder="some/sub")
    invoker.services.board_image_records.get_board_for_image.return_value = None
    invoker.services.urls.get_image_url.return_value = "http://localhost/img.png"
    invoker.services.configuration.image_subfolder_strategy = "flat"
    # By default every named intermediate is still an intermediate when the delete runs.
    invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: list(names)

    svc.start(invoker)
    return svc


def _make_record(
    image_name: str = "abc12345-test.png",
    image_subfolder: str = "",
    is_intermediate: bool = False,
) -> ImageRecord:
    now = get_iso_timestamp()
    return ImageRecord(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        created_at=now,
        updated_at=now,
        is_intermediate=is_intermediate,
        starred=False,
        has_workflow=False,
        image_subfolder=image_subfolder,
    )


@pytest.fixture
def real_image_service(tmp_path: Path) -> tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage]:
    logger = InvokeAILogger.get_logger()
    config = InvokeAIAppConfig(use_memory_db=True, image_subfolder_strategy="flat")
    config._root = tmp_path
    storage = DiskImageFileStorage(tmp_path / "images")
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 6
    storage.start(invoker)
    db = init_db(config=config, logger=logger, image_files=storage)
    records = SqliteImageRecordStorage(db=db)

    invoker.services.configuration.image_subfolder_strategy = "flat"
    invoker.services.names.create_image_name.return_value = "uploaded.png"
    invoker.services.image_records = records
    invoker.services.image_files = storage
    invoker.services.board_image_records.get_board_for_image.return_value = None
    invoker.services.urls.get_image_url.return_value = "/api/v1/images/i/uploaded.png"
    invoker.services.logger = MagicMock()

    service = ImageService()
    service.start(invoker)
    return service, records, storage


def test_create_rolls_back_record_and_files_when_thumbnail_save_fails(
    real_image_service: tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage],
) -> None:
    service, records, storage = real_image_service
    image = Image.new("RGB", (32, 32), "red")
    broken_thumbnail = MagicMock()
    broken_thumbnail.save.side_effect = OSError("thumbnail filesystem failure")

    try:
        with patch(
            "invokeai.app.services.image_files.image_files_disk.make_thumbnail",
            return_value=broken_thumbnail,
        ):
            with pytest.raises(ImageFileSaveException):
                service.create(
                    image=image,
                    image_origin=ResourceOrigin.EXTERNAL,
                    image_category=ImageCategory.GENERAL,
                )

        with pytest.raises(ImageRecordNotFoundException):
            records.get("uploaded.png")
        assert not storage.get_path("uploaded.png").exists()
        assert not storage.get_path("uploaded.png", thumbnail=True).exists()
    finally:
        image.close()


def test_create_accepts_large_16_bit_image(
    real_image_service: tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage],
) -> None:
    service, records, storage = real_image_service
    image = Image.new("I;16", (1024, 1024), 32768)

    try:
        service.create(
            image=image,
            image_origin=ResourceOrigin.EXTERNAL,
            image_category=ImageCategory.GENERAL,
        )

        assert records.get("uploaded.png").image_subfolder == ""
        assert storage.get_path("uploaded.png").exists()
        assert storage.get_path("uploaded.png", thumbnail=True).exists()
    finally:
        image.close()


# ── Point 2: subfolder forwarding tests ──


class TestCreateSubfolderForwarding:
    """Verify that create() computes and forwards the correct subfolder for each strategy."""

    @pytest.mark.parametrize(
        "strategy_name,expected_subfolder",
        [
            ("flat", ""),
            ("type", "general"),
            ("hash", "ab"),  # first 2 chars of "abc12345-test.png"
        ],
        ids=["flat", "type", "hash"],
    )
    def test_create_forwards_subfolder(self, image_service: ImageService, strategy_name: str, expected_subfolder: str):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.configuration.image_subfolder_strategy = strategy_name

        # Make get_dto work by returning a record with the expected subfolder
        invoker.services.image_records.get.return_value = _make_record(image_subfolder=expected_subfolder)

        image = Image.new("RGB", (64, 64))
        image_service.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
        )

        # Assert image_records.save was called with the right subfolder
        save_call = invoker.services.image_records.save
        save_call.assert_called_once()
        assert save_call.call_args.kwargs["image_subfolder"] == expected_subfolder

        # Assert image_files.save was called with the same subfolder
        file_save = invoker.services.image_files.save
        file_save.assert_called_once()
        assert file_save.call_args.kwargs["image_subfolder"] == expected_subfolder

    def test_create_date_strategy_produces_date_subfolder(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.configuration.image_subfolder_strategy = "date"
        invoker.services.image_records.get.return_value = _make_record(image_subfolder="2026/04/05")

        image = Image.new("RGB", (64, 64))
        image_service.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
        )

        subfolder = invoker.services.image_records.save.call_args.kwargs["image_subfolder"]
        # Date strategy should produce YYYY/MM/DD format
        parts = subfolder.split("/")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_create_type_strategy_intermediate(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.configuration.image_subfolder_strategy = "type"
        invoker.services.image_records.get.return_value = _make_record(image_subfolder="intermediate")

        image = Image.new("RGB", (64, 64))
        image_service.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            is_intermediate=True,
        )

        subfolder = invoker.services.image_records.save.call_args.kwargs["image_subfolder"]
        assert subfolder == "intermediate"


class TestReadOperationsForwardSubfolder:
    """Verify that read operations look up the record and forward image_subfolder."""

    def test_get_pil_image(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get.return_value = _make_record(image_subfolder="2026/01/01")

        image_service.get_pil_image("test.png")

        invoker.services.image_files.get.assert_called_once_with("test.png", image_subfolder="2026/01/01")

    def test_get_workflow(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get.return_value = _make_record(image_subfolder="general")

        image_service.get_workflow("test.png")

        invoker.services.image_files.get_workflow.assert_called_once_with("test.png", image_subfolder="general")

    def test_get_graph(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get.return_value = _make_record(image_subfolder="general")

        image_service.get_graph("test.png")

        invoker.services.image_files.get_graph.assert_called_once_with("test.png", image_subfolder="general")

    def test_get_path(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get.return_value = _make_record(image_subfolder="ab")

        image_service.get_path("test.png")

        invoker.services.image_files.get_path.assert_called_once_with("test.png", False, image_subfolder="ab")

    def test_get_path_thumbnail(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get.return_value = _make_record(image_subfolder="ab")

        image_service.get_path("test.png", thumbnail=True)

        invoker.services.image_files.get_path.assert_called_once_with("test.png", True, image_subfolder="ab")


class TestDeleteForwardsSubfolder:
    """Verify that delete operations forward image_subfolder."""

    def test_delete_forwards_subfolder(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get.return_value = _make_record(image_subfolder="2026/04/05")

        image_service.delete("test.png")

        invoker.services.image_files.begin_delete.assert_called_once_with([("test.png", "2026/04/05")])
        invoker.services.image_records.delete.assert_called_once_with("test.png")
        invoker.services.image_files.commit_delete.assert_called_once_with(
            invoker.services.image_files.begin_delete.return_value
        )

    def test_delete_intermediates_forwards_subfolder(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [
            ("img1.png", "intermediate"),
            ("img2.png", "intermediate"),
        ]

        count = image_service.delete_intermediates()

        assert count == 2
        invoker.services.image_files.begin_delete.assert_called_once_with(
            [("img1.png", "intermediate"), ("img2.png", "intermediate")]
        )
        invoker.services.image_records.delete_intermediates_by_names.assert_called_once_with(["img1.png", "img2.png"])
        invoker.services.image_files.commit_delete.assert_called_once_with(
            invoker.services.image_files.begin_delete.return_value, image_names=["img1.png", "img2.png"]
        )


# ── Point 3: delete_images_on_board silent-failure contract ──


class TestDeleteImagesOnBoardContract:
    """A file-delete failure must preserve the corresponding database record."""

    def test_record_preserved_when_file_delete_fails(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.board_image_records.get_all_board_image_names_for_board.return_value = [
            "good.png",
            "bad.png",
        ]

        # First image record lookup succeeds, second fails
        good_record = _make_record(image_name="good.png", image_subfolder="general")
        bad_record = _make_record(image_name="bad.png", image_subfolder="bad/path")

        invoker.services.image_records.get.side_effect = [good_record, bad_record]
        # File staging succeeds for first, fails for second
        invoker.services.image_files.stage_delete.side_effect = [object(), Exception("disk error")]

        deleted, failed = image_service.delete_images_on_board("board-1")

        invoker.services.image_records.delete_many.assert_called_once_with(["good.png"])
        assert deleted == ["good.png"]
        assert failed == ["bad.png"]

    def test_file_cleanup_failure_does_not_raise(self, image_service: ImageService):
        """File cleanup errors are swallowed, not propagated."""
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.board_image_records.get_all_board_image_names_for_board.return_value = ["img.png"]

        record = _make_record(image_name="img.png", image_subfolder="sub")
        invoker.services.image_records.get.return_value = record
        invoker.services.image_files.stage_delete.side_effect = Exception("permission denied")

        deleted, failed = image_service.delete_images_on_board("board-1")

        invoker.services.image_records.delete_many.assert_called_once_with([])
        assert deleted == []
        assert failed == ["img.png"]

    def test_record_lookup_failure_does_not_block_others(self, image_service: ImageService):
        """If getting the record for one image fails, other images are still processed."""
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.board_image_records.get_all_board_image_names_for_board.return_value = [
            "missing.png",
            "ok.png",
        ]

        ok_record = _make_record(image_name="ok.png", image_subfolder="")
        invoker.services.image_records.get.side_effect = [Exception("not found"), ok_record]

        deleted, failed = image_service.delete_images_on_board("board-1")

        # File staging was attempted for the second image only
        invoker.services.image_files.stage_delete.assert_called_once_with("ok.png", image_subfolder="")
        invoker.services.image_records.delete_many.assert_called_once_with(["ok.png"])
        assert deleted == ["ok.png"]
        assert failed == ["missing.png"]

    def test_database_failure_restores_staged_files(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.board_image_records.get_all_board_image_names_for_board.return_value = ["img.png"]
        invoker.services.image_records.get.return_value = _make_record(image_name="img.png", image_subfolder="general")
        token = object()
        invoker.services.image_files.stage_delete.return_value = token
        invoker.services.image_records.delete_many.side_effect = RuntimeError("database unavailable")

        with pytest.raises(RuntimeError, match="database unavailable"):
            image_service.delete_images_on_board("board-1")

        invoker.services.image_files.rollback_delete.assert_called_once_with(token)
        invoker.services.image_files.commit_delete.assert_not_called()


# ── Transactional staged deletion (single image and intermediates) ──


@pytest.fixture
def disk_image_service(tmp_path: Path) -> ImageService:
    """ImageService wired to a real DiskImageFileStorage; all other services are mocks."""
    svc = ImageService()
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 1
    # By default every named intermediate is still an intermediate when the delete runs.
    invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: list(names)
    storage = DiskImageFileStorage(tmp_path / "outputs")
    invoker.services.image_files = storage
    storage.start(invoker)
    svc.start(invoker)
    return svc


def _save_image_file(storage: DiskImageFileStorage, image_name: str, image_subfolder: str = "") -> None:
    storage.save(image=Image.new("RGB", (64, 64)), image_name=image_name, image_subfolder=image_subfolder)


def _staging_dirs(storage: DiskImageFileStorage) -> list[Path]:
    return list(storage.image_root.glob(".delete_*"))


def _failing_directory_fsync(error_number: int):
    """An ``os.fsync`` that fails for directory descriptors only; file fsyncs still work."""
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(error_number, os.strerror(error_number))
        real_fsync(fd)

    return fsync


@pytest.fixture
def wired(tmp_path: Path) -> tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage]:
    """ImageService wired to a real record store and a real disk store — no stub decides anything."""
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    records = SqliteImageRecordStorage(db=create_mock_sqlite_database(config, logger))
    storage = DiskImageFileStorage(tmp_path / "outputs")

    svc = ImageService()
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 1
    invoker.services.image_records = records
    invoker.services.image_files = storage
    invoker.services.image_moves = None
    storage.start(invoker)
    svc.start(invoker)
    return svc, records, storage


@pytest.fixture
def wired_with_move_service(
    tmp_path: Path,
) -> tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage, ImageMoveService]:
    """ImageService and ImageMoveService wired to one real db and disk store, as production does."""
    config = InvokeAIAppConfig(use_memory_db=True, image_subfolder_strategy="flat")
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    records = SqliteImageRecordStorage(db=db)
    storage = DiskImageFileStorage(tmp_path / "outputs")
    moves = ImageMoveService(db=db, image_files=storage, config=config, logger=logger)

    svc = ImageService()
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 1
    invoker.services.image_records = records
    invoker.services.image_files = storage
    invoker.services.image_moves = moves
    invoker.services.names.create_image_name.return_value = "raced.png"
    invoker.services.urls.get_image_url.return_value = "/api/v1/images/i/raced.png"
    invoker.services.board_image_records.get_board_for_image.return_value = None
    storage.start(invoker)
    svc.start(invoker)
    return svc, records, storage, moves


def _seed_record_at_subfolder(records: SqliteImageRecordStorage, name: str, image_subfolder: str) -> None:
    records.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        has_workflow=False,
        is_intermediate=True,
        image_subfolder=image_subfolder,
    )


def _seed_record(records: SqliteImageRecordStorage, name: str, is_intermediate: bool = True) -> None:
    records.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        has_workflow=False,
        is_intermediate=is_intermediate,
    )


def _restart_file_storage(storage: DiskImageFileStorage, records: SqliteImageRecordStorage) -> DiskImageFileStorage:
    """Simulates a restart over the same output folder, running delete-journal recovery."""
    invoker = MagicMock()
    invoker.services.image_records = records
    restarted = DiskImageFileStorage(storage.image_root)
    restarted.start(invoker)
    return restarted


def _unlink_always_fails(path: Path, missing_ok: bool = False) -> None:
    raise OSError("device busy")


class TestDeleteTransactional:
    """delete() journals its intent, deletes the record, then purges — never losing files on failure."""

    def test_delete_success_removes_files_record_and_fires_callback_once(self, disk_image_service: ImageService):
        invoker = disk_image_service._ImageService__invoker  # type: ignore
        storage = invoker.services.image_files
        _save_image_file(storage, "img.png")
        invoker.services.image_records.get.return_value = _make_record(image_name="img.png")
        deleted_callbacks: list[str] = []
        disk_image_service.on_deleted(deleted_callbacks.append)

        disk_image_service.delete("img.png")

        assert not storage.get_path("img.png").exists()
        assert not storage.get_path("img.png", thumbnail=True).exists()
        invoker.services.image_records.delete.assert_called_once_with("img.png")
        assert deleted_callbacks == ["img.png"]
        assert _staging_dirs(storage) == []

    def test_delete_journal_failure_keeps_record_and_raises(self, image_service: ImageService):
        """The journal is written before the record is deleted, so failing to write it aborts."""
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_files.begin_delete.side_effect = ImageFileDeleteException("disk error")
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageFileDeleteException):
            image_service.delete("test.png")

        invoker.services.image_records.delete.assert_not_called()
        invoker.services.image_files.commit_delete.assert_not_called()
        assert deleted_callbacks == []

    def test_delete_db_failure_leaves_files_and_raises(self, disk_image_service: ImageService):
        invoker = disk_image_service._ImageService__invoker  # type: ignore
        storage = invoker.services.image_files
        _save_image_file(storage, "img.png")
        invoker.services.image_records.get.return_value = _make_record(image_name="img.png")
        invoker.services.image_records.delete.side_effect = ImageRecordDeleteException()
        deleted_callbacks: list[str] = []
        disk_image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageRecordDeleteException):
            disk_image_service.delete("img.png")

        # Nothing was moved, so the image and its thumbnail are still exactly where they were.
        assert storage.get_path("img.png").exists()
        assert storage.get_path("img.png", thumbnail=True).exists()
        assert deleted_callbacks == []
        assert _staging_dirs(storage) == []

    def test_delete_journal_cleanup_failure_still_raises_db_error(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.delete.side_effect = ImageRecordDeleteException()
        invoker.services.image_files.abandon_delete.side_effect = ImageFileDeleteException("journal locked")
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageRecordDeleteException):
            image_service.delete("test.png")

        invoker.services.image_files.abandon_delete.assert_called_once_with(
            invoker.services.image_files.begin_delete.return_value
        )
        invoker.services.image_files.commit_delete.assert_not_called()
        assert deleted_callbacks == []

    def test_delete_commit_failure_is_logged_not_raised(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_files.commit_delete.side_effect = ImageFileDeleteException("purge failed")
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        image_service.delete("test.png")

        invoker.services.image_records.delete.assert_called_once_with("test.png")
        invoker.services.image_files.abandon_delete.assert_not_called()
        assert deleted_callbacks == ["test.png"]
        invoker.services.logger.error.assert_called()


class TestDeleteIntermediatesTransactional:
    """delete_intermediates() deletes records first, then purges the files of exactly the rows it
    removed. It never stages or restores a promoted image's files, so there is no restore step for a
    concurrent delete to race (PR #9361, JPPhoto round 2)."""

    def test_success_deletes_multiple_intermediates(self, disk_image_service: ImageService):
        invoker = disk_image_service._ImageService__invoker  # type: ignore
        storage = invoker.services.image_files
        names = ["tmp1.png", "tmp2.png", "tmp3.png"]
        for name in names:
            _save_image_file(storage, name)
        invoker.services.image_records.get_intermediates.return_value = [(name, "") for name in names]
        deleted_callbacks: list[str] = []
        disk_image_service.on_deleted(deleted_callbacks.append)

        count = disk_image_service.delete_intermediates()

        assert count == 3
        for name in names:
            assert not storage.get_path(name).exists()
            assert not storage.get_path(name, thumbnail=True).exists()
        invoker.services.image_records.delete_intermediates_by_names.assert_called_once_with(names)
        assert deleted_callbacks == names
        assert _staging_dirs(storage) == []

    def test_promoted_image_keeps_its_files(self, disk_image_service: ImageService):
        """An image the DB refused to delete (no longer an intermediate) keeps its files untouched."""
        invoker = disk_image_service._ImageService__invoker  # type: ignore
        storage = invoker.services.image_files
        for name in ("tmp1.png", "promoted.png", "tmp2.png"):
            _save_image_file(storage, name)
        invoker.services.image_records.get_intermediates.return_value = [
            ("tmp1.png", ""),
            ("promoted.png", ""),
            ("tmp2.png", ""),
        ]
        # The store reports it removed everything except promoted.png, so that file is never purged.
        invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: [
            name for name in names if name != "promoted.png"
        ]
        deleted_callbacks: list[str] = []
        disk_image_service.on_deleted(deleted_callbacks.append)

        count = disk_image_service.delete_intermediates()

        assert count == 2
        assert storage.get_path("promoted.png").exists()
        assert storage.get_path("promoted.png", thumbnail=True).exists()
        for name in ("tmp1.png", "tmp2.png"):
            assert not storage.get_path(name).exists()
            assert not storage.get_path(name, thumbnail=True).exists()
        assert deleted_callbacks == ["tmp1.png", "tmp2.png"]
        assert _staging_dirs(storage) == []

    def test_only_deleted_rows_are_purged_and_announced(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("tmp1.png", ""), ("promoted.png", "")]
        invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: ["tmp1.png"]
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        count = image_service.delete_intermediates()

        assert count == 1
        # The promoted row's file is never touched: only the deleted row is purged.
        invoker.services.image_files.commit_delete.assert_called_once_with(
            invoker.services.image_files.begin_delete.return_value, image_names=["tmp1.png"]
        )
        assert deleted_callbacks == ["tmp1.png"]

    def test_subfolder_is_forwarded_to_the_file_purge(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("tmp1.png", "a/b")]
        invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: list(names)
        image_service.delete_intermediates()

        invoker.services.image_files.begin_delete.assert_called_once_with([("tmp1.png", "a/b")])

    def test_file_purge_failure_is_logged_and_does_not_raise(self, image_service: ImageService):
        """A filesystem failure must not undo the committed record deletions or raise: the records
        are already gone, and the journal the purge leaves behind is retried at the next startup."""
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("tmp1.png", ""), ("tmp2.png", "")]
        invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: list(names)
        invoker.services.image_files.commit_delete.side_effect = ImageFileDeleteException("purge failed")
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        count = image_service.delete_intermediates()

        assert count == 2
        # Both records were deleted, so both deletions are announced despite the file failure.
        assert deleted_callbacks == ["tmp1.png", "tmp2.png"]
        invoker.services.image_files.abandon_delete.assert_not_called()
        invoker.services.logger.error.assert_called()

    def test_db_failure_raises_and_purges_nothing(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("tmp1.png", ""), ("tmp2.png", "")]
        invoker.services.image_records.delete_intermediates_by_names.side_effect = ImageRecordDeleteException()
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageRecordDeleteException):
            image_service.delete_intermediates()

        # No record was removed, so no file may be purged and the journal must be discarded.
        invoker.services.image_files.commit_delete.assert_not_called()
        invoker.services.image_files.abandon_delete.assert_called_once_with(
            invoker.services.image_files.begin_delete.return_value
        )
        assert deleted_callbacks == []

    def test_nothing_deleted_returns_zero_and_fires_no_callbacks(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("promoted.png", "")]
        invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: []
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        assert image_service.delete_intermediates() == 0

        # The journal still lists the promoted image; the purge selects nothing.
        invoker.services.image_files.commit_delete.assert_called_once_with(
            invoker.services.image_files.begin_delete.return_value, image_names=[]
        )
        assert deleted_callbacks == []

    def test_empty_intermediates_is_a_noop(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = []
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        assert image_service.delete_intermediates() == 0

        # Nothing to delete: no journal is written and the record store is never asked to delete.
        invoker.services.image_records.delete_intermediates_by_names.assert_not_called()
        invoker.services.image_files.begin_delete.assert_not_called()
        assert deleted_callbacks == []


class TestDeleteIntermediatesAgainstRealRecords:
    """delete_intermediates() wired to a real record store, so no stub stands in for the DB decision.

    The mocked tests above can only assert that the service honours whatever the store reports. These
    exercise the real store, which is where the promoted-vs-already-gone distinction is actually made,
    and where the concurrency hazards JPPhoto reported would surface.
    """

    def _seed(self, records: SqliteImageRecordStorage, storage: DiskImageFileStorage, name: str) -> None:
        _seed_record(records, name)
        _save_image_file(storage, name)

    def _promote_after_snapshot(
        self,
        records: SqliteImageRecordStorage,
        monkeypatch,
        image_name: str,
    ) -> None:
        """Promote an image out of intermediate status after the snapshot but before the DB delete.

        Promoting it earlier would drop it from the snapshot entirely; the interesting case is an
        image that is in the snapshot yet is no longer an intermediate by the time the conditional
        DELETE runs, so its record (and files) must survive.
        """
        real_get_intermediates = records.get_intermediates

        def snapshot_then_promote():
            pairs = real_get_intermediates()
            records.update(image_name, ImageRecordChanges(is_intermediate=False))
            return pairs

        monkeypatch.setattr(records, "get_intermediates", snapshot_then_promote)

    def test_all_intermediates_are_deleted(self, wired) -> None:
        svc, records, storage = wired
        self._seed(records, storage, "tmp1.png")
        self._seed(records, storage, "tmp2.png")
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        assert svc.delete_intermediates() == 2

        for name in ("tmp1.png", "tmp2.png"):
            assert not storage.get_path(name).exists()
            with pytest.raises(ImageRecordNotFoundException):
                records.get(name)
        assert sorted(deleted_callbacks) == ["tmp1.png", "tmp2.png"]
        assert _staging_dirs(storage) == []

    def test_promoted_image_keeps_record_and_files(self, wired, monkeypatch) -> None:
        svc, records, storage = wired
        self._seed(records, storage, "tmp1.png")
        self._seed(records, storage, "promoted.png")
        self._promote_after_snapshot(records, monkeypatch, "promoted.png")
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        assert svc.delete_intermediates() == 1

        assert storage.get_path("promoted.png").exists()
        assert records.get("promoted.png").is_intermediate is False
        assert not storage.get_path("tmp1.png").exists()
        assert deleted_callbacks == ["tmp1.png"]
        assert _staging_dirs(storage) == []

    def test_record_removed_by_another_path_between_snapshot_and_delete(self, wired, monkeypatch) -> None:
        """An image fully deleted elsewhere after the snapshot is not counted and its (now absent)
        files are left to the path that owns that deletion — we never touch them."""
        svc, records, storage = wired
        self._seed(records, storage, "tmp1.png")
        self._seed(records, storage, "gone.png")

        real_get_intermediates = records.get_intermediates

        def snapshot_then_delete_gone():
            pairs = real_get_intermediates()
            # A single-image delete elsewhere removes gone.png (record and files) after our snapshot.
            records.delete("gone.png")
            storage.delete("gone.png")
            return pairs

        monkeypatch.setattr(records, "get_intermediates", snapshot_then_delete_gone)
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        count = svc.delete_intermediates()

        assert count == 1
        assert deleted_callbacks == ["tmp1.png"]
        assert not storage.get_path("tmp1.png").exists()
        # gone.png was purged by the other path; we neither resurrect nor re-report it.
        assert not storage.get_path("gone.png").exists()
        assert _staging_dirs(storage) == []

    def test_promoted_record_deleted_after_conditional_delete_is_not_resurrected(self, wired, monkeypatch) -> None:
        """The B3 regression: a promoted image is concurrently deleted (record and files) right after
        the conditional DELETE keeps it. Because we never staged its files, there is nothing to
        restore — its files stay deleted and are not stranded on disk with no record.
        """
        svc, records, storage = wired
        self._seed(records, storage, "tmp1.png")
        self._seed(records, storage, "promoted.png")
        self._promote_after_snapshot(records, monkeypatch, "promoted.png")

        real_delete_by_names = records.delete_intermediates_by_names

        def delete_then_lose_the_promoted_record(names: list[str]):
            deleted = real_delete_by_names(names)
            # A concurrent single-image delete removes the promoted image entirely, right after the
            # conditional DELETE chose to keep it.
            records.delete("promoted.png")
            storage.delete("promoted.png")
            return deleted

        monkeypatch.setattr(records, "delete_intermediates_by_names", delete_then_lose_the_promoted_record)
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        count = svc.delete_intermediates()

        assert count == 1
        assert deleted_callbacks == ["tmp1.png"]
        assert not storage.get_path("tmp1.png").exists()
        # promoted.png's files stay deleted — never resurrected into an orphan.
        assert not storage.get_path("promoted.png").exists()
        assert not storage.get_path("promoted.png", thumbnail=True).exists()
        with pytest.raises(ImageRecordNotFoundException):
            records.get("promoted.png")
        assert _staging_dirs(storage) == []


class TestDeleteAgainstRealRecords:
    """Single-image delete wired to a real record store, covering the concurrent-delete interleaving
    JPPhoto reported (PR #9361 round 3)."""

    def test_a_failed_delete_never_resurrects_files_another_request_removed(self, wired, monkeypatch) -> None:
        """Two requests delete the same image; one commits the record deletion and the other fails.

        The failing request must not put the files back: nothing references them any more, and the
        journal that would let startup recovery find them is gone with the request that won.
        """
        svc, records, storage = wired
        _seed_record(records, "img.png")
        _save_image_file(storage, "img.png")

        real_delete = records.delete

        def competing_delete_then_fail(image_name: str) -> None:
            # The competing request wins the race: it removes the record and purges the files while
            # this delete is still in flight, and only then does this one's own delete fail.
            real_delete(image_name)
            storage.delete(image_name)
            raise ImageRecordDeleteException()

        monkeypatch.setattr(records, "delete", competing_delete_then_fail)
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageRecordDeleteException):
            svc.delete("img.png")

        assert not storage.get_path("img.png").exists()
        assert not storage.get_path("img.png", thumbnail=True).exists()
        assert deleted_callbacks == []
        assert _staging_dirs(storage) == []

    def test_a_database_failure_leaves_the_image_completely_intact(self, wired, monkeypatch) -> None:
        svc, records, storage = wired
        _seed_record(records, "img.png")
        _save_image_file(storage, "img.png")

        def failing_delete(image_name: str) -> None:
            raise ImageRecordDeleteException()

        monkeypatch.setattr(records, "delete", failing_delete)

        with pytest.raises(ImageRecordDeleteException):
            svc.delete("img.png")

        assert storage.get_path("img.png").exists()
        assert storage.get_path("img.png", thumbnail=True).exists()
        assert records.get("img.png").image_name == "img.png"
        # The journal is discarded: the image is live, so nothing must be left pointing at it.
        assert _staging_dirs(storage) == []


class TestDeleteJournalSurvivesFailedPurges:
    """Records-first deletion commits the record removal before the files are purged. A crash or a
    filesystem failure in that window must leave a journal, not a silent orphan (JPPhoto, PR #9361
    round 3)."""

    def test_intermediates_purge_failure_leaves_a_journal_the_next_startup_finishes(self, wired, monkeypatch) -> None:
        svc, records, storage = wired
        for name in ("tmp1.png", "tmp2.png"):
            _seed_record(records, name)
            _save_image_file(storage, name)
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        with patch.object(Path, "unlink", _unlink_always_fails):
            count = svc.delete_intermediates()

        # The records are committed as gone, so the deletion succeeded and is announced...
        assert count == 2
        assert sorted(deleted_callbacks) == ["tmp1.png", "tmp2.png"]
        for name in ("tmp1.png", "tmp2.png"):
            with pytest.raises(ImageRecordNotFoundException):
                records.get(name)
            # ...but the files could not be removed, so they must still be journalled.
            assert storage.get_path(name).exists()
        assert _staging_dirs(storage) != []

        _restart_file_storage(storage, records)

        for name in ("tmp1.png", "tmp2.png"):
            assert not storage.get_path(name).exists()
            assert not storage.get_path(name, thumbnail=True).exists()
        assert _staging_dirs(storage) == []

    def test_a_crash_before_the_purge_leaves_a_journal_the_next_startup_finishes(self, wired) -> None:
        """The process dies between the committed record deletion and the file purge."""
        svc, records, storage = wired
        _seed_record(records, "img.png", is_intermediate=False)
        _save_image_file(storage, "img.png")

        # Everything delete() does up to the point of no return, and then nothing.
        record = records.get("img.png")
        storage.begin_delete([("img.png", record.image_subfolder)])
        records.delete("img.png")

        assert storage.get_path("img.png").exists()

        _restart_file_storage(storage, records)

        assert not storage.get_path("img.png").exists()
        assert not storage.get_path("img.png", thumbnail=True).exists()
        assert _staging_dirs(storage) == []

    def test_a_crash_before_the_record_delete_keeps_the_image(self, wired) -> None:
        """The mirror case: the journal is written but the record deletion never happened."""
        svc, records, storage = wired
        _seed_record(records, "img.png", is_intermediate=False)
        _save_image_file(storage, "img.png")

        storage.begin_delete([("img.png", "")])

        _restart_file_storage(storage, records)

        assert storage.get_path("img.png").exists()
        assert storage.get_path("img.png", thumbnail=True).exists()
        assert records.get("img.png").image_name == "img.png"
        assert _staging_dirs(storage) == []


class TestFailedSaveCleanup:
    """A save that fails halfway must clean up record-first, like every other delete path.

    The order is load-bearing: a concurrent deleter rolling back decides whether to restore an
    image's files by asking whether its record is still there, so purging files while the record
    survives would tell it to put them back and strand them (adversarial review, PR #9361).
    """

    def test_the_record_is_deleted_before_the_files_are_purged(self, image_service: ImageService) -> None:
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_files.save.side_effect = ImageFileSaveException()
        order: list[str] = []
        token = object()

        def journal(images):
            order.append("journal")
            return token

        invoker.services.image_files.begin_delete.side_effect = journal
        invoker.services.image_records.delete.side_effect = lambda name: order.append("record")
        invoker.services.image_files.commit_delete.side_effect = lambda t, image_names=None: order.append("purge")

        with pytest.raises(ImageFileSaveException):
            image_service.create(
                image=Image.new("RGB", (8, 8)),
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
            )

        assert order == ["journal", "record", "purge"]
        invoker.services.image_files.commit_delete.assert_called_once_with(token)

    def test_a_surviving_record_keeps_its_files(self, image_service: ImageService) -> None:
        """If the record cannot be deleted the image is still referenced, so nothing may be purged."""
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_files.save.side_effect = ImageFileSaveException()
        invoker.services.image_records.delete.side_effect = ImageRecordDeleteException()

        with pytest.raises(ImageFileSaveException):
            image_service.create(
                image=Image.new("RGB", (8, 8)),
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
            )

        invoker.services.image_files.commit_delete.assert_not_called()
        invoker.services.image_files.delete.assert_not_called()
        invoker.services.image_files.abandon_delete.assert_called_once_with(
            invoker.services.image_files.begin_delete.return_value
        )

    def test_the_record_is_kept_when_the_cleanup_cannot_be_journaled(self, image_service: ImageService) -> None:
        """No durable journal, no record deletion. Removing the record and purging blind would fail
        at the same journal step and leave whatever survived the save as an orphan nothing can find;
        a surviving record is what lets a later delete find and clear it."""
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_files.save.side_effect = ImageFileSaveException()
        invoker.services.image_files.begin_delete.side_effect = ImageFileDeleteException("fsync: I/O error")

        with pytest.raises(ImageFileSaveException):
            image_service.create(
                image=Image.new("RGB", (8, 8)),
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
            )

        invoker.services.image_records.delete.assert_not_called()
        invoker.services.image_files.commit_delete.assert_not_called()
        invoker.services.image_files.delete.assert_not_called()

    @pytest.mark.skipif(os.name == "nt", reason="Windows cannot open a directory for fsync")
    def test_a_leftover_file_is_never_orphaned_without_a_journal(
        self,
        real_image_service: tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage],
    ) -> None:
        """Real storage, real records: the save leaves a file behind and the disk refuses to make
        the cleanup journal durable. The record must survive, and an ordinary delete once the disk
        is healthy must be able to clear the whole image."""
        service, records, storage = real_image_service
        # save() removes only what it created, so a file that already sits at the image's path is
        # what survives a failed save.
        leftover = storage.get_path("uploaded.png")
        leftover.parent.mkdir(parents=True, exist_ok=True)
        leftover.write_bytes(b"pre-existing")
        broken_thumbnail = MagicMock()
        broken_thumbnail.save.side_effect = OSError("thumbnail filesystem failure")

        with (
            patch(
                "invokeai.app.services.image_files.image_files_disk.make_thumbnail",
                return_value=broken_thumbnail,
            ),
            patch.object(os, "fsync", _failing_directory_fsync(errno.EIO)),
            pytest.raises(ImageFileSaveException),
        ):
            service.create(
                image=Image.new("RGB", (32, 32), "red"),
                image_origin=ResourceOrigin.EXTERNAL,
                image_category=ImageCategory.GENERAL,
            )

        assert records.get("uploaded.png").image_name == "uploaded.png"
        assert leftover.exists()
        assert _staging_dirs(storage) == []

        service.delete("uploaded.png")

        with pytest.raises(ImageRecordNotFoundException):
            records.get("uploaded.png")
        assert not leftover.exists()
        assert _staging_dirs(storage) == []


class TestConcurrentBoardDeleteAgainstRealRecords:
    """delete_images_on_board() still stages, because its per-item contract needs a pre-flight move.
    Two of them racing for one image must not strand it (adversarial review, PR #9361)."""

    def test_committing_an_empty_token_still_purges_the_files(self, wired, monkeypatch) -> None:
        """The loser moved the files aside; the winner staged nothing and removed the record.

        The winner's commit is the only thing standing between the loser's restore and a permanent
        orphan: by the time the loser rolls back, the record is still there, so its own re-check
        tells it to keep the files it just put back.
        """
        svc, records, storage = wired
        invoker = svc._ImageService__invoker  # type: ignore
        _seed_record(records, "img.png", is_intermediate=False)
        _save_image_file(storage, "img.png")
        invoker.services.board_image_records.get_all_board_image_names_for_board.return_value = ["img.png"]

        # The competing request wins the race to the files, so this delete stages an empty token.
        competing = storage.stage_delete("img.png", "")
        real_delete_many = records.delete_many

        def competitor_rolls_back_then_delete(image_names: list[str]) -> None:
            # The competing request's own record deletion failed, so it restores the files — while
            # this record is still present, which is what makes its re-check keep them.
            storage.rollback_delete(competing)
            real_delete_many(image_names)

        monkeypatch.setattr(records, "delete_many", competitor_rolls_back_then_delete)

        deleted, failed = svc.delete_images_on_board("board-1")

        assert deleted == ["img.png"]
        assert failed == []
        with pytest.raises(ImageRecordNotFoundException):
            records.get("img.png")
        assert not storage.get_path("img.png").exists()
        assert not storage.get_path("img.png", thumbnail=True).exists()
        assert _staging_dirs(storage) == []


class TestDeleteVersusSubfolderMove:
    """The image-mutation lock that serializes delete units against subfolder relocations.

    A delete reads an image's subfolder, deletes its record, then purges its files at that
    subfolder; a move relocates the files and repoints the record. If the two interleave, the
    delete purges the path its snapshot named while the files sit at the new one — permanent
    orphans, unrecoverable because the record is gone and a clean purge drops the journal
    (JPPhoto, PR #9361). These tests drive both real services against one real db and disk
    store and pin the serialization from both directions.
    """

    def test_move_cannot_interleave_with_delete_intermediates(
        self,
        wired_with_move_service: tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage, ImageMoveService],
        monkeypatch,
    ) -> None:
        """While delete_intermediates() runs its unit, a concurrent move_all_images() must wait.

        Without the lock the move relocates the image mid-delete and the delete's purge sweeps
        the abandoned path: the files survive at the new subfolder with no record and no journal.
        """
        svc, records, storage, moves = wired_with_move_service
        _seed_record_at_subfolder(records, "raced.png", "old")
        _save_image_file(storage, "raced.png", image_subfolder="old")

        delete_started = threading.Event()
        delete_may_finish = threading.Event()
        real_delete_by_names = records.delete_intermediates_by_names

        def delete_after_pause(names: list[str]) -> list[str]:
            delete_started.set()
            assert delete_may_finish.wait(timeout=10), "the move never let the delete finish"
            return real_delete_by_names(names)

        monkeypatch.setattr(records, "delete_intermediates_by_names", delete_after_pause)

        delete_thread = threading.Thread(target=svc.delete_intermediates)
        delete_thread.start()
        assert delete_started.wait(timeout=10), "the delete never reached its record deletion"

        # The move finds the row (the delete has not removed it yet) and has a relocation to
        # perform (the row says "old"; the strategy says the flat root). It must wait for the
        # delete's unit instead of racing it.
        move_result: list[object] = []
        move_thread = threading.Thread(target=lambda: move_result.append(moves.move_all_images()))
        move_thread.start()
        move_thread.join(timeout=1.0)
        assert move_thread.is_alive(), "move_all_images finished while a delete held the mutation lock"
        assert storage.get_path("raced.png", image_subfolder="old").exists(), (
            "the move relocated files while a delete was mid-unit"
        )

        delete_may_finish.set()
        delete_thread.join(timeout=10)
        move_thread.join(timeout=10)
        assert not delete_thread.is_alive()
        assert not move_thread.is_alive()

        result = move_result[0]
        assert isinstance(result, ImageMoveResult)
        assert result.errors == 0
        with pytest.raises(ImageRecordNotFoundException):
            records.get("raced.png")
        assert not storage.get_path("raced.png", image_subfolder="old").exists()
        assert not storage.get_path("raced.png").exists()
        assert not storage.get_path("raced.png", thumbnail=True).exists()
        assert _staging_dirs(storage) == []

    def test_move_holds_the_lock_across_its_batch_cycle(
        self,
        wired_with_move_service: tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage, ImageMoveService],
    ) -> None:
        """While a move unit runs, a concurrent delete must wait for it rather than race it.

        The lock is held across the whole plan-relocate-repoint cycle: nothing may be relocated
        or repointed, and the cycle may not complete, while another party holds the lock.
        """
        svc, records, storage, moves = wired_with_move_service
        _seed_record_at_subfolder(records, "moved.png", "old")
        _save_image_file(storage, "moved.png", image_subfolder="old")

        with moves.image_mutation_lock():
            move_result: list[object] = []
            move_thread = threading.Thread(target=lambda: move_result.append(moves.move_all_images()))
            move_thread.start()
            move_thread.join(timeout=1.0)
            assert move_thread.is_alive(), "move_all_images completed while the mutation lock was held"
            assert storage.get_path("moved.png", image_subfolder="old").exists()
            assert records.get("moved.png").image_subfolder == "old"

        move_thread.join(timeout=10)
        assert not move_thread.is_alive()

        result = move_result[0]
        assert isinstance(result, ImageMoveResult)
        assert (result.planned, result.committed, result.errors) == (1, 1, 0)
        assert not storage.get_path("moved.png", image_subfolder="old").exists()
        assert storage.get_path("moved.png").exists()
        assert records.get("moved.png").image_subfolder == ""

    def test_failed_save_cleanup_purges_the_relocated_subfolder_too(
        self,
        wired_with_move_service: tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage, ImageMoveService],
    ) -> None:
        """A failed save whose record was relocated mid-save must not strand the moved files.

        The cleanup captured the subfolder the save used, but a move that completed before the
        cleanup began repointed the record; the purge has to cover the path the record names now
        as well as the one the save wrote to.
        """
        svc, records, storage, moves = wired_with_move_service
        _seed_record_at_subfolder(records, "failed.png", "old")
        _save_image_file(storage, "failed.png", image_subfolder="old")

        # A subfolder relocation completes before the cleanup runs: files moved, record repointed.
        # Simulated directly — a real move job would leave item rows whose foreign key blocks the
        # record delete, a pre-existing limitation on main unrelated to this lock.
        old_image = storage.get_path("failed.png", image_subfolder="old")
        old_thumbnail = storage.get_path("failed.png", thumbnail=True, image_subfolder="old")
        new_image = storage.get_path("failed.png")
        new_thumbnail = storage.get_path("failed.png", thumbnail=True)
        new_image.parent.mkdir(parents=True, exist_ok=True)
        new_thumbnail.parent.mkdir(parents=True, exist_ok=True)
        old_image.replace(new_image)
        old_thumbnail.replace(new_thumbnail)
        with records._db.transaction() as cursor:
            cursor.execute("UPDATE images SET image_subfolder = '' WHERE image_name = 'failed.png';")
        assert records.get("failed.png").image_subfolder == ""

        # The failed save's cleanup still holds the subfolder the save captured.
        svc._ImageService__clean_up_failed_save("failed.png", "old")  # type: ignore[attr-defined]

        with pytest.raises(ImageRecordNotFoundException):
            records.get("failed.png")
        assert not new_image.exists()
        assert not new_thumbnail.exists()
        assert not old_image.exists()
        assert not old_thumbnail.exists()
        assert _staging_dirs(storage) == []

    def test_move_cannot_interleave_with_create(
        self,
        wired_with_move_service: tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage, ImageMoveService],
        monkeypatch,
    ) -> None:
        """While create() runs its save unit, a concurrent move_all_images() must wait.

        The record becomes visible to the move planner the moment it commits, but its files do
        not exist until the save lands. Under the date strategy the two subfolders can even
        disagree by a day — the strategy reads the local clock while the move target comes from
        the record's UTC created_at — so a relocation landing mid-save leaves the record naming
        a subfolder the files were never written to.
        """
        svc, records, storage, moves = wired_with_move_service

        # The record this create() writes lands at a subfolder the move service's strategy
        # (flat) does not consider final — the situation the date strategy produces on any
        # non-UTC host. The subfolder directory must exist, or the move errors in preflight
        # before it can repoint anything.
        class _StaleSubfolderStrategy:
            def get_subfolder(self, image_name, image_category, is_intermediate):
                return "old"

        monkeypatch.setattr(
            "invokeai.app.services.images.images_default.create_subfolder_strategy",
            lambda _strategy_name: _StaleSubfolderStrategy(),
        )
        storage.get_path("raced.png", image_subfolder="old").parent.mkdir(parents=True, exist_ok=True)

        save_started = threading.Event()
        save_may_finish = threading.Event()
        real_save = storage.save

        def save_then_pause(**kwargs):
            save_started.set()
            assert save_may_finish.wait(timeout=10), "the move never let the create finish"
            real_save(**kwargs)

        monkeypatch.setattr(storage, "save", save_then_pause)

        create_thread = threading.Thread(
            target=lambda: svc.create(
                image=Image.new("RGB", (64, 64)),
                image_origin=ResourceOrigin.EXTERNAL,
                image_category=ImageCategory.GENERAL,
            )
        )
        create_thread.start()
        assert save_started.wait(timeout=10), "the create never reached its file save"
        assert records.get("raced.png").image_subfolder == "old"

        move_result: list[object] = []
        move_thread = threading.Thread(target=lambda: move_result.append(moves.move_all_images()))
        move_thread.start()
        move_thread.join(timeout=1.0)
        assert move_thread.is_alive(), "move_all_images completed while create() held the mutation lock"
        assert records.get("raced.png").image_subfolder == "old", (
            "the move repointed the record while create() was still writing its files"
        )

        save_may_finish.set()
        create_thread.join(timeout=10)
        move_thread.join(timeout=10)
        assert not create_thread.is_alive()
        assert not move_thread.is_alive()

        # The move then relocates the completed image: the record and the files end up agreeing.
        assert records.get("raced.png").image_subfolder == ""
        assert storage.get_path("raced.png").exists()
        assert not storage.get_path("raced.png", image_subfolder="old").exists()
        assert storage.get_path("raced.png", thumbnail=True).exists()
        assert _staging_dirs(storage) == []
