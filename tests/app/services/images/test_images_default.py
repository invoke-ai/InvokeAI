"""Tests for ImageService (images_default.py).

Covers subfolder forwarding for all strategies, the delete_images_on_board
silent-failure contract (Points 2 & 3 from PR review), and the transactional
staged-deletion contracts of delete() and delete_intermediates().
"""

import sqlite3
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
    # By default every staged intermediate is still an intermediate when the delete runs.
    invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: (list(names), [])

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

        invoker.services.image_files.stage_delete.assert_called_once_with("test.png", image_subfolder="2026/04/05")
        invoker.services.image_records.delete.assert_called_once_with("test.png")
        invoker.services.image_files.commit_delete.assert_called_once_with(
            invoker.services.image_files.stage_delete.return_value
        )

    def test_delete_intermediates_forwards_subfolder(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [
            ("img1.png", "intermediate"),
            ("img2.png", "intermediate"),
        ]

        count = image_service.delete_intermediates()

        assert count == 2
        calls = invoker.services.image_files.stage_delete.call_args_list
        assert calls[0].args == ("img1.png",)
        assert calls[0].kwargs == {"image_subfolder": "intermediate"}
        assert calls[1].args == ("img2.png",)
        assert calls[1].kwargs == {"image_subfolder": "intermediate"}
        invoker.services.image_records.delete_intermediates_by_names.assert_called_once_with(["img1.png", "img2.png"])


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
    # By default every staged intermediate is still an intermediate when the delete runs.
    invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: (list(names), [])
    storage = DiskImageFileStorage(tmp_path / "outputs")
    invoker.services.image_files = storage
    storage.start(invoker)
    svc.start(invoker)
    return svc


def _save_image_file(storage: DiskImageFileStorage, image_name: str, image_subfolder: str = "") -> None:
    storage.save(image=Image.new("RGB", (64, 64)), image_name=image_name, image_subfolder=image_subfolder)


def _staging_dirs(storage: DiskImageFileStorage) -> list[Path]:
    return list(storage.image_root.glob(".delete_*"))


class TestDeleteTransactional:
    """delete() must stage files, delete the record, then commit — never losing files on failure."""

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

    def test_delete_staging_failure_keeps_record_and_raises(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_files.stage_delete.side_effect = ImageFileDeleteException("disk error")
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageFileDeleteException):
            image_service.delete("test.png")

        invoker.services.image_records.delete.assert_not_called()
        invoker.services.image_files.commit_delete.assert_not_called()
        assert deleted_callbacks == []

    def test_delete_db_failure_restores_files_and_raises(self, disk_image_service: ImageService):
        invoker = disk_image_service._ImageService__invoker  # type: ignore
        storage = invoker.services.image_files
        _save_image_file(storage, "img.png")
        invoker.services.image_records.get.return_value = _make_record(image_name="img.png")
        invoker.services.image_records.delete.side_effect = ImageRecordDeleteException()
        deleted_callbacks: list[str] = []
        disk_image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageRecordDeleteException):
            disk_image_service.delete("img.png")

        # The image and its thumbnail must be restored to their original paths.
        assert storage.get_path("img.png").exists()
        assert storage.get_path("img.png", thumbnail=True).exists()
        assert deleted_callbacks == []
        assert _staging_dirs(storage) == []

    def test_delete_rollback_failure_still_raises_db_error(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.delete.side_effect = ImageRecordDeleteException()
        invoker.services.image_files.rollback_delete.side_effect = ImageFileDeleteException("rollback broken")
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageRecordDeleteException):
            image_service.delete("test.png")

        invoker.services.image_files.rollback_delete.assert_called_once_with(
            invoker.services.image_files.stage_delete.return_value
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
        invoker.services.image_files.rollback_delete.assert_not_called()
        assert deleted_callbacks == ["test.png"]
        invoker.services.logger.error.assert_called()


class TestDeleteIntermediatesTransactional:
    """delete_intermediates() must be all-or-nothing: stage everything, delete records once, commit."""

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

    def test_image_promoted_out_of_intermediate_keeps_record_and_files(self, disk_image_service: ImageService):
        """An image that stops being an intermediate mid-operation keeps its record and its files."""
        invoker = disk_image_service._ImageService__invoker  # type: ignore
        storage = invoker.services.image_files
        for name in ("tmp1.png", "promoted.png", "tmp2.png"):
            _save_image_file(storage, name)
        invoker.services.image_records.get_intermediates.return_value = [
            ("tmp1.png", ""),
            ("promoted.png", ""),
            ("tmp2.png", ""),
        ]
        # The database reports that promoted.png was no longer an intermediate, so its record stands.
        invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: (
            [name for name in names if name != "promoted.png"],
            ["promoted.png"],
        )
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

    def test_promoted_image_is_rolled_back_not_committed(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("tmp1.png", ""), ("promoted.png", "")]
        tokens = [object(), object()]
        invoker.services.image_files.stage_delete.side_effect = tokens
        invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: (
            ["tmp1.png"],
            ["promoted.png"],
        )
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        count = image_service.delete_intermediates()

        assert count == 1
        invoker.services.image_files.commit_delete.assert_called_once_with(tokens[0])
        invoker.services.image_files.rollback_delete.assert_called_once_with(tokens[1])
        assert deleted_callbacks == ["tmp1.png"]

    def test_rollback_failure_for_promoted_image_does_not_abort_the_rest(self, image_service: ImageService):
        """A failed restore is logged; the surviving deletions still commit and fire callbacks."""
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [
            ("promoted.png", ""),
            ("tmp1.png", ""),
        ]
        tokens = [object(), object()]
        invoker.services.image_files.stage_delete.side_effect = tokens
        invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: (
            ["tmp1.png"],
            ["promoted.png"],
        )
        invoker.services.image_files.rollback_delete.side_effect = ImageFileDeleteException("rollback broken")
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        count = image_service.delete_intermediates()

        assert count == 1
        invoker.services.image_files.rollback_delete.assert_called_once_with(tokens[0])
        invoker.services.image_files.commit_delete.assert_called_once_with(tokens[1])
        assert deleted_callbacks == ["tmp1.png"]
        invoker.services.logger.error.assert_called()

    def test_nothing_deleted_returns_zero_and_fires_no_callbacks(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("promoted.png", "")]
        token = object()
        invoker.services.image_files.stage_delete.return_value = token
        invoker.services.image_records.delete_intermediates_by_names.side_effect = lambda names: ([], ["promoted.png"])
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        assert image_service.delete_intermediates() == 0

        invoker.services.image_files.rollback_delete.assert_called_once_with(token)
        invoker.services.image_files.commit_delete.assert_not_called()
        assert deleted_callbacks == []

    def test_first_staging_failure_aborts_without_db_delete(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("tmp1.png", ""), ("tmp2.png", "")]
        invoker.services.image_files.stage_delete.side_effect = ImageFileDeleteException("disk error")
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageFileDeleteException):
            image_service.delete_intermediates()

        invoker.services.image_records.delete_intermediates_by_names.assert_not_called()
        # Nothing was staged, so nothing needs rolling back.
        invoker.services.image_files.rollback_delete.assert_not_called()
        invoker.services.image_files.commit_delete.assert_not_called()
        assert deleted_callbacks == []

    def test_later_staging_failure_rolls_back_earlier_stages(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("tmp1.png", ""), ("tmp2.png", "")]
        token1 = object()
        invoker.services.image_files.stage_delete.side_effect = [token1, ImageFileDeleteException("disk error")]
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageFileDeleteException):
            image_service.delete_intermediates()

        invoker.services.image_records.delete_intermediates_by_names.assert_not_called()
        invoker.services.image_files.rollback_delete.assert_called_once_with(token1)
        invoker.services.image_files.commit_delete.assert_not_called()
        assert deleted_callbacks == []

    def test_later_staging_failure_restores_earlier_files_on_disk(self, disk_image_service: ImageService):
        invoker = disk_image_service._ImageService__invoker  # type: ignore
        storage = invoker.services.image_files
        _save_image_file(storage, "tmp1.png")
        # The second entry's subfolder fails path validation, so staging it raises after
        # tmp1.png has already been staged.
        invoker.services.image_records.get_intermediates.return_value = [
            ("tmp1.png", ""),
            ("tmp2.png", "bad\\path"),
        ]
        deleted_callbacks: list[str] = []
        disk_image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ValueError):
            disk_image_service.delete_intermediates()

        assert storage.get_path("tmp1.png").exists()
        assert storage.get_path("tmp1.png", thumbnail=True).exists()
        invoker.services.image_records.delete_intermediates_by_names.assert_not_called()
        assert deleted_callbacks == []
        assert _staging_dirs(storage) == []

    def test_db_failure_restores_all_staged_files(self, disk_image_service: ImageService):
        invoker = disk_image_service._ImageService__invoker  # type: ignore
        storage = invoker.services.image_files
        names = ["tmp1.png", "tmp2.png"]
        for name in names:
            _save_image_file(storage, name)
        invoker.services.image_records.get_intermediates.return_value = [(name, "") for name in names]
        invoker.services.image_records.delete_intermediates_by_names.side_effect = ImageRecordDeleteException()
        deleted_callbacks: list[str] = []
        disk_image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageRecordDeleteException):
            disk_image_service.delete_intermediates()

        for name in names:
            assert storage.get_path(name).exists()
            assert storage.get_path(name, thumbnail=True).exists()
        assert deleted_callbacks == []
        assert _staging_dirs(storage) == []

    def test_one_rollback_failure_does_not_abandon_other_rollbacks(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [
            ("tmp1.png", ""),
            ("tmp2.png", ""),
            ("tmp3.png", ""),
        ]
        tokens = [object(), object(), object()]
        invoker.services.image_files.stage_delete.side_effect = tokens
        invoker.services.image_records.delete_intermediates_by_names.side_effect = ImageRecordDeleteException()
        invoker.services.image_files.rollback_delete.side_effect = [
            ImageFileDeleteException("rollback broken"),
            None,
            None,
        ]
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        with pytest.raises(ImageRecordDeleteException):
            image_service.delete_intermediates()

        # Every staged item must have a rollback attempt, even after one fails.
        rollback_tokens = [call.args[0] for call in invoker.services.image_files.rollback_delete.call_args_list]
        assert rollback_tokens == tokens
        invoker.services.image_files.commit_delete.assert_not_called()
        assert deleted_callbacks == []

    def test_commit_failure_is_logged_and_remaining_commits_attempted(self, image_service: ImageService):
        invoker = image_service._ImageService__invoker  # type: ignore
        invoker.services.image_records.get_intermediates.return_value = [("tmp1.png", ""), ("tmp2.png", "")]
        tokens = [object(), object()]
        invoker.services.image_files.stage_delete.side_effect = tokens
        invoker.services.image_files.commit_delete.side_effect = [ImageFileDeleteException("purge failed"), None]
        deleted_callbacks: list[str] = []
        image_service.on_deleted(deleted_callbacks.append)

        count = image_service.delete_intermediates()

        assert count == 2
        commit_tokens = [call.args[0] for call in invoker.services.image_files.commit_delete.call_args_list]
        assert commit_tokens == tokens
        # Records were deleted, so the deletions are committed and callbacks must fire.
        assert deleted_callbacks == ["tmp1.png", "tmp2.png"]
        invoker.services.logger.error.assert_called()
        invoker.services.image_files.rollback_delete.assert_not_called()


class TestDeleteIntermediatesAgainstRealRecords:
    """delete_intermediates() wired to a real record store, so no stub stands in for the DB decision.

    The mocked tests above can only assert that the service honours whatever the store reports. These
    exercise the real store, which is where the promoted-vs-already-gone distinction is actually made.
    """

    @pytest.fixture
    def wired(self, tmp_path: Path) -> tuple[ImageService, SqliteImageRecordStorage, DiskImageFileStorage]:
        config = InvokeAIAppConfig(use_memory_db=True)
        logger = InvokeAILogger.get_logger(config=config)
        records = SqliteImageRecordStorage(db=create_mock_sqlite_database(config, logger))
        storage = DiskImageFileStorage(tmp_path / "outputs")

        svc = ImageService()
        invoker = MagicMock()
        invoker.services.configuration.pil_compress_level = 1
        invoker.services.image_records = records
        invoker.services.image_files = storage
        storage.start(invoker)
        svc.start(invoker)
        return svc, records, storage

    def _seed(self, records: SqliteImageRecordStorage, storage: DiskImageFileStorage, name: str) -> None:
        records.save(
            image_name=name,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=64,
            height=64,
            has_workflow=False,
            is_intermediate=True,
        )
        _save_image_file(storage, name)

    def _promote_after_staging(
        self,
        records: SqliteImageRecordStorage,
        storage: DiskImageFileStorage,
        monkeypatch,
        image_name: str,
    ) -> None:
        """Promote an image out of intermediate status once its files have been staged.

        Promoting it *before* delete_intermediates() runs would keep it out of the snapshot entirely,
        so it would never reach the retained path these tests are about.
        """
        real_stage_delete = storage.stage_delete

        def stage_then_promote(name: str, *args, **kwargs):
            token = real_stage_delete(name, *args, **kwargs)
            if name == image_name:
                records.update(image_name, ImageRecordChanges(is_intermediate=False))
            return token

        monkeypatch.setattr(storage, "stage_delete", stage_then_promote)

    def test_promoted_image_keeps_record_and_files(self, wired, monkeypatch) -> None:
        svc, records, storage = wired
        self._seed(records, storage, "tmp1.png")
        self._seed(records, storage, "promoted.png")
        self._promote_after_staging(records, storage, monkeypatch, "promoted.png")
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        assert svc.delete_intermediates() == 1

        assert storage.get_path("promoted.png").exists()
        assert records.get("promoted.png").is_intermediate is False
        assert not storage.get_path("tmp1.png").exists()
        assert deleted_callbacks == ["tmp1.png"]
        assert _staging_dirs(storage) == []

    def test_record_removed_by_another_path_does_not_resurrect_its_files(self, wired, monkeypatch) -> None:
        """A record deleted elsewhere while we hold its files must not have those files restored.

        "Not deleted by us" is not the same as "still there". Restoring files for a record that is
        gone orphans them on disk forever: no row references them and no staging dir remains for
        startup recovery to find.
        """
        svc, records, storage = wired
        self._seed(records, storage, "tmp1.png")
        self._seed(records, storage, "gone.png")

        real_stage_delete = storage.stage_delete

        def stage_then_lose_the_record(image_name: str, *args, **kwargs):
            token = real_stage_delete(image_name, *args, **kwargs)
            if image_name == "gone.png":
                # Another path (single-image delete, board delete, maintenance script) removes the
                # record after we have already staged its files.
                records.delete("gone.png")
            return token

        monkeypatch.setattr(storage, "stage_delete", stage_then_lose_the_record)
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        count = svc.delete_intermediates()

        # gone.png's files must be purged, not restored.
        assert not storage.get_path("gone.png").exists()
        assert not storage.get_path("gone.png", thumbnail=True).exists()
        assert not storage.get_path("tmp1.png").exists()
        assert _staging_dirs(storage) == []
        # Only the record this call actually removed is counted and announced.
        assert count == 1
        assert deleted_callbacks == ["tmp1.png"]

    def test_retained_record_deleted_before_rollback_does_not_resurrect_its_files(self, wired, monkeypatch) -> None:
        """A retained record can still be deleted while the commit/rollback loop is running.

        The loop can work through thousands of names before reaching a given token, so "retained at
        DB-call time" is not enough to justify restoring files. Restoring them against a record that
        has since been deleted strands them: no row refers to them and the staging dir is gone, so
        startup recovery can never find them.
        """
        svc, records, storage = wired
        self._seed(records, storage, "tmp1.png")
        self._seed(records, storage, "promoted.png")
        self._promote_after_staging(records, storage, monkeypatch, "promoted.png")

        real_delete_by_names = records.delete_intermediates_by_names

        def delete_then_lose_the_retained_record(names: list[str]):
            deleted, retained = real_delete_by_names(names)
            # Another path deletes the promoted image after we decided to keep its files.
            for name in retained:
                records.delete(name)
            return deleted, retained

        monkeypatch.setattr(records, "delete_intermediates_by_names", delete_then_lose_the_retained_record)
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        count = svc.delete_intermediates()

        assert not storage.get_path("promoted.png").exists()
        assert not storage.get_path("promoted.png", thumbnail=True).exists()
        assert _staging_dirs(storage) == []
        # Only this call's own deletion is counted and announced.
        assert count == 1
        assert deleted_callbacks == ["tmp1.png"]

    def test_lookup_failure_during_rollback_check_keeps_the_files(self, wired, monkeypatch) -> None:
        """If we cannot tell whether the record survived, keep the files — a lost file is final."""
        svc, records, storage = wired
        self._seed(records, storage, "promoted.png")
        self._promote_after_staging(records, storage, monkeypatch, "promoted.png")

        def unavailable(image_name: str):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(records, "get", unavailable)
        deleted_callbacks: list[str] = []
        svc.on_deleted(deleted_callbacks.append)

        assert svc.delete_intermediates() == 0

        assert storage.get_path("promoted.png").exists()
        assert _staging_dirs(storage) == []
        assert deleted_callbacks == []
