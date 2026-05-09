"""DB-backed tests for SqliteImageRecordStorage.

Verifies that image_subfolder round-trips correctly through save(), get(),
get_many(), and delete_intermediates() against a real (in-memory) SQLite database.
"""

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def store() -> SqliteImageRecordStorage:
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    db = create_mock_sqlite_database(config, logger)
    return SqliteImageRecordStorage(db=db)


def _save(store: SqliteImageRecordStorage, name: str, subfolder: str = "", is_intermediate: bool = False) -> None:
    store.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        has_workflow=False,
        is_intermediate=is_intermediate,
        image_subfolder=subfolder,
    )


class TestImageSubfolderRoundTrip:
    """save() -> get() preserves image_subfolder."""

    def test_default_empty_subfolder(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "img_default.png")
        record = store.get("img_default.png")
        assert record.image_subfolder == ""

    def test_custom_subfolder(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "img_sub.png", subfolder="2026/04/11")
        record = store.get("img_sub.png")
        assert record.image_subfolder == "2026/04/11"

    def test_nested_subfolder(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "img_nested.png", subfolder="a/b/c/d")
        record = store.get("img_nested.png")
        assert record.image_subfolder == "a/b/c/d"


class TestGetManySubfolder:
    """get_many() deserializes image_subfolder for every row."""

    def test_get_many_returns_subfolders(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "flat.png", subfolder="")
        _save(store, "dated.png", subfolder="2026/01")
        _save(store, "hashed.png", subfolder="ab")

        result = store.get_many(limit=10, order_dir=SQLiteDirection.Ascending)
        by_name = {r.image_name: r.image_subfolder for r in result.items}

        assert by_name["flat.png"] == ""
        assert by_name["dated.png"] == "2026/01"
        assert by_name["hashed.png"] == "ab"


class TestDeleteIntermediatesSubfolder:
    """delete_intermediates() returns (name, subfolder) pairs and removes rows."""

    def test_returns_subfolder_pairs(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "keep.png", subfolder="general", is_intermediate=False)
        _save(store, "tmp1.png", subfolder="intermediate", is_intermediate=True)
        _save(store, "tmp2.png", subfolder="intermediate", is_intermediate=True)

        pairs = store.delete_intermediates()

        # Should return only intermediate images with their subfolders
        assert len(pairs) == 2
        names_and_subs = {(name, sub) for name, sub in pairs}
        assert ("tmp1.png", "intermediate") in names_and_subs
        assert ("tmp2.png", "intermediate") in names_and_subs

        # Non-intermediate image should still exist
        record = store.get("keep.png")
        assert record.image_subfolder == "general"

    def test_intermediates_are_deleted(self, store: SqliteImageRecordStorage) -> None:
        _save(store, "tmp.png", subfolder="x", is_intermediate=True)
        store.delete_intermediates()

        from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException

        with pytest.raises(ImageRecordNotFoundException):
            store.get("tmp.png")
