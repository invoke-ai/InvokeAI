"""Tests for SqlModelImageRecordStorage."""

import pytest

from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageRecordChanges,
    ImageRecordNotFoundException,
    ResourceOrigin,
)
from invokeai.app.services.image_records.image_records_sqlmodel import SqlModelImageRecordStorage
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


@pytest.fixture
def storage(db: SqliteDatabase) -> SqlModelImageRecordStorage:
    return SqlModelImageRecordStorage(db=db)


def _save(storage, name="img1", category=ImageCategory.GENERAL, intermediate=False, starred=False, user_id="user1"):
    return storage.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=category,
        width=512,
        height=512,
        has_workflow=False,
        is_intermediate=intermediate,
        starred=starred,
        user_id=user_id,
    )


def test_save_and_get(storage):
    _save(storage, "img1")
    record = storage.get("img1")
    assert record.image_name == "img1"
    assert record.width == 512
    assert record.image_category == ImageCategory.GENERAL


def test_get_nonexistent(storage):
    with pytest.raises(ImageRecordNotFoundException):
        storage.get("nonexistent")


def test_save_returns_datetime(storage):
    created_at = _save(storage, "img1")
    assert created_at is not None


def test_update_category(storage):
    _save(storage, "img1")
    storage.update("img1", ImageRecordChanges(image_category=ImageCategory.MASK))
    record = storage.get("img1")
    assert record.image_category == ImageCategory.MASK


def test_update_starred(storage):
    _save(storage, "img1")
    storage.update("img1", ImageRecordChanges(starred=True))
    record = storage.get("img1")
    assert record.starred is True


def test_update_is_intermediate(storage):
    _save(storage, "img1")
    storage.update("img1", ImageRecordChanges(is_intermediate=True))
    record = storage.get("img1")
    assert record.is_intermediate is True


def test_delete(storage):
    _save(storage, "img1")
    storage.delete("img1")
    with pytest.raises(ImageRecordNotFoundException):
        storage.get("img1")


def test_delete_many(storage):
    _save(storage, "img1")
    _save(storage, "img2")
    _save(storage, "img3")
    storage.delete_many(["img1", "img3"])
    with pytest.raises(ImageRecordNotFoundException):
        storage.get("img1")
    assert storage.get("img2").image_name == "img2"


def test_get_many_pagination(storage):
    for i in range(5):
        _save(storage, f"img{i}")

    result = storage.get_many(offset=0, limit=3)
    assert len(result.items) == 3
    assert result.total == 5


def test_get_many_filter_by_category(storage):
    _save(storage, "img1", category=ImageCategory.GENERAL)
    _save(storage, "img2", category=ImageCategory.MASK)

    result = storage.get_many(categories=[ImageCategory.GENERAL])
    assert all(r.image_category == ImageCategory.GENERAL for r in result.items)


def test_get_many_starred_first(storage):
    _save(storage, "img1", starred=False)
    _save(storage, "img2", starred=True)

    result = storage.get_many(starred_first=True, order_dir=SQLiteDirection.Descending)
    assert result.items[0].starred is True


def test_get_intermediates_count(storage):
    _save(storage, "img1", intermediate=True)
    _save(storage, "img2", intermediate=True)
    _save(storage, "img3", intermediate=False)
    assert storage.get_intermediates_count() == 2


def test_get_intermediates_count_by_user(storage):
    _save(storage, "img1", intermediate=True, user_id="user1")
    _save(storage, "img2", intermediate=True, user_id="user2")
    assert storage.get_intermediates_count(user_id="user1") == 1


def test_delete_intermediates(storage):
    _save(storage, "img1", intermediate=True)
    _save(storage, "img2", intermediate=True)
    _save(storage, "img3", intermediate=False)
    deleted = storage.delete_intermediates()
    assert set(deleted) == {"img1", "img2"}
    assert storage.get("img3").image_name == "img3"


def test_get_user_id(storage):
    _save(storage, "img1", user_id="user42")
    assert storage.get_user_id("img1") == "user42"
    assert storage.get_user_id("nonexistent") is None


def test_get_image_names(storage):
    _save(storage, "img1")
    _save(storage, "img2", starred=True)

    result = storage.get_image_names(starred_first=True)
    assert result.total_count == 2
    assert result.starred_count == 1
    # Starred should come first
    assert result.image_names[0] == "img2"
