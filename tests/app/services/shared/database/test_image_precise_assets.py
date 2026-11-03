"""The precise-asset association (`queries/image_precise_assets.py`): one internal row per image, removed with it."""

import pytest

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import ForeignKeyViolation, UniqueViolation
from invokeai.app.services.shared.database.queries.image_precise_assets import PreciseAssetRecord


def _image(database: Database, name: str) -> None:
    ImageRecordStorage(database).save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=16,
        height=16,
        has_workflow=False,
    )


def _add(database: Database, image_name: str) -> None:
    database.queries.image_precise_assets.insert(
        image_name=image_name, asset_format="raster_safetensors", format_version=1
    )


def test_an_inserted_asset_is_read_back_by_image_name(database: Database) -> None:
    _image(database, "a.png")

    _add(database, "a.png")

    assert database.queries.image_precise_assets.get("a.png") == PreciseAssetRecord("a.png", "raster_safetensors", 1)


def test_an_image_without_an_asset_has_none(database: Database) -> None:
    _image(database, "a.png")

    assert database.queries.image_precise_assets.get("a.png") is None
    assert database.queries.image_precise_assets.get("missing.png") is None


def test_insert_joins_the_callers_transaction(database: Database) -> None:
    _image(database, "a.png")

    with pytest.raises(RuntimeError, match="abort"):
        with database.queries.transaction() as q:
            q.image_precise_assets.insert(image_name="a.png", asset_format="raster_safetensors", format_version=1)
            assert q.image_precise_assets.get("a.png") is not None
            raise RuntimeError("abort")

    assert database.queries.image_precise_assets.get("a.png") is None


def test_a_second_asset_for_an_image_is_a_unique_violation(database: Database) -> None:
    _image(database, "a.png")
    _add(database, "a.png")

    with pytest.raises(UniqueViolation):
        database.queries.image_precise_assets.insert(image_name="a.png", asset_format="other", format_version=2)

    assert database.queries.image_precise_assets.get("a.png") == PreciseAssetRecord("a.png", "raster_safetensors", 1)


def test_an_asset_for_an_unknown_image_is_a_foreign_key_violation(database: Database) -> None:
    with pytest.raises(ForeignKeyViolation):
        _add(database, "missing.png")


def test_deleting_the_image_record_removes_its_asset_row_only(database: Database) -> None:
    _image(database, "a.png")
    _image(database, "b.png")
    _add(database, "a.png")
    _add(database, "b.png")

    ImageRecordStorage(database).delete("a.png")

    assert database.queries.image_precise_assets.get("a.png") is None
    assert database.queries.image_precise_assets.get("b.png") is not None
