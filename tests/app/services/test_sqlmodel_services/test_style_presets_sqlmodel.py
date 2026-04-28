"""Tests for SqlModelStylePresetRecordsStorage."""

import pytest

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetData,
    StylePresetChanges,
    StylePresetNotFoundError,
    StylePresetWithoutId,
)
from invokeai.app.services.style_preset_records.style_preset_records_sqlmodel import SqlModelStylePresetRecordsStorage


@pytest.fixture
def storage(db: SqliteDatabase) -> SqlModelStylePresetRecordsStorage:
    return SqlModelStylePresetRecordsStorage(db=db)


def _make_preset(name: str = "Test Preset", preset_type: str = "user") -> StylePresetWithoutId:
    return StylePresetWithoutId(
        name=name,
        preset_data=PresetData(positive_prompt="a cat", negative_prompt=""),
        type=preset_type,
    )


def test_create_and_get(storage):
    preset = storage.create(_make_preset("My Preset"))
    assert preset.name == "My Preset"

    fetched = storage.get(preset.id)
    assert fetched.name == "My Preset"


def test_get_nonexistent(storage):
    with pytest.raises(StylePresetNotFoundError):
        storage.get("nonexistent")


def test_update_name(storage):
    preset = storage.create(_make_preset("Original"))
    updated = storage.update(preset.id, StylePresetChanges(name="Updated", type=None))
    assert updated.name == "Updated"


def test_update_preset_data(storage):
    preset = storage.create(_make_preset())
    new_data = PresetData(positive_prompt="a dog", negative_prompt="ugly")
    updated = storage.update(preset.id, StylePresetChanges(preset_data=new_data, type=None))
    assert updated.preset_data.positive_prompt == "a dog"


def test_delete(storage):
    preset = storage.create(_make_preset())
    storage.delete(preset.id)
    with pytest.raises(StylePresetNotFoundError):
        storage.get(preset.id)


def test_get_many(storage):
    storage.create(_make_preset("Preset A"))
    storage.create(_make_preset("Preset B"))
    results = storage.get_many()
    # Filter out any default presets
    user_presets = [r for r in results if r.type == "user"]
    assert len(user_presets) == 2


def test_get_many_filter_by_type(storage):
    storage.create(_make_preset("User Preset", "user"))
    # There may be defaults loaded; just verify filtering works
    user_presets = storage.get_many(type="user")
    assert all(p.type == "user" for p in user_presets)


def test_create_many(storage):
    presets = [_make_preset(f"Preset {i}") for i in range(3)]
    storage.create_many(presets)
    all_presets = storage.get_many(type="user")
    assert len(all_presets) >= 3


def test_get_many_ordered_by_name(storage):
    storage.create(_make_preset("Zebra"))
    storage.create(_make_preset("Alpha"))
    results = storage.get_many(type="user")
    names = [r.name for r in results]
    assert names == sorted(names, key=str.lower)
