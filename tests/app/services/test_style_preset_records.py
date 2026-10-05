"""Style preset records on every database backend: which presets an account sees, and the bundled ones kept in step
with the file that ships them."""

import json
from pathlib import Path
from unittest import mock

import pytest

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.style_presets import StylePresetQueries
from invokeai.app.services.style_preset_records import style_preset_records_default
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetData,
    PresetType,
    StylePresetChanges,
    StylePresetNotFoundError,
    StylePresetWithoutId,
)
from invokeai.app.services.style_preset_records.style_preset_records_default import (
    SYSTEM_USER_ID,
    StylePresetRecordsStorage,
)

ALICE = "alice"
BOB = "bob"


@pytest.fixture
def presets(database: Database) -> StylePresetRecordsStorage:
    return StylePresetRecordsStorage(database)


def _preset(name: str, *, type: PresetType = PresetType.User, is_public: bool = False) -> StylePresetWithoutId:
    return StylePresetWithoutId(
        name=name,
        preset_data=PresetData(positive_prompt=f"{name}, {{prompt}}", negative_prompt="blurry"),
        type=type,
        is_public=is_public,
    )


def _bundled_names() -> list[str]:
    path = Path(style_preset_records_default.__file__).parent / "default_style_presets.json"
    return sorted(preset["name"] for preset in json.loads(path.read_text(encoding="utf-8")))


def test_an_account_sees_its_own_the_shared_and_the_bundled_presets(presets: StylePresetRecordsStorage) -> None:
    own = presets.create(_preset("alice private"), ALICE)
    shared = presets.create(_preset("bob shared", is_public=True), BOB)
    hidden = presets.create(_preset("bob private"), BOB)
    bundled = presets.create(_preset("bundled", type=PresetType.Default), SYSTEM_USER_ID)

    def visible(user_id: str | None, is_admin: bool = False) -> set[str]:
        return {preset.id for preset in presets.get_many(user_id=user_id, is_admin=is_admin)}

    assert visible(ALICE) == {own.id, shared.id, bundled.id}
    assert visible(BOB) == {shared.id, hidden.id, bundled.id}
    # Without an account only what everyone may see is listed.
    assert visible(None) == {shared.id, bundled.id}
    assert visible(ALICE, is_admin=True) == {own.id, shared.id, hidden.id, bundled.id}


def test_a_type_filter_narrows_what_the_account_sees(presets: StylePresetRecordsStorage) -> None:
    own = presets.create(_preset("mine"), ALICE)
    theirs = presets.create(_preset("theirs"), BOB)
    bundled = presets.create(_preset("bundled", type=PresetType.Default), SYSTEM_USER_ID)

    assert [preset.id for preset in presets.get_many(type=PresetType.User, user_id=ALICE)] == [own.id]
    assert [preset.id for preset in presets.get_many(type=PresetType.Default, user_id=ALICE)] == [bundled.id]
    # The export: every account's own presets, and no bundled one.
    exported = presets.get_many(type=PresetType.User, user_id=ALICE, is_admin=True)
    assert {preset.id for preset in exported} == {own.id, theirs.id}


def test_presets_are_listed_by_name_ignoring_case_and_equal_names_by_id(
    database: Database, presets: StylePresetRecordsStorage
) -> None:
    # Stored against id order, so that only the tie-break lists equal names by id.
    named = (("p1", "beta"), ("p4", "alpha"), ("p3", "ALPHA"), ("p2", "Alpha"))
    database.queries.style_presets.insert({preset_id: _preset(name) for preset_id, name in named}, ALICE)

    assert [preset.id for preset in presets.get_many(user_id=ALICE)] == ["p2", "p3", "p4", "p1"]


def test_a_created_preset_is_stored_as_returned(presets: StylePresetRecordsStorage) -> None:
    created = presets.create(_preset("shared", is_public=True), ALICE)

    assert (created.name, created.type, created.user_id, created.is_public) == ("shared", PresetType.User, ALICE, True)
    assert presets.get(created.id) == created


def test_an_update_sets_only_the_fields_it_names_of_only_that_preset(presets: StylePresetRecordsStorage) -> None:
    created = presets.create(_preset("before"), ALICE)
    bystander = presets.create(_preset("bystander", is_public=True), BOB)
    data = PresetData(positive_prompt="after, {prompt}", negative_prompt="")

    renamed = presets.update(created.id, StylePresetChanges(name="after", type=None))
    shared = presets.update(created.id, StylePresetChanges(preset_data=data, is_public=True, type=None))
    unshared = presets.update(created.id, StylePresetChanges(is_public=False, type=None))

    assert (renamed.name, renamed.preset_data, renamed.is_public) == ("after", created.preset_data, False)
    assert (shared.name, shared.preset_data, shared.is_public) == ("after", data, True)
    assert (unshared.name, unshared.preset_data, unshared.is_public) == ("after", data, False)
    assert presets.get(created.id) == unshared
    # Made private again, it is hidden from the other accounts again.
    assert created.id not in {preset.id for preset in presets.get_many(user_id=BOB)}
    assert presets.get(bystander.id) == bystander


def test_deleting_removes_only_that_preset_and_deleting_it_again_is_harmless(
    presets: StylePresetRecordsStorage,
) -> None:
    created = presets.create(_preset("short-lived"), ALICE)
    bystander = presets.create(_preset("bystander"), BOB)

    presets.delete(created.id)
    presets.delete(created.id)

    with pytest.raises(StylePresetNotFoundError):
        presets.get(created.id)
    with pytest.raises(StylePresetNotFoundError):
        presets.update(created.id, StylePresetChanges(name="revived", type=None))
    assert presets.get(bystander.id) == bystander


def test_an_import_stores_every_preset_for_the_account(presets: StylePresetRecordsStorage) -> None:
    presets.create_many([_preset("one"), _preset("two")], ALICE)
    # An import of an empty file stores nothing rather than failing.
    presets.create_many([], ALICE)

    imported = presets.get_many(user_id=ALICE)

    assert [(preset.name, preset.user_id, preset.type) for preset in imported] == [
        ("one", ALICE, PresetType.User),
        ("two", ALICE, PresetType.User),
    ]
    assert len({preset.id for preset in imported}) == 2


def test_starting_replaces_the_bundled_presets_and_keeps_the_accounts_own(presets: StylePresetRecordsStorage) -> None:
    own = presets.create(_preset("mine"), ALICE)
    # In single-user mode every preset belongs to the system user, so only the type tells the bundled ones apart.
    single_user = presets.create(_preset("single-user preset"), SYSTEM_USER_ID)
    retired = presets.create(_preset("retired", type=PresetType.Default), SYSTEM_USER_ID)

    presets.start(mock.Mock())
    # A restart stores them again in place of the first copies, not beside them.
    presets.start(mock.Mock())

    bundled = presets.get_many(type=PresetType.Default, is_admin=True)
    assert sorted(preset.name for preset in bundled) == _bundled_names()
    assert {preset.user_id for preset in bundled} == {SYSTEM_USER_ID}
    assert presets.get(own.id) == own
    assert presets.get(single_user.id) == single_user
    with pytest.raises(StylePresetNotFoundError):
        presets.get(retired.id)


def test_a_failed_sync_keeps_the_bundled_presets_it_had(
    presets: StylePresetRecordsStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    presets.start(mock.Mock())

    def failing_insert(self: StylePresetQueries, bundled: object, user_id: object) -> None:
        raise RuntimeError("the database went away")

    monkeypatch.setattr(StylePresetQueries, "insert", failing_insert)
    with pytest.raises(RuntimeError):
        presets.start(mock.Mock())

    assert (
        sorted(preset.name for preset in presets.get_many(type=PresetType.Default, is_admin=True)) == _bundled_names()
    )
