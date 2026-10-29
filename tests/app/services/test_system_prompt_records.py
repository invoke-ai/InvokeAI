"""Storage-layer tests for system_prompt_records.

Covers the per-user scoping semantics added on top of the original CRUD:
- get_many returns own + public for a user_id, all rows for None (admin)
- update/delete with a non-owner user_id raises NotFound and leaves the row untouched
- the migration-seeded defaults (user_id='system', is_public=TRUE) are visible to every user
- the per-prompt `max_tokens` cap, whose explicit null clears rather than meaning "no change"
"""

import pytest
from pydantic import ValidationError

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    EXPAND_PROMPT_MAX_TOKENS_MAX,
    SystemPromptChanges,
    SystemPromptNotFoundError,
    SystemPromptWithoutId,
)
from invokeai.app.services.system_prompt_records.system_prompt_records_default import SystemPromptRecordsStorage


@pytest.fixture
def svc(database: Database) -> SystemPromptRecordsStorage:
    return SystemPromptRecordsStorage(database)


def test_seeded_defaults_visible_to_every_user(svc: SystemPromptRecordsStorage) -> None:
    seeded = svc.get_many(user_id=None)
    assert len(seeded) >= 1
    assert all(p.user_id == "system" and p.is_public for p in seeded)
    seeded_ids = {p.id for p in seeded}

    # Any non-admin user sees the same system rows because they are public.
    for_alice = svc.get_many(user_id="alice")
    assert seeded_ids.issubset({p.id for p in for_alice})


def test_private_prompt_hidden_from_other_user_visible_to_owner_and_admin(svc: SystemPromptRecordsStorage) -> None:
    alice = svc.create(
        SystemPromptWithoutId(name="alice secret", content="x"),
        user_id="alice",
        is_public=False,
    )

    # (a) Other users do NOT see alice's private prompt
    bob_view = svc.get_many(user_id="bob")
    assert alice.id not in {p.id for p in bob_view}

    # (b) Owner sees it
    alice_view = svc.get_many(user_id="alice")
    assert alice.id in {p.id for p in alice_view}

    # (b) Admin (user_id=None) sees it
    admin_view = svc.get_many(user_id=None)
    assert alice.id in {p.id for p in admin_view}


def test_public_prompt_visible_to_everyone(svc: SystemPromptRecordsStorage) -> None:
    bob_pub = svc.create(
        SystemPromptWithoutId(name="bob shared", content="x"),
        user_id="bob",
        is_public=True,
    )
    alice_view = svc.get_many(user_id="alice")
    assert bob_pub.id in {p.id for p in alice_view}


def test_update_with_non_owner_user_id_raises_and_does_not_mutate(svc: SystemPromptRecordsStorage) -> None:
    alice = svc.create(
        SystemPromptWithoutId(name="alice secret", content="original"),
        user_id="alice",
        is_public=False,
    )

    # (c) Non-owner update raises NotFound and does not change the row
    try:
        svc.update(alice.id, SystemPromptChanges(content="hijacked"), user_id="bob")
        raise AssertionError("expected SystemPromptNotFoundError")
    except SystemPromptNotFoundError:
        pass
    after = svc.get(alice.id)
    assert after.content == "original"


def test_delete_with_non_owner_user_id_raises_and_does_not_delete(svc: SystemPromptRecordsStorage) -> None:
    alice = svc.create(
        SystemPromptWithoutId(name="alice secret", content="x"),
        user_id="alice",
        is_public=False,
    )

    # (d) Non-owner delete raises NotFound (symmetric with update) and leaves the row alone.
    try:
        svc.delete(alice.id, user_id="bob")
        raise AssertionError("expected SystemPromptNotFoundError")
    except SystemPromptNotFoundError:
        pass
    assert svc.get(alice.id).id == alice.id


def test_delete_of_unknown_id_raises(svc: SystemPromptRecordsStorage) -> None:
    # Unscoped (single-user / admin) deletes must not report success for an id that does not
    # exist -- the router turns this into a 404, matching GET.
    try:
        svc.delete("does-not-exist", user_id=None)
        raise AssertionError("expected SystemPromptNotFoundError")
    except SystemPromptNotFoundError:
        pass


def test_deleting_the_same_row_twice_raises_the_second_time(svc: SystemPromptRecordsStorage) -> None:
    created = svc.create(SystemPromptWithoutId(name="temp", content="x"), user_id="alice", is_public=False)
    svc.delete(created.id, user_id=None)
    try:
        svc.delete(created.id, user_id=None)
        raise AssertionError("expected SystemPromptNotFoundError on second delete")
    except SystemPromptNotFoundError:
        pass


def test_admin_can_delete_any_row(svc: SystemPromptRecordsStorage) -> None:
    alice = svc.create(
        SystemPromptWithoutId(name="alice secret", content="x"),
        user_id="alice",
        is_public=False,
    )
    svc.delete(alice.id, user_id=None)
    try:
        svc.get(alice.id)
        raise AssertionError("expected SystemPromptNotFoundError after admin delete")
    except SystemPromptNotFoundError:
        pass


def test_owner_can_flip_is_public(svc: SystemPromptRecordsStorage) -> None:
    alice = svc.create(
        SystemPromptWithoutId(name="alice", content="x"),
        user_id="alice",
        is_public=False,
    )
    flipped = svc.update(alice.id, SystemPromptChanges(is_public=True), user_id="alice")
    assert flipped.is_public is True

    # Now visible to everyone
    bob_view = svc.get_many(user_id="bob")
    assert alice.id in {p.id for p in bob_view}

    # And private again, hidden again
    unshared = svc.update(alice.id, SystemPromptChanges(is_public=False), user_id="alice")
    assert unshared.is_public is False
    assert alice.id not in {p.id for p in svc.get_many(user_id="bob")}


def test_a_created_prompt_is_stored_as_returned(svc: SystemPromptRecordsStorage) -> None:
    created = svc.create(SystemPromptWithoutId(name="shared", content="x", max_tokens=700), "alice", is_public=True)

    assert (created.user_id, created.is_public, created.max_tokens) == ("alice", True, 700)
    assert created.updated_at == created.created_at
    assert svc.get(created.id) == created


def test_updates_and_deletes_touch_only_the_prompt_they_name(svc: SystemPromptRecordsStorage) -> None:
    alice = svc.create(SystemPromptWithoutId(name="alice", content="x"), user_id="alice")
    alices_other = svc.create(SystemPromptWithoutId(name="alice too", content="x"), user_id="alice")
    bobs = svc.create(SystemPromptWithoutId(name="bob", content="x"), user_id="bob", is_public=True)

    svc.update(alice.id, SystemPromptChanges(content="by its owner"), user_id="alice")
    svc.update(alice.id, SystemPromptChanges(name="by an admin"), user_id=None)
    assert svc.get(alices_other.id) == alices_other
    assert svc.get(bobs.id) == bobs

    svc.delete(alice.id, user_id="alice")
    assert svc.get(alices_other.id) == alices_other
    svc.delete(alices_other.id, user_id=None)
    assert svc.get(bobs.id) == bobs


def test_created_prompt_defaults_to_no_cap(svc: SystemPromptRecordsStorage) -> None:
    # NULL is what tells the client to fall back to the endpoint default.
    created = svc.create(SystemPromptWithoutId(name="alice", content="x"), user_id="alice")
    assert created.max_tokens is None


def test_create_and_update_round_trip_a_cap(svc: SystemPromptRecordsStorage) -> None:
    created = svc.create(SystemPromptWithoutId(name="alice", content="x", max_tokens=500), user_id="alice")
    assert created.max_tokens == 500

    raised = svc.update(created.id, SystemPromptChanges(max_tokens=900), user_id="alice")
    assert raised.max_tokens == 900


def test_update_that_omits_max_tokens_leaves_the_cap_alone(svc: SystemPromptRecordsStorage) -> None:
    created = svc.create(SystemPromptWithoutId(name="alice", content="x", max_tokens=500), user_id="alice")

    renamed = svc.update(created.id, SystemPromptChanges(name="renamed"), user_id="alice")

    assert renamed.max_tokens == 500


def test_explicit_null_clears_the_cap_back_to_the_default(svc: SystemPromptRecordsStorage) -> None:
    # The one field where null is a value rather than "no change" -- without this a cap could be
    # set but never removed.
    created = svc.create(SystemPromptWithoutId(name="alice", content="x", max_tokens=500), user_id="alice")

    cleared = svc.update(created.id, SystemPromptChanges(max_tokens=None), user_id="alice")

    assert cleared.max_tokens is None


def test_non_owner_cannot_change_the_cap(svc: SystemPromptRecordsStorage) -> None:
    alice = svc.create(SystemPromptWithoutId(name="alice", content="x", max_tokens=500), user_id="alice")

    try:
        svc.update(alice.id, SystemPromptChanges(max_tokens=2048), user_id="bob")
        raise AssertionError("expected SystemPromptNotFoundError")
    except SystemPromptNotFoundError:
        pass
    assert svc.get(alice.id).max_tokens == 500


def test_out_of_range_caps_are_rejected_by_the_model() -> None:
    for out_of_range in (0, EXPAND_PROMPT_MAX_TOKENS_MAX + 1):
        with pytest.raises(ValidationError):
            SystemPromptWithoutId(name="alice", content="x", max_tokens=out_of_range)
        with pytest.raises(ValidationError):
            SystemPromptChanges(max_tokens=out_of_range)


def test_an_update_without_changes_still_needs_a_prompt_the_account_may_change(svc: SystemPromptRecordsStorage) -> None:
    # An empty PATCH changes nothing, but must not succeed where one with changes would not.
    alice = svc.create(SystemPromptWithoutId(name="alice", content="x"), user_id="alice")

    assert svc.update(alice.id, SystemPromptChanges(), user_id="alice") == alice
    assert svc.update(alice.id, SystemPromptChanges(), user_id=None) == alice
    with pytest.raises(SystemPromptNotFoundError):
        svc.update(alice.id, SystemPromptChanges(), user_id="bob")
    with pytest.raises(SystemPromptNotFoundError):
        svc.update("does-not-exist", SystemPromptChanges(), user_id=None)


def test_prompts_are_listed_by_name_ignoring_case_and_equal_names_by_id(
    database: Database, svc: SystemPromptRecordsStorage
) -> None:
    # Stored against id order, so that only the tie-break lists equal names by id.
    for prompt_id, name in (("p1", "Beta"), ("p3", "alpha"), ("p2", "Alpha")):
        database.queries.system_prompts.insert(prompt_id, SystemPromptWithoutId(name=name, content="x"), "alice", False)

    for user_id in ("alice", None):
        listed = [prompt.id for prompt in svc.get_many(user_id=user_id) if prompt.user_id == "alice"]
        assert listed == ["p2", "p3", "p1"]
