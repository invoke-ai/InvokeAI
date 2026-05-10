"""Storage-layer tests for system_prompt_records.

Covers the per-user scoping semantics added on top of the original CRUD:
- get_many returns own + public for a user_id, all rows for None (admin)
- update/delete with a non-owner user_id is a clean no-op (NotFound), no mutation
- the migration-seeded defaults (user_id='system', is_public=TRUE) are visible to every user
"""

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    SystemPromptChanges,
    SystemPromptNotFoundError,
    SystemPromptWithoutId,
)
from invokeai.app.services.system_prompt_records.system_prompt_records_sqlite import (
    SqliteSystemPromptRecordsStorage,
)
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


def _storage() -> SqliteSystemPromptRecordsStorage:
    config = InvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    db = create_mock_sqlite_database(config, InvokeAILogger.get_logger())
    return SqliteSystemPromptRecordsStorage(db=db)


def test_seeded_defaults_visible_to_every_user() -> None:
    svc = _storage()
    seeded = svc.get_many(user_id=None)
    assert len(seeded) >= 1
    assert all(p.user_id == "system" and p.is_public for p in seeded)
    seeded_ids = {p.id for p in seeded}

    # Any non-admin user sees the same system rows because they are public.
    for_alice = svc.get_many(user_id="alice")
    assert seeded_ids.issubset({p.id for p in for_alice})


def test_private_prompt_hidden_from_other_user_visible_to_owner_and_admin() -> None:
    svc = _storage()
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


def test_public_prompt_visible_to_everyone() -> None:
    svc = _storage()
    bob_pub = svc.create(
        SystemPromptWithoutId(name="bob shared", content="x"),
        user_id="bob",
        is_public=True,
    )
    alice_view = svc.get_many(user_id="alice")
    assert bob_pub.id in {p.id for p in alice_view}


def test_update_with_non_owner_user_id_raises_and_does_not_mutate() -> None:
    svc = _storage()
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


def test_delete_with_non_owner_user_id_is_noop() -> None:
    svc = _storage()
    alice = svc.create(
        SystemPromptWithoutId(name="alice secret", content="x"),
        user_id="alice",
        is_public=False,
    )

    # (d) Non-owner delete silently no-ops; row remains.
    svc.delete(alice.id, user_id="bob")
    assert svc.get(alice.id).id == alice.id


def test_admin_can_delete_any_row() -> None:
    svc = _storage()
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


def test_owner_can_flip_is_public() -> None:
    svc = _storage()
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
