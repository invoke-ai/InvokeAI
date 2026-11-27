"""Client state persistence on every database backend."""

import threading
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import insert, select

from invokeai.app.services.client_state_persistence.client_state_persistence_default import ClientStatePersistence
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.client_state import ClientStateQueries
from invokeai.app.services.shared.database.schema.client_state import client_state
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.users.users_common import SYSTEM_USER_ID, UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from tests.fixtures.database import external_test_db_url

server_only = pytest.mark.skipif(
    external_test_db_url() is None, reason="needs a MySQL or MariaDB server (INVOKEAI_TEST_DB_URL)"
)

# Names an image and a video; client state values hold whatever JSON the web client writes.
DOCUMENT = '{"layers": [{"image_name": "a.png"}], "video": {"video_name": "v.mp4"}}'


@pytest.fixture
def store(database: Database) -> ClientStatePersistence:
    return ClientStatePersistence(database)


@pytest.fixture
def other_user(database: Database) -> str:
    request = UserCreateRequest(email="other@test.com", display_name="Other", password="Password123")
    return UserService(database).create(request, strict_password_checking=False).user_id


def _insert(database: Database, user_id: str, key: str, updated_at: str) -> None:
    with database.begin(write=True) as conn:
        conn.execute(insert(client_state).values(user_id=user_id, key=key, value="{}", updated_at=updated_at))


def _references(database: Database) -> set[tuple[Any, ...]]:
    """Every row of the reference index: (owner_kind, user_id, owner_id, media_kind, media_name)."""
    with database.begin(write=False) as conn:
        return {tuple(row) for row in conn.execute(select(media_references))}


def _client_state_reference(user_id: str, key: str, media_kind: str, media_name: str) -> tuple[str, ...]:
    return ("client_state", user_id, key, media_kind, media_name)


def test_a_value_is_kept_per_account_and_replaced_by_the_next(store: ClientStatePersistence, other_user: str) -> None:
    store.set_by_key(SYSTEM_USER_ID, "key", "first")
    store.set_by_key(SYSTEM_USER_ID, "key", "second")
    store.set_by_key(other_user, "key", "theirs")

    assert store.get_by_key(SYSTEM_USER_ID, "key") == "second"
    assert store.get_by_key(other_user, "key") == "theirs"
    assert store.get_by_key(SYSTEM_USER_ID, "missing") is None


def test_replacing_a_value_renews_when_it_was_written(database: Database, store: ClientStatePersistence) -> None:
    _insert(database, SYSTEM_USER_ID, "key", "2000-01-01 00:00:00.000")

    store.set_by_key(SYSTEM_USER_ID, "key", "new")

    with database.begin(write=False) as conn:
        assert conn.execute(select(client_state.c.updated_at)).scalar_one() > "2000-01-01 00:00:00.000"


def test_keys_with_a_prefix_list_most_recently_written_first_ignoring_case(
    database: Database, store: ClientStatePersistence, other_user: str
) -> None:
    _insert(database, SYSTEM_USER_ID, "snap:old", "2026-01-01 00:00:00.000")
    _insert(database, SYSTEM_USER_ID, "SNAP:upper", "2026-01-03 00:00:00.000")
    _insert(database, SYSTEM_USER_ID, "snap:new", "2026-01-02 00:00:00.000")
    _insert(database, SYSTEM_USER_ID, "other:key", "2026-01-04 00:00:00.000")
    _insert(database, other_user, "snap:theirs", "2026-01-05 00:00:00.000")

    assert store.get_keys_by_prefix(SYSTEM_USER_ID, "snap:") == ["SNAP:upper", "snap:new", "snap:old"]


def test_like_wildcards_in_a_prefix_match_only_themselves(store: ClientStatePersistence) -> None:
    for key in ("a%b", "axb", "a_c", "ayc", "a\\d", "aed"):
        store.set_by_key(SYSTEM_USER_ID, key, "{}")

    assert store.get_keys_by_prefix(SYSTEM_USER_ID, "a%") == ["a%b"]
    assert store.get_keys_by_prefix(SYSTEM_USER_ID, "a_") == ["a_c"]
    assert store.get_keys_by_prefix(SYSTEM_USER_ID, "a\\") == ["a\\d"]


def test_media_references_follow_the_values_that_name_them(database: Database, store: ClientStatePersistence) -> None:
    def references(*names: tuple[str, str, str]) -> set[tuple[str, ...]]:
        return {_client_state_reference(SYSTEM_USER_ID, key, kind, name) for key, kind, name in names}

    store.set_by_key(SYSTEM_USER_ID, "canvas", DOCUMENT)
    store.set_by_key(SYSTEM_USER_ID, "staging", '{"imageName": "c.png"}')
    assert _references(database) == references(
        ("canvas", "image", "a.png"), ("canvas", "video", "v.mp4"), ("staging", "image", "c.png")
    )

    store.set_by_key(SYSTEM_USER_ID, "canvas", '{"imageName": "b.png"}')
    assert _references(database) == references(("canvas", "image", "b.png"), ("staging", "image", "c.png"))

    store.delete_by_key(SYSTEM_USER_ID, "canvas")
    assert _references(database) == references(("staging", "image", "c.png"))

    store.delete(SYSTEM_USER_ID)
    assert _references(database) == set()
    assert store.get_keys_by_prefix(SYSTEM_USER_ID, "") == []


def test_one_accounts_client_state_leaves_other_accounts_alone(
    database: Database, store: ClientStatePersistence, other_user: str
) -> None:
    # The web client uses the same keys for every account.
    store.set_by_key(other_user, "canvas", DOCUMENT)
    store.set_by_key(other_user, "other", DOCUMENT)
    store.set_by_key(SYSTEM_USER_ID, "canvas", DOCUMENT)
    project_reference = ("project", SYSTEM_USER_ID, "project", "image", "p.png")
    with database.begin(write=True) as conn:
        conn.execute(
            insert(media_references).values(dict(zip(media_references.c.keys(), project_reference, strict=False)))
        )

    store.set_by_key(SYSTEM_USER_ID, "canvas", "{}")
    store.delete_by_key(SYSTEM_USER_ID, "canvas")
    store.set_by_key(SYSTEM_USER_ID, "other", DOCUMENT)
    store.delete(SYSTEM_USER_ID)

    assert store.get_by_key(other_user, "canvas") == DOCUMENT
    assert store.get_by_key(other_user, "other") == DOCUMENT
    assert _references(database) == {
        _client_state_reference(other_user, key, kind, name)
        for key in ("canvas", "other")
        for kind, name in (("image", "a.png"), ("video", "v.mp4"))
    } | {project_reference}


def _pause_after(monkeypatch: pytest.MonkeyPatch, method: str, while_paused: Callable[[], None]) -> Callable[[], None]:
    """Makes `ClientStateQueries.<method>` run `while_paused` on another thread right after it, inside its
    transaction, and wait for it. Returns a function to call afterwards, which raises what that thread raised."""
    real = getattr(ClientStateQueries, method)
    finished = threading.Event()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            while_paused()
        except BaseException as e:  # noqa: BLE001 - raised again by the returned function
            errors.append(e)
        finally:
            finished.set()

    def then_pause(self: ClientStateQueries, *args: Any, **kwargs: Any) -> Any:
        result = real(self, *args, **kwargs)
        threading.Thread(target=run).start()
        assert finished.wait(timeout=30)
        return result

    monkeypatch.setattr(ClientStateQueries, method, then_pause)

    def check() -> None:
        assert finished.is_set()
        if errors:
            raise errors[0]

    return check


@server_only
def test_deleting_a_key_while_it_is_first_written_leaves_the_new_value_its_references(
    database: Database, store: ClientStatePersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a server the delete finds no row to lock, so the first write goes ahead while the delete's transaction is
    # still open. (SQLite runs one transaction at a time; there the write would wait for the delete.)
    check = _pause_after(monkeypatch, "delete", lambda: store.set_by_key(SYSTEM_USER_ID, "canvas", DOCUMENT))

    store.delete_by_key(SYSTEM_USER_ID, "canvas")
    check()

    assert store.get_by_key(SYSTEM_USER_ID, "canvas") == DOCUMENT
    assert _references(database) == {
        _client_state_reference(SYSTEM_USER_ID, "canvas", "image", "a.png"),
        _client_state_reference(SYSTEM_USER_ID, "canvas", "video", "v.mp4"),
    }


@server_only
def test_deleting_all_values_while_a_new_key_is_written_leaves_that_key_its_references(
    database: Database, store: ClientStatePersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.set_by_key(SYSTEM_USER_ID, "old", '{"imageName": "old.png"}')
    check = _pause_after(monkeypatch, "delete_all", lambda: store.set_by_key(SYSTEM_USER_ID, "new", DOCUMENT))

    store.delete(SYSTEM_USER_ID)
    check()

    assert store.get_keys_by_prefix(SYSTEM_USER_ID, "") == ["new"]
    assert _references(database) == {
        _client_state_reference(SYSTEM_USER_ID, "new", "image", "a.png"),
        _client_state_reference(SYSTEM_USER_ID, "new", "video", "v.mp4"),
    }
