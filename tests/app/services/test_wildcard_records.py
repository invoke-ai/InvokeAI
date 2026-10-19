import pytest
from dynamicprompts.generators import CombinatorialPromptGenerator
from sqlalchemy import delete, insert

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import ForeignKeyViolation
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.wildcard_records.wildcard_records_common import (
    WildcardChanges,
    WildcardNameConflictError,
    WildcardNotFoundError,
    WildcardWithoutId,
    build_wildcard_manager,
)
from invokeai.app.services.wildcard_records.wildcard_records_default import WildcardRecordsStorage
from invokeai.app.util.dynamicprompts import expand_dynamic_prompt


@pytest.fixture
def store(database: Database) -> WildcardRecordsStorage:
    # Wildcards cascade from their owner, so the owners have to exist. Inserted
    # directly rather than through UserService: these tests are about storage, and
    # the only columns the foreign key needs are the ones without a default.
    with database.begin(write=True) as conn:
        conn.execute(
            insert(users),
            [
                {"user_id": user_id, "email": f"{user_id}@example.com", "password_hash": "unused"}
                for user_id in ("user-1", "user-2")
            ],
        )
    return WildcardRecordsStorage(database)


def test_create_and_list_is_scoped_to_the_owner(store: WildcardRecordsStorage) -> None:
    store.create(WildcardWithoutId(name="colors", values=["red", "green"]), user_id="user-1")
    store.create(WildcardWithoutId(name="animals", values=["cat"]), user_id="user-2")

    assert [w.name for w in store.get_many("user-1")] == ["colors"]
    assert [w.name for w in store.get_many("user-2")] == ["animals"]


def test_get_many_is_ordered_by_name(store: WildcardRecordsStorage) -> None:
    for name in ["zebra", "apple", "mango"]:
        store.create(WildcardWithoutId(name=name, values=["x"]), user_id="user-1")

    assert [w.name for w in store.get_many("user-1")] == ["apple", "mango", "zebra"]


def test_a_name_is_unique_per_user_but_not_across_users(store: WildcardRecordsStorage) -> None:
    store.create(WildcardWithoutId(name="colors", values=["red"]), user_id="user-1")
    store.create(WildcardWithoutId(name="colors", values=["blue"]), user_id="user-2")

    with pytest.raises(WildcardNameConflictError):
        store.create(WildcardWithoutId(name="colors", values=["green"]), user_id="user-1")


def test_a_created_wildcard_is_stored_as_returned(store: WildcardRecordsStorage) -> None:
    created = store.create(WildcardWithoutId(name="colors", values=[" red ", "green"]), user_id="user-1")

    assert (created.name, created.values, created.user_id) == ("colors", ["red", "green"], "user-1")
    assert store.get(created.id) == created


def test_update_changes_name_and_values_of_only_that_wildcard(store: WildcardRecordsStorage) -> None:
    created = store.create(WildcardWithoutId(name="colors", values=["red"]), user_id="user-1")
    bystander = store.create(WildcardWithoutId(name="animals", values=["cat"]), user_id="user-2")

    updated = store.update(created.id, WildcardChanges(name="colours", values=["red", "green"]))

    assert updated.name == "colours"
    assert updated.values == ["red", "green"]
    assert store.get(bystander.id) == bystander


def test_update_to_a_taken_name_conflicts(store: WildcardRecordsStorage) -> None:
    store.create(WildcardWithoutId(name="colors", values=["red"]), user_id="user-1")
    other = store.create(WildcardWithoutId(name="animals", values=["cat"]), user_id="user-1")

    with pytest.raises(WildcardNameConflictError):
        store.update(other.id, WildcardChanges(name="colors"))


def test_delete_removes_the_record(store: WildcardRecordsStorage) -> None:
    created = store.create(WildcardWithoutId(name="colors", values=["red"]), user_id="user-1")
    bystander = store.create(WildcardWithoutId(name="colors", values=["red"]), user_id="user-2")

    store.delete(created.id)

    assert store.get_many("user-1") == []
    assert store.get_many("user-2") == [bystander]
    with pytest.raises(WildcardNotFoundError):
        store.get(created.id)


def test_deleting_the_owner_takes_their_wildcards_with_it(database: Database, store: WildcardRecordsStorage) -> None:
    # `UserService.delete` is a bare DELETE FROM users; without the cascade these
    # rows would survive their owner, unreachable but forever.
    kept = store.create(WildcardWithoutId(name="colors", values=["red"]), user_id="user-2")
    orphaned = store.create(WildcardWithoutId(name="colors", values=["red"]), user_id="user-1")

    with database.begin(write=True) as conn:
        conn.execute(delete(users).where(users.c.user_id == "user-1"))

    with pytest.raises(WildcardNotFoundError):
        store.get(orphaned.id)
    assert [w.id for w in store.get_many("user-2")] == [kept.id]


def test_an_unknown_owner_is_not_reported_as_a_name_conflict(store: WildcardRecordsStorage) -> None:
    # The name is free; it is the owner that does not exist. Mapping this to a
    # conflict would send the caller renaming a wildcard that is fine.
    with pytest.raises(ForeignKeyViolation):
        store.create(WildcardWithoutId(name="colors", values=["red"]), user_id="ghost")


@pytest.mark.parametrize(
    "name",
    [
        "",
        "  ",
        "has space",
        # A leading or trailing underscore runs into the `__` delimiters: `__trailing___` is
        # ambiguous, so the reference could not be parsed back reliably.
        "trailing_",
        "_leading",
        "nested/_leading",
        "trailing/",
        "a/../b",
        "x" * 129,
    ],
)
def test_invalid_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        WildcardWithoutId(name=name, values=["x"])


@pytest.mark.parametrize("name", ["colors", "animals/dogs", "a-b_c", "has__inner__underscores", "deeply/nested/name"])
def test_valid_names_are_accepted(name: str) -> None:
    assert WildcardWithoutId(name=name, values=["x"]).name == name


def test_values_are_trimmed_and_blanks_dropped() -> None:
    assert WildcardWithoutId(name="colors", values=["  red ", "", "   ", "green"]).values == ["red", "green"]


def test_build_wildcard_manager_resolves_a_users_wildcards(store: WildcardRecordsStorage) -> None:
    store.create(WildcardWithoutId(name="colors", values=["red", "green", "blue"]), user_id="user-1")
    manager = build_wildcard_manager(store.get_many("user-1"))

    assert sorted(manager.get_all_values("colors")) == ["blue", "green", "red"]
    assert sorted(CombinatorialPromptGenerator(manager).generate("a __colors__ ball", max_prompts=5)) == [
        "a blue ball",
        "a green ball",
        "a red ball",
    ]


def test_build_wildcard_manager_omits_valueless_wildcards(store: WildcardRecordsStorage) -> None:
    # A name that is "known" but expands to nothing would silently drop the prompt, so it must stay
    # unresolvable and be reported as missing instead.
    store.create(WildcardWithoutId(name="empty", values=[]), user_id="user-1")
    manager = build_wildcard_manager(store.get_many("user-1"))

    assert (
        expand_dynamic_prompt("a __empty__ ball", max_prompts=4, combinatorial=True, wildcard_manager=manager).error
        == "No values found for wildcard(s): empty"
    )


@pytest.mark.timeout(20)
def test_a_resolvable_wildcard_in_a_variant_generates_instead_of_hanging(
    store: WildcardRecordsStorage,
) -> None:
    store.create(WildcardWithoutId(name="colors", values=["red", "green"]), user_id="user-1")
    manager = build_wildcard_manager(store.get_many("user-1"))

    expanded = expand_dynamic_prompt("{__colors__|x}", max_prompts=5, combinatorial=True, wildcard_manager=manager)

    assert expanded.error is None
    assert sorted(expanded.prompts) == ["green", "red", "x"]


def test_one_users_wildcards_do_not_resolve_for_another(store: WildcardRecordsStorage) -> None:
    store.create(WildcardWithoutId(name="colors", values=["red", "green"]), user_id="user-1")

    other = expand_dynamic_prompt(
        "a __colors__ ball",
        max_prompts=4,
        combinatorial=True,
        wildcard_manager=build_wildcard_manager(store.get_many("user-2")),
    )

    assert other.error == "No values found for wildcard(s): colors"
