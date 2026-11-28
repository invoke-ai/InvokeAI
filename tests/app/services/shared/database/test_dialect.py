"""Constructs of `dialect.py` beyond what the query modules exercise on every backend."""

import sys

import pytest
from sqlalchemy import Column, Index, Integer, MetaData, String, Table, bindparam, insert, select
from sqlalchemy.dialects import mysql, postgresql, sqlite

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.dialect import (
    MAX_PAIRS,
    CaseInsensitiveLike,
    InBoundSet,
    LikePattern,
    bound_set,
    case_variants,
    fixed_limit,
    fold_case,
    insert_ignore,
    like_contains,
    like_prefix,
    upsert,
)
from invokeai.app.services.shared.database.schema.client_state import client_state
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.app.services.shared.database.schema.users import users


def test_an_upsert_refuses_a_table_with_a_unique_key_besides_its_primary_key() -> None:
    # MySQL would update the row whichever unique key matched; the other backends only on the primary key.
    with pytest.raises(ValueError, match="unique key besides its primary key"):
        upsert("sqlite", users, update=["display_name"])


def test_an_insert_ignore_refuses_a_table_with_a_unique_key_besides_its_primary_key() -> None:
    # MySQL would skip the row whichever unique key matched; the other backends only on the primary key.
    with pytest.raises(ValueError, match="unique key besides its primary key"):
        insert_ignore("sqlite", users)


def _keyed_table(*unique_index_columns: str) -> Table:
    table = Table("keyed", MetaData(), Column("name", String(10), primary_key=True), Column("value", Integer))
    if unique_index_columns:
        Index("keyed_unique", *(table.c[name] for name in unique_index_columns), unique=True)
    return table


def test_a_unique_index_of_the_primary_key_columns_is_that_same_key() -> None:
    # Some migrated SQLite tables index their primary key once more, uniquely (images.image_name).
    table = _keyed_table("name")

    sql = str(insert_ignore("sqlite", table).compile(dialect=sqlite.dialect()))
    upsert("sqlite", table, update=["value"])

    assert "ON CONFLICT (name) DO NOTHING" in sql


@pytest.mark.parametrize("columns", [("value",), ("name", "value")])
def test_a_unique_index_of_other_columns_is_refused(columns: tuple[str, ...]) -> None:
    # A key on more columns than the primary key's is another key too: MySQL resolves a conflict on either.
    with pytest.raises(ValueError, match="unique key besides its primary key"):
        insert_ignore("sqlite", _keyed_table(*columns))


def test_an_upsert_refuses_a_table_without_a_primary_key() -> None:
    table = Table("no_key", MetaData(), Column("value", Integer))

    with pytest.raises(ValueError, match="no primary key"):
        upsert("sqlite", table, update=["value"])


@pytest.mark.parametrize("dialect", [sqlite.dialect(), mysql.dialect()])
def test_a_case_insensitive_like_is_a_condition_as_it_stands(dialect: object) -> None:
    # Compared with 1, a LIKE keeps SQLite from searching an index with its collation by range.
    statement = select(client_state.c.key).where(CaseInsensitiveLike(client_state.c.key, "pattern", 2))

    sql = str(statement.compile(dialect=dialect))  # type: ignore[arg-type]

    assert "LIKE" in sql and not sql.rstrip().endswith(("= 1", "= true"))


def test_a_case_insensitive_like_replaces_on_sqlite_and_lowers_on_a_server() -> None:
    statement = select(client_state.c.key).where(CaseInsensitiveLike(client_state.c.key, "pattern", 2))

    on_sqlite = statement.compile(dialect=sqlite.dialect())
    on_server = statement.compile(dialect=mysql.dialect())

    assert 'replace(replace(client_state."key", ?, ?), ?, ?) LIKE ?' in str(on_sqlite)
    assert set(on_sqlite.params) == {"pattern", "pattern_from_0", "pattern_to_0", "pattern_from_1", "pattern_to_1"}
    assert "lower(client_state.`key`) LIKE lower(%s)" in str(on_server)
    assert set(on_server.params) == {"pattern_unfolded"}


def test_a_case_insensitive_like_without_slots_is_the_plain_like() -> None:
    statement = select(client_state.c.key).where(CaseInsensitiveLike(client_state.c.key, "pattern", 0))

    assert "replace" not in str(statement.compile(dialect=sqlite.dialect()))


def test_fold_case_lowers_each_character_on_its_own() -> None:
    assert fold_case("ÅLAND Åland") == "åland åland"
    # A character whose lowercase is longer stays as it is.
    assert fold_case("İ") == "İ"
    assert fold_case("\u212a") == "k"  # KELVIN SIGN
    assert fold_case("ΑΣ") == "ασ"  # no final sigma
    assert fold_case("ß") == "ß"
    assert fold_case("Hello, World_1%") == "hello, world_1%"


def test_a_pattern_replaces_the_characters_sqlite_does_not_fold() -> None:
    pattern = like_contains("Åland")

    assert pattern.pattern == "%åland%"
    # The letters of ASCII are folded by SQLite's LIKE itself, so `A` and `L` need none.
    # The ANGSTROM SIGN is no NFC text, so it is no pair.
    assert pattern.pairs == (("\u00c5", "å"),)
    assert pattern.slots == 1
    assert pattern.parameters("p", 4) == {
        "p": "%åland%",
        "p_unfolded": "%Åland%",
        "p_from_0": "\u00c5",
        "p_to_0": "å",
        "p_from_1": "",
        "p_to_1": "",
        "p_from_2": "",
        "p_to_2": "",
        "p_from_3": "",
        "p_to_3": "",
    }


def test_a_pattern_of_ascii_letters_has_no_replacements() -> None:
    assert like_contains("abc").pairs == ()
    assert like_contains("abc").slots == 0
    assert like_prefix("Abc_1").pattern == "abc\\_1%"
    assert like_contains("kit").pairs == ()
    # What lets an ASCII text skip the search for pairs: only ASCII capitals fold to an ASCII character.
    assert all(variant.isascii() for code in range(128) for variant in case_variants(chr(code)))


def test_the_characters_that_fold_to_a_letter_are_the_nfc_ones_of_one_character() -> None:
    # Not the KELVIN SIGN (canonically `K`), nor `İ`, whose lowercase has two characters.
    assert case_variants("k") == ("K",)
    assert case_variants("i") == ("I",)
    assert case_variants("å") == ("\u00c5",)
    assert case_variants("ω") == ("\u03a9",)


def test_no_character_beyond_the_scanned_planes_has_a_lowercase() -> None:
    # `case_variants` scans code points below 0x20000: a new Unicode version with cased letters above fails here.
    assert [hex(x) for x in range(0x20000, sys.maxunicode + 1) if chr(x).lower() != chr(x)] == []


@pytest.mark.parametrize(("pairs", "slots"), [(0, 0), (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8)])
def test_slots_round_the_pairs_up_to_a_few_statement_shapes(pairs: int, slots: int) -> None:
    pattern = LikePattern("%", "%", tuple((str(i), "x") for i in range(pairs)))

    assert pattern.slots == slots


_STORED = [
    "Åland",
    "xxåLAND",
    "Aland",
    "aland",
    "50%",
    "a_b",
    "a\\b",
    "ab",
    "İstanbul",
    "STRAẞE",
    "\u212bngstrom",  # ANGSTROM SIGN
]


def _names_like(database: Database, pattern: LikePattern) -> set[str]:
    slots = pattern.slots
    statement = select(images.c.image_name).where(CaseInsensitiveLike(images.c.image_name, "pattern", slots))
    with database.begin(write=False) as conn:
        return set(conn.execute(statement, pattern.parameters("pattern", slots)).scalars())


@pytest.fixture
def stored_names(database: Database) -> Database:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(images),
            [
                {"image_name": name, "image_origin": "internal", "image_category": "general", "width": 1, "height": 1}
                for name in _STORED
            ],
        )
    return database


@pytest.mark.parametrize("term", ["åland", "ÅLAND", "Åland"])
def test_a_contains_pattern_finds_the_text_whatever_the_case_of_either_side(stored_names: Database, term: str) -> None:
    assert _names_like(stored_names, like_contains(term)) == {"Åland", "xxåLAND"}


def test_a_prefix_pattern_finds_only_the_values_that_start_with_the_text(stored_names: Database) -> None:
    assert _names_like(stored_names, like_prefix("ÅLAND")) == {"Åland"}
    assert _names_like(stored_names, like_prefix("aland")) == {"Aland", "aland"}


def test_a_value_whose_lowercase_is_longer_finds_itself(stored_names: Database) -> None:
    # `fold_case` keeps `İ`; a server's LOWER() may fold it, on the stored value and the pattern alike.
    assert _names_like(stored_names, like_contains("İstanbul")) == {"İstanbul"}


def test_a_value_a_server_does_not_lower_finds_itself(stored_names: Database) -> None:
    # `fold_case` folds `ẞ` to `ß`; MariaDB's LOWER() keeps it (MySQL 8.4's lowers it to `ß`).
    assert _names_like(stored_names, like_contains("STRAẞE")) == {"STRAẞE"}


@pytest.mark.parametrize(("term", "found"), [("%", "50%"), ("_", "a_b"), ("\\", "a\\b")])
def test_the_characters_of_a_like_pattern_match_only_themselves(stored_names: Database, term: str, found: str) -> None:
    assert _names_like(stored_names, like_contains(term)) == {found}


def test_a_text_typed_with_a_character_nfc_text_never_holds_finds_it(stored_names: Database) -> None:
    # The ANGSTROM SIGN folds to `å` only canonically, so a plain `Å` does not find it on SQLite; typed, it does.
    assert _names_like(stored_names, like_contains("\u212bngstrom")) == {"\u212bngstrom"}


def test_the_typed_characters_nfc_text_never_holds_are_pairs_of_their_own() -> None:
    assert like_contains("\u212bngstrom").pairs == (("\u00c5", "å"), ("\u212b", "å"))
    assert like_contains("\u212ait").pairs == (("\u212a", "k"),)
    assert like_contains("Ångstrom").pairs == (("\u00c5", "å"),)
    assert like_contains("kit").pairs == ()


def test_a_pattern_has_at_most_the_pairs_a_scan_of_rows_can_afford() -> None:
    letters = "абвгдежзийклмнопрстуфхцчшщъыьэюяё"
    pattern = like_contains(letters.upper())

    assert len(pattern.pairs) == pattern.slots == MAX_PAIRS
    # The first letters have their pair, the others match any one character on SQLite.
    assert pattern.pairs[0] == ("А", "а")
    assert pattern.pattern == "%" + letters[:MAX_PAIRS] + "_" * (len(letters) - MAX_PAIRS) + "%"
    assert pattern.unfolded == f"%{letters.upper()}%"


def test_a_text_of_many_foldable_letters_finds_its_value_on_every_backend(database: Database) -> None:
    letters = "абвгдежзийклмнопрстуфхцчшщъыьэюяё"
    differing = letters.upper()[:-1] + "Ж"  # differs in a letter SQLite has no pair for
    names = [f"x-{letters.upper()}-x", letters, differing, letters[:-1], "other"]
    with database.begin(write=True) as conn:
        conn.execute(
            insert(images),
            [
                {"image_name": name, "image_origin": "internal", "image_category": "general", "width": 1, "height": 1}
                for name in names
            ],
        )

    found = _names_like(database, like_contains(letters.upper()))

    assert {names[0], names[1]} <= found
    assert names[3] not in found and names[4] not in found
    # SQLite over-matches on the letters beyond the pairs; a server compares them all.
    assert (differing in found) == (database.dialect_name == "sqlite")


@pytest.mark.parametrize("dialect", [sqlite.dialect(), mysql.dialect()])
def test_a_fixed_limit_is_written_into_the_statement(dialect: object) -> None:
    # Bound, it would cost SQLite 10-20 µs per execution.
    compiled = select(users.c.email).limit(fixed_limit(1)).compile(dialect=dialect)  # type: ignore[arg-type]

    assert "LIMIT 1" in str(compiled)
    assert 1 not in compiled.params.values()


def test_an_upsert_compiles_for_postgresql() -> None:
    statement = upsert("postgresql", client_state, update=["value", "updated_at"])

    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "ON CONFLICT (user_id, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at" in sql


def test_an_insert_ignore_compiles_for_postgresql() -> None:
    statement = insert_ignore("postgresql", client_state)

    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "ON CONFLICT (user_id, key) DO NOTHING" in sql


def test_a_bound_set_matches_exactly_the_values_it_holds(database: Database) -> None:
    quoted = 'it\'s "quoted" \\ here.png'
    stored = ["a.png", "A.png", quoted, "ünïcode.png", "x" * 251 + ".png", "other.png"]
    with database.begin(write=True) as conn:
        conn.execute(
            insert(images),
            [
                {"image_name": name, "image_origin": "internal", "image_category": "general", "width": 1, "height": 1}
                for name in stored
            ],
        )
    statement = select(images.c.image_name).where(InBoundSet(images.c.image_name, bindparam("names")))

    def matching(values: list[str]) -> set[str]:
        with database.begin(write=False) as conn:
            return set(conn.execute(statement, {"names": bound_set(values)}).scalars())

    # Compared as stored: case, quotes, backslashes and letters beyond ASCII, up to the longest name.
    wanted = ["a.png", quoted, "ünïcode.png", "x" * 251 + ".png", "missing.png"]
    assert matching(wanted) == set(wanted) - {"missing.png"}
    assert matching([]) == set()


def test_a_bound_set_of_an_integer_column_matches_its_integers(database: Database) -> None:
    # Beyond 32 bits, so that reading the set as INT would miss it.
    large = 2**40
    with database.begin(write=True) as conn:
        for item_id in (large, large + 1, large + 2):
            conn.execute(
                insert(session_queue).values(
                    item_id=item_id, queue_id="q", batch_id="b", session_id=f"s{item_id}", session="{}"
                )
            )
    statement = select(session_queue.c.item_id).where(InBoundSet(session_queue.c.item_id, bindparam("ids")))

    with database.begin(write=False) as conn:
        found = set(conn.execute(statement, {"ids": bound_set([large, large + 2, 7])}).scalars())

    assert found == {large, large + 2}


def test_a_bound_set_of_an_integer_column_is_read_as_integers_on_every_server() -> None:
    statement = select(session_queue.c.item_id).where(InBoundSet(session_queue.c.item_id, bindparam("ids")))

    assert "value BIGINT PATH '$'" in str(statement.compile(dialect=mysql.dialect()))
    assert "CAST(value AS BIGINT)" in str(statement.compile(dialect=postgresql.dialect()))
