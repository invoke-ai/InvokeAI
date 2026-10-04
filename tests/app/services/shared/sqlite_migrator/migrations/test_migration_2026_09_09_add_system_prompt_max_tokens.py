import sqlite3

import pytest

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_10_create_system_prompts import (
    DEFAULT_SYSTEM_PROMPTS,
    CreateSystemPromptsCallback,
)
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_07_add_minimax_h3_ref2va_system_prompt import (
    MINIMAX_H3_REF2VA_PROMPT_ID,
    AddMiniMaxH3Ref2VASystemPromptCallback,
)
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_09_add_system_prompt_max_tokens import (
    MINIMAX_H3_REF2VA_MAX_TOKENS,
    AddSystemPromptMaxTokensCallback,
    build_migration,
)
from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    EXPAND_PROMPT_MAX_TOKENS_MAX,
    EXPAND_PROMPT_MAX_TOKENS_MIN,
)


@pytest.fixture
def db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    CreateSystemPromptsCallback()(cursor)
    AddMiniMaxH3Ref2VASystemPromptCallback()(cursor)
    yield connection
    connection.close()


def _max_tokens(cursor: sqlite3.Cursor, prompt_id: str) -> int | None:
    cursor.execute("SELECT max_tokens FROM system_prompts WHERE id = ?;", (prompt_id,))
    return cursor.fetchone()[0]


def test_adds_a_nullable_column_that_leaves_existing_prompts_on_the_default(db: sqlite3.Connection) -> None:
    cursor = db.cursor()

    AddSystemPromptMaxTokensCallback()(cursor)

    # Every prompt but Ref2VA keeps NULL, which the API reads as "use the endpoint default".
    for default_id, _, _ in DEFAULT_SYSTEM_PROMPTS:
        assert _max_tokens(cursor, default_id) is None


def test_raises_the_ref2va_cap(db: sqlite3.Connection) -> None:
    cursor = db.cursor()

    AddSystemPromptMaxTokensCallback()(cursor)

    assert _max_tokens(cursor, MINIMAX_H3_REF2VA_PROMPT_ID) == MINIMAX_H3_REF2VA_MAX_TOKENS


def test_seeded_cap_is_within_the_range_the_api_accepts() -> None:
    assert EXPAND_PROMPT_MAX_TOKENS_MIN <= MINIMAX_H3_REF2VA_MAX_TOKENS <= EXPAND_PROMPT_MAX_TOKENS_MAX


def test_is_idempotent_and_preserves_a_user_chosen_cap(db: sqlite3.Connection) -> None:
    cursor = db.cursor()

    AddSystemPromptMaxTokensCallback()(cursor)
    cursor.execute("UPDATE system_prompts SET max_tokens = 800 WHERE id = ?;", (MINIMAX_H3_REF2VA_PROMPT_ID,))

    # A re-run must neither fail on the already-present column nor undo the user's choice.
    AddSystemPromptMaxTokensCallback()(cursor)

    assert _max_tokens(cursor, MINIMAX_H3_REF2VA_PROMPT_ID) == 800


def test_tolerates_a_deleted_ref2va_prompt(db: sqlite3.Connection) -> None:
    cursor = db.cursor()
    cursor.execute("DELETE FROM system_prompts WHERE id = ?;", (MINIMAX_H3_REF2VA_PROMPT_ID,))

    AddSystemPromptMaxTokensCallback()(cursor)

    cursor.execute("SELECT COUNT(*) FROM system_prompts WHERE id = ?;", (MINIMAX_H3_REF2VA_PROMPT_ID,))
    assert cursor.fetchone()[0] == 0


def test_migration_id_matches_its_module_name() -> None:
    migration = build_migration()
    assert migration.id == "2026_09_09_add_system_prompt_max_tokens"
    # The Ref2VA row must exist before its cap can be raised.
    assert migration.depends_on == "2026_09_07_add_minimax_h3_ref2va_system_prompt"
