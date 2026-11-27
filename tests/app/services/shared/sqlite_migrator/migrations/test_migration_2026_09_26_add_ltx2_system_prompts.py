import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_10_create_system_prompts import (
    DEFAULT_SYSTEM_PROMPTS,
    CreateSystemPromptsCallback,
)
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_07_add_minimax_h3_ref2va_system_prompt import (
    MINIMAX_H3_REF2VA_PROMPT_ID,
    AddMiniMaxH3Ref2VASystemPromptCallback,
)
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_09_add_system_prompt_max_tokens import (
    AddSystemPromptMaxTokensCallback,
)
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_26_add_ltx2_system_prompts import (
    LTX2_5_I2V_PROMPT_ID,
    LTX2_5_MAX_TOKENS,
    LTX2_5_SYSTEM_PROMPTS,
    LTX2_5_T2V_PROMPT_ID,
    AddLtx2SystemPromptsCallback,
)


def _db_before_migration() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    cursor = db.cursor()
    CreateSystemPromptsCallback()(cursor)
    AddMiniMaxH3Ref2VASystemPromptCallback()(cursor)
    AddSystemPromptMaxTokensCallback()(cursor)
    return db


def test_seeds_both_prompts_shared_with_their_token_cap() -> None:
    db = _db_before_migration()
    cursor = db.cursor()

    AddLtx2SystemPromptsCallback()(cursor)

    cursor.execute(
        "SELECT id, name, content, user_id, is_public, max_tokens FROM system_prompts WHERE id IN (?, ?) ORDER BY id;",
        (LTX2_5_T2V_PROMPT_ID, LTX2_5_I2V_PROMPT_ID),
    )
    assert cursor.fetchall() == [
        (prompt_id, name, content, "system", 1, LTX2_5_MAX_TOKENS) for prompt_id, name, content in LTX2_5_SYSTEM_PROMPTS
    ]
    db.close()


def test_leaves_a_row_that_already_holds_the_id_alone() -> None:
    db = _db_before_migration()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO system_prompts (id, name, content, user_id, is_public) VALUES (?, 'Mine', 'edited', 'system', 0);",
        (LTX2_5_T2V_PROMPT_ID,),
    )

    AddLtx2SystemPromptsCallback()(cursor)

    cursor.execute("SELECT name, content, max_tokens FROM system_prompts WHERE id = ?;", (LTX2_5_T2V_PROMPT_ID,))
    assert cursor.fetchone() == ("Mine", "edited", None)
    db.close()


def test_ids_do_not_collide_with_earlier_seeds() -> None:
    earlier = {default_id for default_id, _, _ in DEFAULT_SYSTEM_PROMPTS} | {MINIMAX_H3_REF2VA_PROMPT_ID}
    assert earlier.isdisjoint({LTX2_5_T2V_PROMPT_ID, LTX2_5_I2V_PROMPT_ID})
