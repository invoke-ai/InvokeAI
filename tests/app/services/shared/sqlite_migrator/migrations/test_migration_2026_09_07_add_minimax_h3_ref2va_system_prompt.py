import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_07_10_create_system_prompts import (
    DEFAULT_SYSTEM_PROMPTS,
    CreateSystemPromptsCallback,
)
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_07_add_minimax_h3_ref2va_system_prompt import (
    MINIMAX_H3_REF2VA_PROMPT_CONTENT,
    MINIMAX_H3_REF2VA_PROMPT_ID,
    MINIMAX_H3_REF2VA_PROMPT_NAME,
    AddMiniMaxH3Ref2VASystemPromptCallback,
    build_migration,
)


def _seeded_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    CreateSystemPromptsCallback()(db.cursor())
    return db


def test_seeds_the_prompt_shared_and_owned_by_system() -> None:
    db = _seeded_db()
    cursor = db.cursor()

    AddMiniMaxH3Ref2VASystemPromptCallback()(cursor)

    cursor.execute(
        "SELECT name, content, user_id, is_public FROM system_prompts WHERE id = ?;",
        (MINIMAX_H3_REF2VA_PROMPT_ID,),
    )
    assert cursor.fetchone() == (MINIMAX_H3_REF2VA_PROMPT_NAME, MINIMAX_H3_REF2VA_PROMPT_CONTENT, "system", 1)
    cursor.execute("SELECT COUNT(*) FROM system_prompts;")
    assert cursor.fetchone()[0] == len(DEFAULT_SYSTEM_PROMPTS) + 1

    db.close()


def test_id_does_not_collide_with_the_original_seed() -> None:
    assert MINIMAX_H3_REF2VA_PROMPT_ID not in {default_id for default_id, _, _ in DEFAULT_SYSTEM_PROMPTS}


def test_is_idempotent_and_preserves_user_edits() -> None:
    # Migrations run once per id, but a re-run must not duplicate the row or undo an edit.
    db = _seeded_db()
    cursor = db.cursor()

    AddMiniMaxH3Ref2VASystemPromptCallback()(cursor)
    cursor.execute("UPDATE system_prompts SET content = 'edited by user' WHERE id = ?;", (MINIMAX_H3_REF2VA_PROMPT_ID,))

    AddMiniMaxH3Ref2VASystemPromptCallback()(cursor)

    cursor.execute("SELECT content FROM system_prompts WHERE id = ?;", (MINIMAX_H3_REF2VA_PROMPT_ID,))
    assert cursor.fetchone()[0] == "edited by user"
    cursor.execute("SELECT COUNT(*) FROM system_prompts;")
    assert cursor.fetchone()[0] == len(DEFAULT_SYSTEM_PROMPTS) + 1

    db.close()


def test_prompt_teaches_every_required_section() -> None:
    # The whole point of the prompt is the section contract; a dropped heading silently degrades
    # the structure the model emits.
    for heading in (
        "### subject_definitions",
        "### summary",
        "### retention_analysis",
        "### detailed_description",
        "### overall_soundscape",
        "### non_diegetic_music",
    ):
        assert heading in MINIMAX_H3_REF2VA_PROMPT_CONTENT
    assert "[TO FILL]" in MINIMAX_H3_REF2VA_PROMPT_CONTENT


def test_the_shot_list_is_length_bounded() -> None:
    # detailed_description is the section that grows without limit, and it sits ahead of the two
    # audio sections -- unbounded, it crowds them out of the token budget entirely.
    assert "two to four sentences" in MINIMAX_H3_REF2VA_PROMPT_CONTENT


def test_migration_id_matches_its_module_name() -> None:
    assert build_migration().id == "2026_09_07_add_minimax_h3_ref2va_system_prompt"
    assert build_migration().depends_on == "2026_07_10_create_system_prompts"
