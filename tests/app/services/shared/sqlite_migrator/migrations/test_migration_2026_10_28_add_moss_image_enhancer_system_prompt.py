"""The MOSS-Image prompt enhancer is seeded once; a row the user changed keeps their changes."""

from pathlib import Path

from sqlalchemy import delete, select, update

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.database.schema.system_prompts import system_prompts
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_10_28_add_moss_image_enhancer_system_prompt import (
    MOSS_IMAGE_ENHANCER_PROMPT_CONTENT,
    MOSS_IMAGE_ENHANCER_PROMPT_ID,
)
from tests.fixtures.database import migrate_to_newest

MIGRATION_ID = "2026_10_28_add_moss_image_enhancer_system_prompt"


def _row(database: Database) -> tuple | None:
    with database.begin(write=False) as conn:
        row = conn.execute(
            select(
                system_prompts.c.name,
                system_prompts.c.content,
                system_prompts.c.user_id,
                system_prompts.c.is_public,
                system_prompts.c.max_tokens,
            ).where(system_prompts.c.id == MOSS_IMAGE_ENHANCER_PROMPT_ID)
        ).one_or_none()
    return None if row is None else tuple(row)


def _forget_migration(database: Database) -> None:
    with database.begin(write=True) as conn:
        conn.execute(delete(applied_migrations).where(applied_migrations.c.migration_id == MIGRATION_ID))


def test_seeds_a_shared_prompt_with_its_token_cap(database: Database) -> None:
    assert _row(database) == (
        "MOSS-Image Prompt Enhancer",
        MOSS_IMAGE_ENHANCER_PROMPT_CONTENT,
        "system",
        True,
        1536,
    )
    # The prompt asks for the draft convention the layout prompt node compiles.
    assert '"bbox_coord_space":"abs","bbox_format":"xyxy"' in MOSS_IMAGE_ENHANCER_PROMPT_CONTENT


def test_running_again_leaves_an_edited_prompt_alone(database: Database, tmp_path: Path) -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            update(system_prompts)
            .where(system_prompts.c.id == MOSS_IMAGE_ENHANCER_PROMPT_ID)
            .values(name="Mine", content="edited", is_public=False, max_tokens=None)
        )
    _forget_migration(database)

    migrate_to_newest(database, tmp_path)

    assert _row(database) == ("Mine", "edited", "system", False, None)


def test_running_again_does_not_restore_a_deleted_prompt(database: Database, tmp_path: Path) -> None:
    """Deleting a seeded prompt is the user's choice; only a database that never ran the migration gets it."""
    with database.begin(write=True) as conn:
        conn.execute(delete(system_prompts).where(system_prompts.c.id == MOSS_IMAGE_ENHANCER_PROMPT_ID))

    migrate_to_newest(database, tmp_path)

    assert _row(database) is None
