"""The MOSS-Image prompt enhancer is seeded once; a row the user changed keeps their changes."""

import re
from logging import Logger
from pathlib import Path

from sqlalchemy import MetaData, create_engine, delete, select, update

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.database.schema.system_prompts import system_prompts
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_10_28_add_moss_image_enhancer_system_prompt import (
    MOSS_IMAGE_ENHANCER_PROMPT_CONTENT,
    MOSS_IMAGE_ENHANCER_PROMPT_ID,
    build_migration,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import PortableMigrationContext
from tests.fixtures.database import migrate_to_newest
from tests.fixtures.migration_state import alembic_operations

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
    with database.begin(write=False) as conn:
        created_at, updated_at = conn.execute(
            select(system_prompts.c.created_at, system_prompts.c.updated_at).where(
                system_prompts.c.id == MOSS_IMAGE_ENHANCER_PROMPT_ID
            )
        ).one()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", created_at)
    assert updated_at == created_at


def test_supplies_every_value_a_server_has_no_default_for() -> None:
    """A server declares no defaults for the timestamps, which only SQLite has (`inserted_at`, `updated_at`).

    The table here is the schema's with the SQLite-only defaults left out, so it holds the server's NOT NULL
    columns without a default; the insert must name each of them.
    """
    server_shaped = system_prompts.to_metadata(MetaData())
    for column in server_shaped.columns:
        column.info.clear()
    assert {
        column.name for column in server_shaped.columns if not column.nullable and column.server_default is None
    } >= {
        "created_at",
        "updated_at",
    }
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        server_shaped.create(conn)
        context = PortableMigrationContext(conn=conn, op=alembic_operations(conn), logger=Logger("test"))
        build_migration().callback(context)
        row = conn.execute(select(server_shaped.c.id, server_shaped.c.created_at, server_shaped.c.updated_at)).one()
    engine.dispose()
    assert row.id == MOSS_IMAGE_ENHANCER_PROMPT_ID
    assert row.created_at is not None and row.updated_at == row.created_at


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
