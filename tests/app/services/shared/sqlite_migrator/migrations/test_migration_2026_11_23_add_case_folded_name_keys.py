"""Workflows and fonts gain the case-folded keys they sort by, filled from the names they hold."""

import json
from logging import Logger
from pathlib import Path
from unittest.mock import MagicMock

from sqlalchemy import delete, inspect

from invokeai.app.services.config.config_default import DefaultInvokeAIAppConfig
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.migrator import applied_migrations
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import MigrationBase
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import Migrator
from tests.fixtures.database import migrate_to_newest

MIGRATION_ID = "2026_11_23_add_case_folded_name_keys"
WORKFLOW_NAMES = ["ÅLAND Trip", "zebra", "Ångström"]
FONTS = [("f1", "ÅLAND Sans", "Ångström Bold"), ("f2", "zebra", "Regular")]


def _run(db: Database, migrations: list[MigrationBase]) -> bool:
    migrator = Migrator(db)
    for migration in migrations:
        migrator.register_migration(migration)
    return migrator.run_migrations()


def _columns(db: Database, table: str) -> set[str]:
    with db.begin(write=False) as conn:
        return {column["name"] for column in inspect(conn).get_columns(table)}


def _keys(db: Database) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    with db.begin(write=False) as conn:
        workflows = conn.exec_driver_sql("SELECT name, name_key FROM workflow_library ORDER BY workflow_id").all()
        fonts = conn.exec_driver_sql("SELECT id, family_key, label_key FROM fonts ORDER BY id").all()
    return [tuple(row) for row in workflows], [tuple(row) for row in fonts]


def test_existing_names_get_their_folded_keys(tmp_path: Path) -> None:
    logger = Logger("test")
    config = DefaultInvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    config._root = tmp_path
    migrations = build_migrations(MigrationBuildContext(app_config=config, logger=logger, image_files=MagicMock()))
    db = Database.open_sqlite(None, logger)
    assert _run(db, [migration for migration in migrations if migration.id != MIGRATION_ID])
    assert "name_key" not in _columns(db, "workflow_library")
    assert not {"family_key", "label_key"} & _columns(db, "fonts")
    with db.begin(write=True) as conn:
        for index, name in enumerate(WORKFLOW_NAMES):
            document = {"name": name, "description": "", "meta": {"category": "user", "version": "3.0.0"}}
            conn.exec_driver_sql(
                "INSERT INTO workflow_library (workflow_id, workflow) VALUES (?, ?)",
                (f"w{index}", json.dumps(document)),
            )
        for font_id, family, label in FONTS:
            conn.exec_driver_sql(
                "INSERT INTO fonts (id, scope, source, filename, source_path, family, label, style, weight, "
                "content_hash, byte_size) VALUES (?, 'shared', 'directory', 'f.ttf', ?, ?, ?, 'normal', 400, ?, 1)",
                (font_id, f"{font_id}.ttf", family, label, font_id[-1] * 64),
            )

    assert _run(db, migrations)

    workflows, fonts = _keys(db)
    assert workflows == [("ÅLAND Trip", "åland trip"), ("zebra", "zebra"), ("Ångström", "ångström")]
    assert fonts == [("f1", "åland sans", "ångström bold"), ("f2", "zebra", "regular")]
    assert {"family_key", "label_key"} <= _columns(db, "fonts")
    with db.begin(write=False) as conn:
        indexed = [
            column["name"]
            for column in inspect(conn).get_indexes("fonts")
            if column["name"] == "idx_fonts_visible_private"
        ]
        columns = next(
            index["column_names"]
            for index in inspect(conn).get_indexes("fonts")
            if index["name"] == "idx_fonts_visible_private"
        )
    assert indexed and columns == ["source", "scope", "owner_id", "family_key"]
    db.dispose()


def test_running_again_changes_nothing(database: Database, tmp_path: Path) -> None:
    """On MySQL and MariaDB a migration that failed after its DDL runs again on columns it already added."""
    with database.begin(write=True) as conn:
        conn.exec_driver_sql(
            "INSERT INTO fonts (id, scope, source, filename, source_path, family, label, style, weight, content_hash, "
            "byte_size, created_at, updated_at) VALUES ('f1', 'shared', 'directory', 'f.ttf', 'f.ttf', 'ÅLAND', "
            f"'Bold', 'normal', 400, '{'a' * 64}', 1, '2026-11-23 00:00:00.000', '2026-11-23 00:00:00.000')"
        )
        conn.execute(delete(applied_migrations).where(applied_migrations.c.migration_id == MIGRATION_ID))
        conn.exec_driver_sql("UPDATE fonts SET family_key = NULL")

    migrate_to_newest(database, tmp_path)
    first = _keys(database)
    migrate_to_newest(database, tmp_path)

    assert first[1] == [("f1", "åland", "bold")]
    assert _keys(database) == first
