"""Persisted cleanup operations are removed; browser holds stay."""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_09_24_drop_intermediates_operations import (
    build_migration,
)


def _tables(cursor: sqlite3.Cursor) -> set[str]:
    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
    return {row[0] for row in cursor.fetchall()}


def test_drops_operation_tables_and_keeps_browser_holds() -> None:
    cursor = sqlite3.connect(":memory:").cursor()
    cursor.execute("CREATE TABLE intermediates_operations (operation_id TEXT PRIMARY KEY, state_json TEXT);")
    cursor.execute("CREATE TABLE intermediates_operation_targets (operation_id TEXT, media_name TEXT);")
    cursor.execute("CREATE TABLE intermediates_browser_holds (user_id TEXT, lease_id TEXT, media_name TEXT);")
    cursor.execute("INSERT INTO intermediates_browser_holds VALUES ('alice', 'tab', 'held.png');")

    build_migration().callback(cursor)

    assert _tables(cursor) == {"intermediates_browser_holds"}
    cursor.execute("SELECT COUNT(*) FROM intermediates_browser_holds;")
    assert cursor.fetchone()[0] == 1


def test_runs_on_databases_with_or_without_the_tables() -> None:
    cursor = sqlite3.connect(":memory:").cursor()
    build_migration().callback(cursor)
    build_migration().callback(cursor)
    assert _tables(cursor) == set()
