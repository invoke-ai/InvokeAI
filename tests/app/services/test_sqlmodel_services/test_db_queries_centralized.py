"""Guard: every SQL query in the application must live in the central DbQueries wrapper.

The contract (see ``invokeai/app/services/shared/sqlite/db_queries.py``): services contain
business logic only and delegate all data access to ``db.queries``. This test walks the
services tree and fails if any file outside the DB infrastructure opens sessions, starts
raw transactions, or imports sqlalchemy/sqlmodel/sqlite3.

If this test fails for a file you just changed: move the query into ``DbQueries`` and call
it via ``db.queries`` instead of querying inline.
"""

import re
from pathlib import Path

from invokeai.app.services.shared.sqlite.db_queries import DbQueries
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

SERVICES_DIR = Path(__file__).parents[4] / "invokeai" / "app" / "services"

# Patterns that indicate inline data access.
FORBIDDEN_PATTERNS = (
    re.compile(r"\.get_session\("),
    re.compile(r"\.get_readonly_session\("),
    re.compile(r"\.transaction\("),
    re.compile(r"^\s*(?:from|import)\s+sqlalchemy", re.MULTILINE),
    re.compile(r"^\s*(?:from|import)\s+sqlmodel", re.MULTILINE),
    re.compile(r"^\s*import\s+sqlite3", re.MULTILINE),
)

# The DB layer itself: schema, engine, wrapper, migrations, copy tool.
DB_INFRA_PREFIXES = (
    "shared/sqlite/",
    "shared/sqlite_migrator/",
)

# Frozen legacy raw-SQL implementations. They are NOT wired up (dependencies.py uses the
# SQLModel services) and are kept only for reference. Do not add new entries here —
# new queries belong in DbQueries.
LEGACY_RAW_SQL_FILES = {
    "app_settings/app_settings_service.py",
    "board_image_records/board_image_records_sqlite.py",
    "board_records/board_records_sqlite.py",
    "client_state_persistence/client_state_persistence_sqlite.py",
    "image_records/image_records_sqlite.py",
    "model_records/model_records_sql.py",
    "model_relationship_records/model_relationship_records_sqlite.py",
    "session_queue/session_queue_sqlite.py",
    "style_preset_records/style_preset_records_sqlite.py",
    "users/users_default.py",
    "workflow_records/workflow_records_sqlite.py",
}


def test_no_queries_outside_db_wrapper() -> None:
    """No service file outside the DB layer may touch sessions or SQL libraries."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(SERVICES_DIR.rglob("*.py")):
        rel = path.relative_to(SERVICES_DIR).as_posix()
        if rel.startswith(DB_INFRA_PREFIXES) or rel in LEGACY_RAW_SQL_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [pattern.pattern for pattern in FORBIDDEN_PATTERNS if pattern.search(text)]
        if hits:
            offenders[rel] = hits

    assert not offenders, (
        "Inline data access found outside the central DB wrapper. Move these queries into "
        f"DbQueries (db.queries) in shared/sqlite/db_queries.py: {offenders}"
    )


def test_legacy_allowlist_is_not_stale() -> None:
    """Every allowlisted legacy file must still exist — remove deleted files from the list."""
    missing = [rel for rel in sorted(LEGACY_RAW_SQL_FILES) if not (SERVICES_DIR / rel).exists()]
    assert not missing, f"Stale entries in LEGACY_RAW_SQL_FILES (files deleted?): {missing}"


def test_db_queries_property(db: SqliteDatabase) -> None:
    """db.queries exposes the wrapper and is cached per database instance."""
    queries = db.queries
    assert isinstance(queries, DbQueries)
    assert db.queries is queries
