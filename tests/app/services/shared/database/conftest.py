from collections.abc import Iterator
from pathlib import Path

import pytest

from invokeai.app.services.shared.database.database import Database
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.database_probe import ProbeQueries, create_probe_tables


@pytest.fixture
def probe(empty_database: Database) -> ProbeQueries:
    """Queries over two probe tables, created in the empty database under test."""
    create_probe_tables(empty_database)
    return ProbeQueries(empty_database)


@pytest.fixture
def sqlite_file_database(tmp_path: Path) -> Iterator[Database]:
    """A SQLite database file with the probe tables, for tests that open it from a second connection too."""
    database = Database.open_sqlite(tmp_path / "probe.db", InvokeAILogger.get_logger("test_database"))
    create_probe_tables(database)
    try:
        yield database
    finally:
        database.dispose()
