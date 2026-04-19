"""Shared fixtures for SQLModel service tests."""

from logging import Logger

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def logger() -> Logger:
    return InvokeAILogger.get_logger()


@pytest.fixture
def db(logger: Logger) -> SqliteDatabase:
    """Create an in-memory database with all migrations applied."""
    config = InvokeAIAppConfig(use_memory_db=True)
    return create_mock_sqlite_database(config=config, logger=logger)
