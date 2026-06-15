"""Tests for SqliteBoardCanvasProjectRecordStorage.

Covers the one-to-many board↔project association: add/remove, idempotent
re-association (moving from one board to another), counts, and FK cascades.
"""

import pytest

from invokeai.app.services.board_canvas_project_records.board_canvas_project_records_sqlite import (
    SqliteBoardCanvasProjectRecordStorage,
)
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.canvas_project_records.canvas_project_records_sqlite import (
    SqliteCanvasProjectRecordStorage,
)
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_records.image_records_common import ResourceOrigin
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def db() -> SqliteDatabase:
    config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger(config=config)
    return create_mock_sqlite_database(config, logger)


@pytest.fixture
def projects(db: SqliteDatabase) -> SqliteCanvasProjectRecordStorage:
    return SqliteCanvasProjectRecordStorage(db=db)


@pytest.fixture
def boards(db: SqliteDatabase) -> SqliteBoardRecordStorage:
    return SqliteBoardRecordStorage(db=db)


@pytest.fixture
def store(db: SqliteDatabase) -> SqliteBoardCanvasProjectRecordStorage:
    return SqliteBoardCanvasProjectRecordStorage(db=db)


def _save_project(records: SqliteCanvasProjectRecordStorage, project_name: str) -> None:
    records.save(
        project_name=project_name,
        project_origin=ResourceOrigin.INTERNAL,
        name=project_name,
        app_version="test-1.0",
        width=512,
        height=512,
        image_count=0,
        has_thumbnail=False,
    )


@pytest.fixture
def seeded(
    boards: SqliteBoardRecordStorage,
    projects: SqliteCanvasProjectRecordStorage,
) -> tuple[str, str]:
    """Two boards and two projects, ready to be associated by individual tests."""
    board_a = boards.save(board_name="Board A", user_id="system").board_id
    board_b = boards.save(board_name="Board B", user_id="system").board_id
    _save_project(projects, "p1")
    _save_project(projects, "p2")
    return board_a, board_b


class TestAddProjectToBoard:
    def test_add_creates_association(
        self,
        store: SqliteBoardCanvasProjectRecordStorage,
        seeded: tuple[str, str],
    ) -> None:
        board_a, _ = seeded
        store.add_project_to_board(board_id=board_a, project_name="p1")
        assert store.get_board_for_project("p1") == board_a

    def test_re_add_moves_project_between_boards(
        self,
        store: SqliteBoardCanvasProjectRecordStorage,
        seeded: tuple[str, str],
    ) -> None:
        # The PK on `project_name` means a project can only belong to one board at a time —
        # `ON CONFLICT DO UPDATE SET board_id` enforces that. Re-adding to a second board
        # must move it rather than create a duplicate row.
        board_a, board_b = seeded
        store.add_project_to_board(board_id=board_a, project_name="p1")
        store.add_project_to_board(board_id=board_b, project_name="p1")
        assert store.get_board_for_project("p1") == board_b
        assert "p1" not in store.get_all_board_project_names_for_board(board_a)
        assert "p1" in store.get_all_board_project_names_for_board(board_b)


class TestRemoveProjectFromBoard:
    def test_remove_drops_association(
        self,
        store: SqliteBoardCanvasProjectRecordStorage,
        seeded: tuple[str, str],
    ) -> None:
        board_a, _ = seeded
        store.add_project_to_board(board_id=board_a, project_name="p1")
        store.remove_project_from_board(project_name="p1")
        assert store.get_board_for_project("p1") is None

    def test_remove_unassociated_is_noop(
        self,
        store: SqliteBoardCanvasProjectRecordStorage,
        seeded: tuple[str, str],
    ) -> None:
        # Removing a project that was never on a board mustn't raise — the row
        # simply doesn't exist.
        store.remove_project_from_board(project_name="p1")
        assert store.get_board_for_project("p1") is None


class TestQueries:
    def test_get_board_for_project_returns_none_when_unassociated(
        self,
        store: SqliteBoardCanvasProjectRecordStorage,
        seeded: tuple[str, str],
    ) -> None:
        assert store.get_board_for_project("p1") is None

    def test_get_all_board_project_names_for_board(
        self,
        store: SqliteBoardCanvasProjectRecordStorage,
        seeded: tuple[str, str],
    ) -> None:
        board_a, board_b = seeded
        store.add_project_to_board(board_id=board_a, project_name="p1")
        store.add_project_to_board(board_id=board_a, project_name="p2")
        assert set(store.get_all_board_project_names_for_board(board_a)) == {"p1", "p2"}
        assert store.get_all_board_project_names_for_board(board_b) == []

    def test_get_all_board_project_names_for_none_returns_uncategorized(
        self,
        store: SqliteBoardCanvasProjectRecordStorage,
        seeded: tuple[str, str],
    ) -> None:
        # `board_id="none"` is the sentinel for "uncategorized" — projects without
        # a board row at all.
        board_a, _ = seeded
        store.add_project_to_board(board_id=board_a, project_name="p1")
        # p1 is on board_a, p2 has no association.
        assert set(store.get_all_board_project_names_for_board("none")) == {"p2"}

    def test_get_project_count_for_board_ignores_intermediate(
        self,
        db: SqliteDatabase,
        store: SqliteBoardCanvasProjectRecordStorage,
        projects: SqliteCanvasProjectRecordStorage,
        boards: SqliteBoardRecordStorage,
    ) -> None:
        from invokeai.app.services.canvas_project_records.canvas_project_records_common import (
            CanvasProjectRecordChanges,
        )

        board_id = boards.save(board_name="Counted", user_id="system").board_id
        _save_project(projects, "p_normal")
        _save_project(projects, "p_intermediate")
        projects.update("p_intermediate", CanvasProjectRecordChanges(is_intermediate=True))
        store.add_project_to_board(board_id=board_id, project_name="p_normal")
        store.add_project_to_board(board_id=board_id, project_name="p_intermediate")
        # Intermediate projects are excluded from the count, mirroring how images/videos do it.
        assert store.get_project_count_for_board(board_id) == 1


class TestForeignKeyCascade:
    def test_deleting_project_removes_association(
        self,
        store: SqliteBoardCanvasProjectRecordStorage,
        projects: SqliteCanvasProjectRecordStorage,
        seeded: tuple[str, str],
    ) -> None:
        # `board_canvas_projects` FK CASCADE on `canvas_projects.project_name` must drop the
        # association row when the project is deleted directly via the records service.
        board_a, _ = seeded
        store.add_project_to_board(board_id=board_a, project_name="p1")
        projects.delete("p1")
        # The orphaned association row is gone — `get_board_for_project` returns None and the
        # board listing no longer includes p1.
        assert store.get_board_for_project("p1") is None
        assert "p1" not in store.get_all_board_project_names_for_board(board_a)
