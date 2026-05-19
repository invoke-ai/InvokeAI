"""Tests for SqlModelBoardRecordStorage."""

import pytest

from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
    BoardVisibility,
)
from invokeai.app.services.board_records.board_records_sqlmodel import SqlModelBoardRecordStorage
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


@pytest.fixture
def storage(db: SqliteDatabase) -> SqlModelBoardRecordStorage:
    return SqlModelBoardRecordStorage(db=db)


def test_save_and_get(storage: SqlModelBoardRecordStorage):
    board = storage.save("Test Board", "user1")
    assert board.board_name == "Test Board"
    assert board.user_id == "user1"

    fetched = storage.get(board.board_id)
    assert fetched.board_name == "Test Board"


def test_get_nonexistent(storage: SqlModelBoardRecordStorage):
    with pytest.raises(BoardRecordNotFoundException):
        storage.get("nonexistent")


def test_update_name(storage: SqlModelBoardRecordStorage):
    board = storage.save("Original", "user1")
    updated = storage.update(board.board_id, BoardChanges(board_name="Updated"))
    assert updated.board_name == "Updated"


def test_update_archived(storage: SqlModelBoardRecordStorage):
    board = storage.save("Board", "user1")
    updated = storage.update(board.board_id, BoardChanges(archived=True))
    assert updated.archived is True


def test_update_visibility(storage: SqlModelBoardRecordStorage):
    board = storage.save("Board", "user1")
    updated = storage.update(board.board_id, BoardChanges(board_visibility=BoardVisibility.Shared))
    assert updated.board_visibility == BoardVisibility.Shared


def test_delete(storage: SqlModelBoardRecordStorage):
    board = storage.save("Board", "user1")
    storage.delete(board.board_id)
    with pytest.raises(BoardRecordNotFoundException):
        storage.get(board.board_id)


def test_get_many_pagination(storage: SqlModelBoardRecordStorage):
    for i in range(5):
        storage.save(f"Board {i}", "user1")

    result = storage.get_many(
        user_id="user1",
        is_admin=False,
        order_by=BoardRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Ascending,
        offset=0,
        limit=3,
    )
    assert len(result.items) == 3
    assert result.total == 5


def test_get_many_admin_sees_all(storage: SqlModelBoardRecordStorage):
    storage.save("User1 Board", "user1")
    storage.save("User2 Board", "user2")

    result = storage.get_many(
        user_id="admin",
        is_admin=True,
        order_by=BoardRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Ascending,
    )
    assert result.total == 2


def test_get_many_user_sees_own_and_shared(storage: SqlModelBoardRecordStorage):
    storage.save("User1 Board", "user1")
    storage.save("User2 Board", "user2")
    b3 = storage.save("Shared Board", "user1")
    storage.update(b3.board_id, BoardChanges(board_visibility=BoardVisibility.Shared))

    # User2 sees own + shared
    result = storage.get_many(
        user_id="user2",
        is_admin=False,
        order_by=BoardRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Ascending,
    )
    names = [b.board_name for b in result.items]
    assert "User2 Board" in names
    assert "Shared Board" in names
    assert "User1 Board" not in names


def test_get_many_exclude_archived(storage: SqlModelBoardRecordStorage):
    storage.save("Active", "user1")
    b2 = storage.save("Archived", "user1")
    storage.update(b2.board_id, BoardChanges(archived=True))

    result = storage.get_many(
        user_id="user1",
        is_admin=True,
        order_by=BoardRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Ascending,
        include_archived=False,
    )
    assert result.total == 1
    assert result.items[0].board_name == "Active"


def test_get_all(storage: SqlModelBoardRecordStorage):
    for i in range(3):
        storage.save(f"Board {i}", "user1")

    boards = storage.get_all(
        user_id="user1",
        is_admin=True,
        order_by=BoardRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Ascending,
    )
    assert len(boards) == 3


def test_get_all_order_by_name(storage: SqlModelBoardRecordStorage):
    storage.save("Zebra", "user1")
    storage.save("Alpha", "user1")

    boards = storage.get_all(
        user_id="user1",
        is_admin=True,
        order_by=BoardRecordOrderBy.Name,
        direction=SQLiteDirection.Ascending,
    )
    assert boards[0].board_name == "Alpha"
    assert boards[1].board_name == "Zebra"


def test_sql_injection_in_name(storage: SqlModelBoardRecordStorage):
    payload = "name'); DROP TABLE boards; --"
    board = storage.save(payload, "user1")
    fetched = storage.get(board.board_id)
    assert fetched.board_name == payload


def test_sql_injection_in_id(storage: SqlModelBoardRecordStorage):
    storage.save("board", "user1")
    with pytest.raises(BoardRecordNotFoundException):
        storage.get("fake' OR '1'='1")
