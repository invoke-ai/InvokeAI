import pytest

from invokeai.app.services.board_records.board_records_common import (
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
)
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


def _create_board_storage() -> SqliteBoardRecordStorage:
    config = InvokeAIAppConfig(use_memory_db=True)
    db = create_mock_sqlite_database(config=config, logger=InvokeAILogger.get_logger())
    return SqliteBoardRecordStorage(db=db)


def test_sql_injection_payload_in_board_name_is_stored_as_plain_text() -> None:
    storage = _create_board_storage()

    payload = "name'); DROP TABLE boards; --"
    injected_board = storage.save(payload)

    fetched = storage.get(injected_board.board_id)
    assert fetched.board_name == payload

    another_board = storage.save("safe board")
    assert storage.get(another_board.board_id).board_name == "safe board"


def test_sql_injection_payload_in_board_id_does_not_bypass_where_clause() -> None:
    storage = _create_board_storage()

    storage.save("first board")
    storage.save("second board")

    payload = "does-not-exist' OR '1'='1"

    with pytest.raises(BoardRecordNotFoundException):
        storage.get(payload)


def test_sql_injection_payload_in_delete_does_not_delete_other_rows() -> None:
    storage = _create_board_storage()

    first = storage.save("first board")
    second = storage.save("second board")

    payload = f"{first.board_id}' OR '1'='1"
    storage.delete(payload)

    remaining = storage.get_many(
        order_by=BoardRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Ascending,
        limit=10,
        offset=0,
        include_archived=True,
    )

    assert {board.board_id for board in remaining.items} == {first.board_id, second.board_id}
