"""The database work of the image import script, on a database file of its own."""

import datetime
import sqlite3
from contextlib import closing
from pathlib import Path

from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_common import BoardChanges
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.frontend.install.import_images import DatabaseMapper
from tests.fixtures.database import migrate_to_newest


def test_imports_images_onto_boards_found_or_created_by_name(tmp_path: Path) -> None:
    path = tmp_path / "databases" / "invokeai.db"
    database = Database.open_sqlite(path, InvokeAILogger.get_logger("test"))
    migrate_to_newest(database, tmp_path)
    database.dispose()
    mapper = DatabaseMapper(str(path), str(tmp_path / "databases" / "backup"))
    mapper.connect()
    try:
        assert mapper.get_board_names() == []
        board_id = mapper.get_board_id_with_create("Import")
        assert mapper.get_board_id_with_create("IMPORT") == board_id
        assert mapper.get_board_names() == ["Import"]

        # Any board of the name serves, whoever owns it, archived or not.
        assert mapper.database is not None
        other = UserService(mapper.database).create(
            UserCreateRequest(email="other@test.com", password="OtherPass123", display_name="other")
        )
        boards = BoardRecordStorage(mapper.database)
        theirs = boards.save("Theirs", other.user_id).board_id
        archived = boards.save("Archived", "system").board_id
        boards.update(archived, BoardChanges(archived=True))
        assert mapper.get_board_id_with_create("THEIRS") == theirs
        assert mapper.get_board_id_with_create("archived") == archived

        assert not mapper.does_image_exist("old.png")
        modified = datetime.datetime(2001, 2, 3, 4, 5, 6, 789000)
        mapper.add_new_image_to_database("old.png", 64, 32, '{"seed": 1}', modified)
        mapper.add_image_to_board("old.png", board_id)
        assert mapper.does_image_exist("old.png")

        mapper.backup("20010203T040506Z")
    finally:
        mapper.disconnect()

    database = Database.open_sqlite(path, InvokeAILogger.get_logger("test"))
    try:
        record = ImageRecordStorage(database).get("old.png")
        # The image keeps the time its file was made, so it sorts among the images of that time.
        assert str(record.created_at).startswith("2001-02-03 04:05:06.789")
        assert (record.width, record.height) == (64, 32)
        assert ImageRecordStorage(database).get_metadata("old.png") is not None
        assert BoardImageRecordStorage(database).get_board_for_image("old.png") == board_id
    finally:
        database.dispose()
    backup = tmp_path / "databases" / "backup" / "backup-20010203T040506Z-invokeai.db"
    with closing(sqlite3.connect(backup)) as copy:
        assert copy.execute("SELECT image_name FROM images").fetchall() == [("old.png",)]
