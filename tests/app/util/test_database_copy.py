"""`invoke-db-copy`: an install's SQLite database into a new server database, checked before and verified after."""

import json
import logging
from pathlib import Path
from typing import Optional
from unittest.mock import Mock

import pytest
from sqlalchemy import URL, insert, select

from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.config.config_default import load_config_from_root
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database import startup
from invokeai.app.services.shared.database.copy import count_normalized, delete_orphans, find_orphans
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import board_images
from invokeai.app.services.shared.database.schema.image_index import image_index_vocab_terms
from invokeai.app.services.shared.database.schema.models import models
from invokeai.app.services.shared.database.startup import init_database, open_migrated_database
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.app.util.database_copy import copy
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.database import external_test_db_url

server_only = pytest.mark.skipif(
    external_test_db_url() is None, reason="needs a MySQL or MariaDB server (INVOKEAI_TEST_DB_URL)"
)
LOGGER = InvokeAILogger.get_logger("test_database_copy")
MODEL_CONFIG = {
    "hash": "blake3:0123456789abcdef",
    "base": "sdxl",
    "type": "main",
    "path": "sdxl/main/model.safetensors",
    "format": "checkpoint",
    "name": "Model",
    "source": "https://example.com/model.safetensors",
    "source_type": "url",
    # Stored as JSON wrote it, which a server's integer column refuses.
    "file_size": 6_938_078_334.5,
}


def _install(root: Path, *, orphan: bool) -> None:
    """An install whose SQLite database holds an account, a board with an image, a model, two vocabulary terms a
    server treats as one, and (with `orphan`) an image membership of a board that is gone."""
    (root / "invokeai.yaml").write_text('schema_version: "4.0.3"\n')
    database = init_database(load_config_from_root(root), LOGGER, Mock(spec=ImageFileStorageBase))
    try:
        user = UserService(database).create(UserCreateRequest(email="alice@test.com", password="AlicePass123"))
        board = BoardRecordStorage(database).save("Holiday", user.user_id)
        ImageRecordStorage(database).save(
            image_name="beach.png",
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            width=64,
            height=64,
            has_workflow=False,
            user_id=user.user_id,
        )
        BoardImageRecordStorage(database).add_image_to_board(board.board_id, "beach.png")
        with database.begin(write=True) as conn:
            conn.execute(insert(models).values(id="model-1", config=json.dumps(MODEL_CONFIG)))
            conn.execute(insert(image_index_vocab_terms), [{"term": "Äpfel"}, {"term": "äpfel"}, {"term": "beach"}])
        if orphan:
            raw = database.sqlite.conn
            raw.execute("PRAGMA foreign_keys = OFF")
            raw.execute("DELETE FROM board_images")
            raw.execute("INSERT INTO board_images (board_id, image_name) VALUES ('gone', 'beach.png')")
            raw.commit()
            raw.execute("PRAGMA foreign_keys = ON")
    finally:
        database.dispose()


def test_the_check_finds_orphans_and_what_the_copy_changes(tmp_path: Path) -> None:
    _install(tmp_path, orphan=True)
    database = open_migrated_database(load_config_from_root(tmp_path), LOGGER)
    try:
        orphans = find_orphans(database)
        assert [(problem.table, problem.count) for problem in orphans] == [("board_images", 1)]
        assert [(problem.table, problem.count) for problem in count_normalized(database)] == [("models", 1)]

        assert delete_orphans(database) == 1
        assert find_orphans(database) == []
    finally:
        database.dispose()


@server_only
def test_a_copy_holds_every_row_of_its_source(
    tmp_path: Path,
    empty_database: Database,
    _external_test_schema: Optional[URL],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _external_test_schema is not None
    # The test servers keep their own defaults; what is checked here is the copy.
    monkeypatch.setattr(startup, "MINIMUM_MAX_ALLOWED_PACKET", 1)
    _install(tmp_path, orphan=True)
    url = _external_test_schema.render_as_string(hide_password=False)

    assert copy(url, root=tmp_path, check_only=False, skip_orphans=False) == 1
    assert "--orphans skip" in capsys.readouterr().out

    assert copy(url, root=tmp_path, check_only=True, skip_orphans=True) == 0
    assert "Nothing was copied" in capsys.readouterr().out

    assert copy(url, root=tmp_path, check_only=False, skip_orphans=True) == 0
    output = capsys.readouterr().out
    assert "every table matches its source" in output
    assert "image_index_vocab_terms: 1 rows the target treats as equal to others were merged" in output

    # The app takes the copy as it takes a database it created.
    config = load_config_from_root(tmp_path).model_copy(update={"db_url": url})
    target = open_migrated_database(config, LOGGER)
    try:
        [user] = [user for user in UserService(target).list_users() if user.email == "alice@test.com"]
        [board] = BoardRecordStorage(target).get_all(user.user_id, False, "board_name", "ASC")  # type: ignore[arg-type]
        assert board.board_name == "Holiday"
        assert ImageRecordStorage(target).exists("beach.png")
        with target.begin(write=False) as conn:
            assert conn.execute(select(models.c.file_size)).scalar_one() == round(MODEL_CONFIG["file_size"])
            assert conn.execute(select(board_images)).all() == []
    finally:
        target.dispose()

    # A database is copied once: into a new, empty one.
    assert copy(url, root=tmp_path, check_only=False, skip_orphans=True) == 1
    assert "not empty" in capsys.readouterr().out


def test_the_copy_reports_a_target_it_cannot_reach(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _install(tmp_path, orphan=False)
    logging.getLogger("invoke-db-copy").setLevel(logging.CRITICAL)

    assert copy("postgresql://u:p@localhost/db", root=tmp_path, check_only=True, skip_orphans=False) == 1
    assert "Cannot copy" in capsys.readouterr().out
