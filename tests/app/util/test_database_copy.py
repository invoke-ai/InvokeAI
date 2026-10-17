"""`invoke-db-copy`: an install's SQLite database into a new server database, checked before and verified after."""

import json
from pathlib import Path
from typing import Optional
from unittest.mock import Mock

import pytest
from sqlalchemy import URL, delete, func, insert, select, update

from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.config.config_default import load_config_from_root
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database import startup
from invokeai.app.services.shared.database.copy import (
    Problem,
    copy_database,
    copy_records,
    count_normalized,
    delete_orphans,
    find_orphans,
    find_oversized,
    verify_copy,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import board_images, boards
from invokeai.app.services.shared.database.schema.image_index import image_index_vocab_terms
from invokeai.app.services.shared.database.schema.models import models
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.app.services.shared.database.startup import init_database, open_migrated_database
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.app.util import database_copy
from invokeai.app.util.database_copy import copy, copy_to_sqlite
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


def _install(root: Path, *, orphans: bool) -> None:
    """An install whose SQLite database holds an account, a board with an image, a model, two vocabulary terms a
    server treats as one, and (with `orphans`) an image membership of a board that is gone and a board whose cover
    image is gone."""
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
        if orphans:
            raw = database.sqlite.conn
            raw.execute("PRAGMA foreign_keys = OFF")
            raw.execute("DELETE FROM board_images")
            raw.execute("INSERT INTO board_images (board_id, image_name) VALUES ('gone', 'beach.png')")
            raw.execute("UPDATE boards SET cover_image_name = 'gone.png'")
            raw.commit()
            raw.execute("PRAGMA foreign_keys = ON")
    finally:
        database.dispose()


def _open(root: Path) -> Database:
    return open_migrated_database(load_config_from_root(root), LOGGER)


def test_orphans_are_found_and_left_out_or_cleared(tmp_path: Path) -> None:
    _install(tmp_path, orphans=True)
    database = _open(tmp_path)
    try:
        assert [(problem.table, problem.count) for problem in find_orphans(database)] == [
            ("board_images", 1),
            ("boards", 1),
        ]
        assert [(problem.table, problem.count) for problem in count_normalized(database)] == [("models", 1)]

        assert delete_orphans(database) == [
            Problem("board_images", 1, "left out"),
            Problem("boards", 1, "with a reference to a missing row, which was cleared"),
        ]
        assert find_orphans(database) == []
        # The board stays, without the cover the database would have cleared itself.
        with database.begin(write=False) as conn:
            assert conn.execute(select(boards.c.board_name, boards.c.cover_image_name)).all() == [("Holiday", None)]
    finally:
        database.dispose()


def test_values_a_server_cannot_store_are_found(tmp_path: Path) -> None:
    _install(tmp_path, orphans=False)
    database = _open(tmp_path)
    try:
        assert find_oversized(database, max_allowed_packet=64 * 1024 * 1024) == []

        with database.begin(write=True) as conn:
            conn.execute(insert(image_index_vocab_terms).values(term="t" * 256))
            # The server computes `path` from the config, into a column of 768 characters.
            config = {**MODEL_CONFIG, "path": "p" * 769}
            conn.execute(insert(models).values(id="m" * 256, config=json.dumps(config)))
        found = find_oversized(database, max_allowed_packet=64 * 1024 * 1024)
        assert sorted(found) == [
            Problem("image_index_vocab_terms", 1, "with a term longer than 255 characters"),
            Problem("models", 1, "with a path longer than 768 characters"),
            Problem("models", 1, "with an id longer than 255 characters"),
        ]

        # A row is sent in one statement: it must fit half a packet, which leaves room for escaping.
        assert Problem("models", 1, "larger than half the server's max_allowed_packet") in find_oversized(
            database, max_allowed_packet=1600
        )
    finally:
        database.dispose()


def test_a_copy_is_verified_row_by_row(tmp_path: Path) -> None:
    _install(tmp_path, orphans=False)
    source = _open(tmp_path)
    target = Database.open_sqlite(tmp_path / "target.db", LOGGER)
    try:
        copy_database(source, target)
        assert verify_copy(source, target) == []
        copy_records(source, target)
        assert verify_copy(source, target, records=True) == []

        with target.begin(write=True) as conn:
            conn.execute(update(boards).values(board_name="Renamed"))
            conn.execute(delete(board_images))
        assert [mismatch.split(":")[0] for mismatch in verify_copy(source, target)] == ["boards", "board_images"]
    finally:
        source.dispose()
        target.dispose()


def test_a_merged_table_holds_one_of_each_set_of_equal_rows_and_nothing_else(tmp_path: Path) -> None:
    _install(tmp_path, orphans=False)
    source = _open(tmp_path)
    target = Database.open_sqlite(tmp_path / "target.db", LOGGER)
    try:
        with source.begin(write=True) as conn:
            # Equal to "beach" on a server, whose collations ignore a soft hyphen.
            conn.execute(insert(image_index_vocab_terms).values(term="bea\u00adch"))
        copy_database(source, target)
        # As a server merges them.
        with target.begin(write=True) as conn:
            conn.execute(
                delete(image_index_vocab_terms).where(image_index_vocab_terms.c.term.in_(["äpfel", "bea\u00adch"]))
            )
        assert verify_copy(source, target) == []

        with target.begin(write=True) as conn:
            conn.execute(delete(image_index_vocab_terms).where(image_index_vocab_terms.c.term == "beach"))
        assert verify_copy(source, target) == ["image_index_vocab_terms: rows or contents differ"]

        with target.begin(write=True) as conn:
            conn.execute(insert(image_index_vocab_terms).values(term="beach", created_at="2000-01-01 00:00:00.000"))
        assert verify_copy(source, target) == ["image_index_vocab_terms: rows or contents differ"]
    finally:
        source.dispose()
        target.dispose()


def test_without_a_target_the_copy_asks_for_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _install(tmp_path, orphans=False)

    assert copy(None, root=tmp_path, check_only=True, skip_orphans=False) == 1
    assert "Name the target" in capsys.readouterr().out


def test_the_copy_reports_a_target_it_cannot_use(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _install(tmp_path, orphans=False)

    assert copy("postgresql://u:p@localhost/db", root=tmp_path, check_only=True, skip_orphans=False) == 1
    assert "Cannot copy" in capsys.readouterr().out


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
    _install(tmp_path, orphans=True)
    url = _external_test_schema.render_as_string(hide_password=False)
    # The target is the database the install's config names.
    (tmp_path / "invokeai.yaml").write_text(f'schema_version: "4.0.3"\ndb_url: "{url}"\n')

    assert copy(None, root=tmp_path, check_only=False, skip_orphans=False) == 1
    assert "--orphans skip" in capsys.readouterr().out

    assert copy(None, root=tmp_path, check_only=True, skip_orphans=True) == 0
    assert "Nothing was copied" in capsys.readouterr().out

    assert copy(None, root=tmp_path, check_only=False, skip_orphans=True) == 0
    output = capsys.readouterr().out
    assert "board_images: 1 row left out" in output
    assert "every table matches its source" in output
    assert "image_index_vocab_terms: 1 row the target treats as equal to others were merged" in output

    # The install's own database keeps what the copy left out.
    source = open_migrated_database(load_config_from_root(tmp_path).model_copy(update={"db_url": None}), LOGGER)
    try:
        assert [(problem.table, problem.count) for problem in find_orphans(source)] == [
            ("board_images", 1),
            ("boards", 1),
        ]
    finally:
        source.dispose()

    # The app takes the copy as it takes a database it created.
    target = open_migrated_database(load_config_from_root(tmp_path), LOGGER)
    try:
        [user] = [user for user in UserService(target).list_users() if user.email == "alice@test.com"]
        [board] = BoardRecordStorage(target).get_all(user.user_id, False, "board_name", "ASC")  # type: ignore[arg-type]
        assert (board.board_name, board.cover_image_name) == ("Holiday", None)
        assert ImageRecordStorage(target).exists("beach.png")
        with target.begin(write=False) as conn:
            assert conn.execute(select(models.c.file_size)).scalar_one() == round(MODEL_CONFIG["file_size"])
            assert conn.execute(select(func.count()).select_from(board_images)).scalar_one() == 0
    finally:
        target.dispose()

    # A database is copied once: into a new, empty one.
    assert copy(None, root=tmp_path, check_only=False, skip_orphans=True) == 1
    assert "not empty" in capsys.readouterr().out


@server_only
def test_a_failed_copy_leaves_a_target_the_app_refuses_and_says_how_to_start_again(
    tmp_path: Path,
    empty_database: Database,
    _external_test_schema: Optional[URL],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _external_test_schema is not None
    monkeypatch.setattr(startup, "MINIMUM_MAX_ALLOWED_PACKET", 1)
    _install(tmp_path, orphans=False)
    url = _external_test_schema.render_as_string(hide_password=False)

    def lost(source: Database, target: Database) -> None:
        raise ConnectionError("the connection to the server was lost")

    monkeypatch.setattr(database_copy, "copy_records", lost)

    assert copy(url, root=tmp_path, check_only=False, skip_orphans=False) == 1
    assert "Drop the target database" in capsys.readouterr().out
    config = load_config_from_root(tmp_path).model_copy(update={"db_url": url})
    with pytest.raises(Exception, match="no record of the migrations"):
        open_migrated_database(config, LOGGER).dispose()


def test_a_copy_to_sqlite_asks_for_its_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _install(tmp_path, orphans=False)

    assert copy_to_sqlite(tmp_path / "back.db", source_url=None, root=tmp_path, check_only=False) == 1
    assert "Name the source" in capsys.readouterr().out


def test_a_copy_to_sqlite_overwrites_no_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _install(tmp_path, orphans=False)
    existing = tmp_path / "databases" / "invokeai.db"
    before = existing.read_bytes()

    url = "mariadb+pymysql://invokeai:secret@127.0.0.1:1/invokeai"
    assert copy_to_sqlite(existing, source_url=url, root=tmp_path, check_only=False) == 1
    assert "overwrites no database" in capsys.readouterr().out
    assert existing.read_bytes() == before


def _moved_to_a_server(tmp_path: Path, url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """An install whose database `invoke-db-copy` moved to the server, which its config now names."""
    monkeypatch.setattr(startup, "MINIMUM_MAX_ALLOWED_PACKET", 1)
    _install(tmp_path, orphans=False)
    (tmp_path / "invokeai.yaml").write_text(f'schema_version: "4.0.3"\ndb_url: "{url}"\n')
    assert copy(None, root=tmp_path, check_only=False, skip_orphans=False) == 0


@server_only
def test_a_server_database_goes_back_to_sqlite(
    tmp_path: Path,
    empty_database: Database,
    _external_test_schema: Optional[URL],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _external_test_schema is not None
    url = _external_test_schema.render_as_string(hide_password=False)
    _moved_to_a_server(tmp_path, url, monkeypatch)

    # Work done on the server after the move, which the old SQLite file does not hold.
    server = open_migrated_database(load_config_from_root(tmp_path), LOGGER)
    try:
        with server.engine.connect() as conn:
            # As MySQL's background statistics update does: it then serves the id counter of that moment from a
            # cache, until the cache expires a day later.
            conn.exec_driver_sql("ANALYZE TABLE session_queue").all()
            conn.commit()
        [user] = [user for user in UserService(server).list_users() if user.email == "alice@test.com"]
        BoardRecordStorage(server).save("Made on the server", user.user_id)
        with server.begin(write=True) as conn:
            item = {"batch_id": "b", "queue_id": "default", "session": "{}"}
            conn.execute(insert(session_queue), [{**item, "session_id": f"s{n}"} for n in range(3)])
            newest = conn.execute(select(func.max(session_queue.c.item_id))).scalar_one()
            conn.execute(delete(session_queue).where(session_queue.c.item_id == newest))
    finally:
        server.dispose()
    capsys.readouterr()

    back = tmp_path / "back" / "invokeai.db"
    back.parent.mkdir()
    assert copy_to_sqlite(back, source_url=None, root=tmp_path, check_only=False) == 0
    assert "every table matches its source" in capsys.readouterr().out
    assert sorted(path.name for path in back.parent.iterdir()) == ["invokeai.db"]

    # The app takes the file as an up-to-date database, holding what the server held.
    config = load_config_from_root(tmp_path).model_copy(update={"db_url": None, "db_dir": back.parent})
    copied = open_migrated_database(config, LOGGER)
    try:
        boards_held = BoardRecordStorage(copied).get_all(user.user_id, False, "board_name", "ASC")  # type: ignore[arg-type]
        assert sorted(board.board_name for board in boards_held) == ["Holiday", "Made on the server"]
        with copied.begin(write=True) as conn:
            conn.execute(insert(session_queue).values(batch_id="b", queue_id="default", session="{}", session_id="new"))
            # The id the server issued and deleted is not issued again.
            assert conn.execute(select(func.max(session_queue.c.item_id))).scalar_one() > newest
    finally:
        copied.dispose()


@server_only
def test_a_copy_to_sqlite_waits_for_no_running_app_and_keeps_no_failed_copy(
    tmp_path: Path,
    empty_database: Database,
    _external_test_schema: Optional[URL],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _external_test_schema is not None
    url = _external_test_schema.render_as_string(hide_password=False)
    _moved_to_a_server(tmp_path, url, monkeypatch)
    back = tmp_path / "back.db"

    running = Database.open_url(url, LOGGER)
    try:
        running.hold_instance_lock()
        assert copy_to_sqlite(back, source_url=None, root=tmp_path, check_only=False) == 1
        assert "Another InvokeAI process" in capsys.readouterr().out
    finally:
        running.dispose()

    assert copy_to_sqlite(back, source_url=None, root=tmp_path, check_only=True) == 0
    assert "Nothing was copied" in capsys.readouterr().out
    assert not back.exists()

    def lost(source: Database, target: Database) -> None:
        raise ConnectionError("the connection to the server was lost")

    monkeypatch.setattr(database_copy, "copy_records", lost)
    assert copy_to_sqlite(back, source_url=None, root=tmp_path, check_only=False) == 1
    assert "not kept" in capsys.readouterr().out
    assert not back.exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("invoke-db-copy-")]

    # A file made at the target path while copying is not overwritten.
    def made_meanwhile(source: Database, target: Database) -> dict[str, int]:
        back.write_bytes(b"made meanwhile")
        return copy_records(source, target)

    monkeypatch.setattr(database_copy, "copy_records", made_meanwhile)
    assert copy_to_sqlite(back, source_url=None, root=tmp_path, check_only=False) == 1
    assert "overwrites no file" in capsys.readouterr().out
    assert back.read_bytes() == b"made meanwhile"
    assert not [path for path in tmp_path.iterdir() if path.name.startswith("invoke-db-copy-")]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--to-sqlite", "back.db", "--target", "mysql+pymysql://u:p@db/invokeai"],
        ["--to-sqlite", "back.db", "--orphans", "skip"],
        ["--source", "mysql+pymysql://u:p@db/invokeai"],
    ],
)
def test_arguments_of_the_other_direction_are_refused(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["invoke-db-copy", *arguments])

    with pytest.raises(SystemExit) as refused:
        database_copy.main()
    assert refused.value.code == 2
