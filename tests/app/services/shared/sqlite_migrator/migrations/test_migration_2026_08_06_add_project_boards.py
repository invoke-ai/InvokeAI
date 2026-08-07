import json
import sqlite3
from logging import Logger
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.shared.sqlite_migrator.migration_loader import MigrationBuildContext, build_migrations
from invokeai.app.services.shared.sqlite_migrator.migrations.migration_2026_08_06_add_project_boards import (
    AddProjectBoardsMigrationCallback,
    build_migration,
)
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_impl import SqliteMigrator

MIGRATION_ID = "2026_08_06_add_project_boards"


def _make_db() -> sqlite3.Connection:
    """A database holding just the tables this migration reads and writes."""
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys = ON;")
    db.executescript(
        """
        CREATE TABLE users (user_id TEXT NOT NULL PRIMARY KEY);

        CREATE TABLE boards (
            board_id TEXT NOT NULL PRIMARY KEY,
            board_name TEXT NOT NULL,
            cover_image_name TEXT,
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            deleted_at DATETIME,
            archived BOOLEAN DEFAULT FALSE,
            user_id TEXT DEFAULT 'system',
            is_public BOOLEAN NOT NULL DEFAULT FALSE,
            board_visibility TEXT NOT NULL DEFAULT 'private'
        );

        CREATE TABLE shared_boards (
            board_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            can_edit BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY (board_id, user_id),
            FOREIGN KEY (board_id) REFERENCES boards(board_id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TABLE images (image_name TEXT NOT NULL PRIMARY KEY);
        CREATE TABLE videos (video_name TEXT NOT NULL PRIMARY KEY);

        CREATE TABLE board_images (
            board_id TEXT NOT NULL,
            image_name TEXT NOT NULL,
            PRIMARY KEY (image_name),
            FOREIGN KEY (board_id) REFERENCES boards (board_id) ON DELETE CASCADE,
            FOREIGN KEY (image_name) REFERENCES images (image_name) ON DELETE CASCADE
        );

        CREATE TABLE board_videos (
            board_id TEXT NOT NULL,
            video_name TEXT NOT NULL,
            PRIMARY KEY (video_name),
            FOREIGN KEY (board_id) REFERENCES boards (board_id) ON DELETE CASCADE,
            FOREIGN KEY (video_name) REFERENCES videos (video_name) ON DELETE CASCADE
        );

        CREATE TABLE projects (
            project_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            data TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1,
            created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
            PRIMARY KEY (user_id, project_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        CREATE TRIGGER tg_projects_updated_at
        AFTER UPDATE ON projects
        FOR EACH ROW
        BEGIN
          UPDATE projects
            SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
          WHERE user_id = OLD.user_id AND project_id = OLD.project_id;
        END;
        """
    )
    return db


def _add_user(db: sqlite3.Connection, user_id: str) -> None:
    db.execute("INSERT INTO users VALUES (?);", (user_id,))


def _add_board(
    db: sqlite3.Connection,
    board_id: str,
    *,
    user_id: str = "u1",
    name: str = "Board",
    visibility: str = "private",
    archived: bool = False,
) -> None:
    db.execute(
        "INSERT INTO boards (board_id, board_name, user_id, board_visibility, archived) VALUES (?, ?, ?, ?, ?);",
        (board_id, name, user_id, visibility, archived),
    )


def _gallery_document(*board_ids: Optional[str], legacy: bool = False) -> str:
    """A project document naming one gallery widget per supplied board id."""
    if legacy:
        (board_id,) = board_ids
        return json.dumps({"widgetStates": {"gallery": {"values": {"projectBoardId": board_id}}}})

    instances: dict[str, Any] = {}
    for index, board_id in enumerate(board_ids):
        values: dict[str, Any] = {} if board_id is None else {"projectBoardId": board_id}
        instances[f"gallery-{index}"] = {"typeId": "gallery", "state": {"values": values}}
    # A non-gallery widget that happens to carry the key must never be read.
    instances["canvas-0"] = {"typeId": "canvas", "state": {"values": {"projectBoardId": "board-decoy"}}}
    return json.dumps({"widgetInstances": instances})


def _add_project(
    db: sqlite3.Connection,
    project_id: str,
    *,
    user_id: str = "u1",
    name: str = "Project",
    data: str = "{}",
) -> None:
    db.execute(
        "INSERT INTO projects (project_id, user_id, name, data) VALUES (?, ?, ?, ?);",
        (project_id, user_id, name, data),
    )


def _board_of(db: sqlite3.Connection, project_id: str, user_id: str = "u1") -> str:
    cursor = db.execute("SELECT board_id FROM projects WHERE user_id = ? AND project_id = ?;", (user_id, project_id))
    row = cursor.fetchone()
    assert row is not None
    return row[0]


def _run(db: sqlite3.Connection) -> None:
    AddProjectBoardsMigrationCallback()(db.cursor())


# --- adoption -------------------------------------------------------------------------------


def test_adopts_the_single_valid_candidate_and_renames_it_to_the_project() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_board(db, "board-1", name="Old board name")
    _add_project(db, "p1", name="My project", data=_gallery_document("board-1"))

    _run(db)

    assert _board_of(db, "p1") == "board-1"
    assert db.execute("SELECT board_name FROM boards WHERE board_id = 'board-1';").fetchone()[0] == "My project"


def test_adopts_a_candidate_written_in_the_legacy_widget_states_shape() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_board(db, "board-1")
    _add_project(db, "p1", data=_gallery_document("board-1", legacy=True))

    _run(db)

    assert _board_of(db, "p1") == "board-1"


def test_adopts_when_several_gallery_instances_name_the_same_board() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_board(db, "board-1")
    _add_project(db, "p1", data=_gallery_document("board-1", "board-1"))

    _run(db)

    assert _board_of(db, "p1") == "board-1"


def test_creates_a_fresh_board_when_two_candidates_disagree() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_board(db, "board-1")
    _add_board(db, "board-2")
    _add_project(db, "p1", data=_gallery_document("board-1", "board-2"))

    _run(db)

    assert _board_of(db, "p1") not in {"board-1", "board-2"}


def test_creates_a_fresh_board_for_each_disqualified_candidate() -> None:
    """Every ownership test, one project each, so a regression names the rule it broke."""
    db = _make_db()
    _add_user(db, "u1")
    _add_user(db, "u2")

    _add_board(db, "board-foreign", user_id="u2")
    _add_project(db, "p-foreign", data=_gallery_document("board-foreign"))

    _add_board(db, "board-shared-visibility", visibility="shared")
    _add_project(db, "p-shared-visibility", data=_gallery_document("board-shared-visibility"))

    _add_board(db, "board-public", visibility="public")
    _add_project(db, "p-public", data=_gallery_document("board-public"))

    _add_board(db, "board-share-row")
    db.execute("INSERT INTO shared_boards (board_id, user_id) VALUES ('board-share-row', 'u2');")
    _add_project(db, "p-share-row", data=_gallery_document("board-share-row"))

    _add_project(db, "p-missing", data=_gallery_document("board-does-not-exist"))
    _add_project(db, "p-none", data="{}")
    _add_project(db, "p-malformed", data="not json at all")
    _add_project(db, "p-empty-id", data=_gallery_document(""))
    _add_project(db, "p-no-values", data=_gallery_document(None))

    _run(db)

    disqualified = {
        "board-foreign",
        "board-shared-visibility",
        "board-public",
        "board-share-row",
        "board-does-not-exist",
        "board-decoy",
    }
    for project_id in (
        "p-foreign",
        "p-shared-visibility",
        "p-public",
        "p-share-row",
        "p-missing",
        "p-none",
        "p-malformed",
        "p-empty-id",
        "p-no-values",
    ):
        assert _board_of(db, project_id) not in disqualified


def test_the_earlier_project_wins_a_board_two_projects_both_name() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_board(db, "board-1")
    _add_project(db, "p-first", data=_gallery_document("board-1"))
    _add_project(db, "p-second", data=_gallery_document("board-1"))

    _run(db)

    assert _board_of(db, "p-first") == "board-1"
    assert _board_of(db, "p-second") != "board-1"


def test_two_users_may_share_a_project_id_and_still_get_distinct_boards() -> None:
    """`projects` is keyed (user_id, project_id), so the same id on two users is legal."""
    db = _make_db()
    _add_user(db, "u1")
    _add_user(db, "u2")
    _add_project(db, "shared-id", user_id="u1", name="Mine")
    _add_project(db, "shared-id", user_id="u2", name="Theirs")

    _run(db)

    assert _board_of(db, "shared-id", "u1") != _board_of(db, "shared-id", "u2")


def test_a_created_board_is_private_unarchived_and_owned_by_the_project_owner() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_project(db, "p1", name="Fresh project")

    _run(db)

    row = db.execute(
        "SELECT board_name, user_id, board_visibility, archived FROM boards WHERE board_id = ?;",
        (_board_of(db, "p1"),),
    ).fetchone()
    assert row == ("Fresh project", "u1", "private", 0)


def test_an_adopted_board_is_un_archived() -> None:
    """Someone who archived their project's gallery board before upgrading would otherwise end up
    with a project whose board is hidden from every listing, and no route left to unhide it: the
    generic board API refuses `archived` on a claimed board, and there is no project archive API."""
    db = _make_db()
    _add_user(db, "u1")
    _add_board(db, "b1", name="Archived board", archived=True)
    _add_project(db, "p1", name="Project", data=_gallery_document("b1"))

    _run(db)

    assert _board_of(db, "p1") == "b1"
    archived = db.execute("SELECT archived FROM boards WHERE board_id = 'b1';").fetchone()[0]
    assert archived == 0


def test_a_long_project_name_is_truncated_to_what_the_board_api_accepts() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_project(db, "p1", name="x" * 400)

    _run(db)

    name = db.execute("SELECT board_name FROM boards WHERE board_id = ?;", (_board_of(db, "p1"),)).fetchone()[0]
    assert len(name) == 300


# --- structure ------------------------------------------------------------------------------


def test_every_project_gets_a_unique_board_and_the_column_rejects_reuse() -> None:
    db = _make_db()
    _add_user(db, "u1")
    for index in range(5):
        _add_project(db, f"p{index}")

    _run(db)

    board_ids = [row[0] for row in db.execute("SELECT board_id FROM projects;").fetchall()]
    assert len(board_ids) == 5
    assert all(board_id is not None for board_id in board_ids)
    assert len(set(board_ids)) == 5

    try:
        db.execute(
            "INSERT INTO projects (project_id, user_id, name, data, board_id) VALUES ('px','u1','n','{}',?);",
            (board_ids[0],),
        )
        raise AssertionError("a second project claimed the same board")
    except sqlite3.IntegrityError:
        pass


def test_a_claimed_board_cannot_be_deleted_out_from_under_its_project() -> None:
    """The FK is the database-level backstop behind the API's 409."""
    db = _make_db()
    _add_user(db, "u1")
    _add_project(db, "p1")
    _run(db)

    try:
        db.execute("DELETE FROM boards WHERE board_id = ?;", (_board_of(db, "p1"),))
        raise AssertionError("a claimed board was deleted")
    except sqlite3.IntegrityError:
        pass


def test_board_membership_is_never_touched() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_board(db, "board-1")
    _add_board(db, "board-2")
    db.execute("INSERT INTO images VALUES ('i1.png');")
    db.execute("INSERT INTO videos VALUES ('v1.mp4');")
    db.execute("INSERT INTO board_images VALUES ('board-1', 'i1.png');")
    db.execute("INSERT INTO board_videos VALUES ('board-2', 'v1.mp4');")
    _add_project(db, "p1", data=_gallery_document("board-1"))

    _run(db)

    assert db.execute("SELECT board_id, image_name FROM board_images;").fetchall() == [("board-1", "i1.png")]
    assert db.execute("SELECT board_id, video_name FROM board_videos;").fetchall() == [("board-2", "v1.mp4")]


def test_the_rebuilt_table_keeps_its_key_columns_ordering_and_trigger() -> None:
    db = _make_db()
    _add_user(db, "u1")
    db.execute(
        "INSERT INTO projects (project_id, user_id, name, data, revision, created_at, updated_at)"
        " VALUES ('p1','u1','first','{}', 7, '2020-01-01 00:00:00.000', '2020-01-02 00:00:00.000');"
    )
    # Same created_at as p1, so only insertion order can separate them.
    db.execute(
        "INSERT INTO projects (project_id, user_id, name, data, revision, created_at, updated_at)"
        " VALUES ('p2','u1','second','{}', 1, '2020-01-01 00:00:00.000', '2020-01-02 00:00:00.000');"
    )

    _run(db)

    assert db.execute(
        "SELECT project_id, name, revision, created_at, updated_at FROM projects ORDER BY created_at, rowid;"
    ).fetchall() == [
        ("p1", "first", 7, "2020-01-01 00:00:00.000", "2020-01-02 00:00:00.000"),
        ("p2", "second", 1, "2020-01-01 00:00:00.000", "2020-01-02 00:00:00.000"),
    ]

    # The composite primary key survives the rebuild.
    try:
        _add_project(db, "p1")
        raise AssertionError("duplicate (user_id, project_id) was accepted")
    except sqlite3.IntegrityError:
        pass

    db.execute("UPDATE projects SET name = 'renamed' WHERE project_id = 'p1';")
    assert db.execute("SELECT updated_at FROM projects WHERE project_id = 'p1';").fetchone()[0] != (
        "2020-01-02 00:00:00.000"
    )


def test_deleting_a_user_still_cascades_their_projects() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_project(db, "p1")
    _run(db)

    db.execute("DELETE FROM users WHERE user_id = 'u1';")

    assert db.execute("SELECT COUNT(*) FROM projects;").fetchone()[0] == 0


def test_foreign_keys_are_consistent_afterwards() -> None:
    db = _make_db()
    _add_user(db, "u1")
    _add_board(db, "board-1")
    _add_project(db, "p1", data=_gallery_document("board-1"))
    _add_project(db, "p2")

    _run(db)

    assert db.execute("PRAGMA foreign_key_check;").fetchall() == []


def test_an_empty_projects_table_migrates_cleanly() -> None:
    db = _make_db()

    _run(db)

    assert db.execute("SELECT COUNT(*) FROM projects;").fetchone()[0] == 0
    assert db.execute("PRAGMA foreign_key_check;").fetchall() == []


# --- registration ---------------------------------------------------------------------------


def test_migration_has_a_dated_id_and_depends_on_the_projects_repair() -> None:
    migration = build_migration()
    assert migration.id == MIGRATION_ID
    assert migration.depends_on == "2026_07_30_repair_projects_table"
    assert migration.from_version is None
    assert migration.to_version is None


def test_it_runs_once_through_the_real_migrator_and_is_not_reapplied(tmp_path: Path) -> None:
    """End-to-end against the real schema: migrate to just before this migration, seed a project
    with a live board reference, then let the full migrator carry it the rest of the way."""
    logger = Logger("test")
    context = MigrationBuildContext(app_config=MagicMock(), logger=logger, image_files=MagicMock())
    all_migrations = build_migrations(context)

    db = SqliteDatabase(db_path=tmp_path / "projects.db", logger=logger, verbose=False)

    before = SqliteMigrator(db=db)
    for migration in all_migrations:
        if migration.id != MIGRATION_ID:
            before.register_migration(migration)
    before.run_migrations()

    cursor = db._conn.cursor()
    cursor.execute("INSERT INTO boards (board_id, board_name, user_id) VALUES ('b1', 'Old name', 'system');")
    cursor.execute(
        "INSERT INTO projects (project_id, user_id, name, data) VALUES ('p1', 'system', 'Kept', ?);",
        (_gallery_document("b1"),),
    )
    db._conn.commit()

    full = SqliteMigrator(db=db)
    for migration in all_migrations:
        full.register_migration(migration)
    assert full.run_migrations() is True

    # SqliteDatabase sets row_factory = sqlite3.Row, which does not compare equal to a tuple.
    cursor.execute("SELECT name, board_id FROM projects WHERE project_id = 'p1';")
    assert tuple(cursor.fetchone()) == ("Kept", "b1")
    cursor.execute("SELECT board_name FROM boards WHERE board_id = 'b1';")
    assert tuple(cursor.fetchone()) == ("Kept",)
    cursor.execute("PRAGMA foreign_key_check;")
    assert cursor.fetchall() == []

    # Migration tracking, not idempotent DDL, is what stops a second run rebuilding the table again.
    again = SqliteMigrator(db=db)
    for migration in all_migrations:
        again.register_migration(migration)
    assert again.run_migrations() is False
