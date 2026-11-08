"""Give every project a first-class board that SQLite owns.

A "project board" used to be a client-side fiction: an ordinary board whose id lived in the opaque
project document. Nothing on the server knew it belonged to a project, so it could be renamed,
archived or deleted out from under one, and two projects could name the same board.

`projects.board_id` becomes `NOT NULL UNIQUE` with a foreign key to `boards`, making "at most one
project per board" and "a claimed board cannot be deleted" database facts rather than application
conventions.

Existing projects keep their board **only when that is unambiguously safe**. The document is
client-written, so its `projectBoardId` is a hint to verify, never an assertion: adopted only when
exactly one distinct value survives every ownership test. Anything ambiguous gets a fresh empty
board — a project that adopts nothing merely starts empty, whereas one that adopts the wrong board
takes ownership of media it does not own.

Nothing moves between boards: `board_images` and `board_videos` are read but never written.
"""

import json
import sqlite3
from collections.abc import Callable
from logging import Logger
from typing import Any, Optional

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.app.util.misc import uuid_string

# `BoardChanges.board_name` caps names at 300 characters. Project names are unbounded, so a board
# adopting a project's name is truncated to stay inside what the generic board API would accept.
#
# Pinned here rather than imported from `board_records_common`, deliberately: a migration has to
# keep doing what it did on the day it shipped, and a database migrated at 300 does not un-migrate
# itself if that ceiling later moves.
BOARD_NAME_MAX_LENGTH = 300


class AddProjectBoardsMigrationCallback:
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        if _table_exists(cursor, "users"):
            self._quarantine_orphaned_projects(cursor)
            assignments = self._assign_boards(cursor)
        else:
            # No project can survive a rebuild whose foreign key has no table to point at, so every
            # one of them is quarantined and none is assigned a board. See
            # `_quarantine_every_project` for why a database can reach this migration in that state.
            self._quarantine_every_project(cursor)
            assignments = []

        self._rebuild_projects_table(cursor, assignments)

    def _quarantine(self, cursor: sqlite3.Cursor, where: str, describe: Callable[[str, str], str]) -> None:
        """Copy the project rows matching `where` into a FK-free table, keeping them whole.

        The rebuild re-inserts every row under `FOREIGN KEY (user_id) REFERENCES users` with
        `PRAGMA foreign_keys = ON`, which cannot be turned off inside the migrator's transaction, so
        an unreachable row would abort the whole migration — and with it the app's ability to open
        the database. Deleting the documents instead would make repair impossible.
        """
        cursor.execute(f"SELECT project_id, user_id FROM projects WHERE {where};")
        rows = cursor.fetchall()

        if not rows:
            return

        self._create_quarantine_table(cursor)
        cursor.execute(
            f"""--sql
            INSERT OR IGNORE INTO orphaned_projects_2026_08_06
                (project_id, user_id, name, data, revision, created_at, updated_at)
            SELECT project_id, user_id, name, data, revision, created_at, updated_at
            FROM projects WHERE {where};
            """
        )

        for project_id, user_id in rows:
            self._logger.warning(describe(project_id, user_id))

    def _quarantine_orphaned_projects(self, cursor: sqlite3.Cursor) -> None:
        """Rescue rows whose owner no longer exists, then delete them from the live table."""
        orphaned = "user_id NOT IN (SELECT user_id FROM users)"

        self._quarantine(
            cursor,
            orphaned,
            lambda project_id, user_id: (
                f"Project boards migration: quarantining project {project_id}, whose owner {user_id}"
                " no longer exists, in orphaned_projects_2026_08_06."
            ),
        )
        cursor.execute(f"DELETE FROM projects WHERE {orphaned};")

    def _quarantine_every_project(self, cursor: sqlite3.Cursor) -> None:
        """The same rescue, for a database that has no `users` table at all.

        `users` is created by *this fork's* `migration_27`. A root last opened by an upstream build
        at schema version 27 or later already has that id recorded, against upstream's unrelated
        migration of the same number, so the fork's version never runs there — the same shared-root
        divergence `2026_07_30_repair_projects_table` exists to handle for `projects`. Every earlier
        reference to `users` was a foreign key clause, which SQLite does not resolve until DML, so
        such a root opened without complaint until this migration became the first to read the table.

        Note what is *not* done here, unlike the orphan case: the rows are copied out but never
        deleted. With the foreign key enforced and its target missing, SQLite refuses any DML against
        `projects` — including a `DELETE`. Dropping the table in the rebuild is what removes them,
        and that needs no resolution.
        """
        self._quarantine(
            cursor,
            "1 = 1",
            lambda project_id, user_id: (
                f"Project boards migration: quarantining project {project_id} in"
                " orphaned_projects_2026_08_06. This database has no users table, so its owner"
                f" {user_id} cannot be resolved — it was last opened by an upstream build and this"
                " fork's multiuser migration never ran on it."
            ),
        )

    def _create_quarantine_table(self, cursor: sqlite3.Cursor) -> None:
        """Deliberately free of foreign keys, so it can hold rows the live table cannot."""
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS orphaned_projects_2026_08_06 (
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                data TEXT NOT NULL,
                revision INTEGER NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                quarantined_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (user_id, project_id)
            );
            """
        )

    def _assign_boards(self, cursor: sqlite3.Cursor) -> list[tuple[int, str]]:
        """Resolve one board per project, creating boards as needed.

        Returns (rowid, board_id) pairs, visited in `rowid` order so the earlier project wins when
        two name the same board.

        Row at a time, through a cursor of its own: `data` is a whole project document, so
        `fetchall()` would hold every project in memory at startup. The second cursor is what makes
        that safe — the writes below run on `cursor`, which would discard the result set being
        iterated.
        """
        reader = cursor.connection.cursor()
        reader.execute(
            """--sql
            SELECT rowid, project_id, user_id, name, data FROM projects ORDER BY rowid ASC;
            """
        )

        assignments: list[tuple[int, str]] = []
        # Boards claimed earlier in this run. The column does not exist yet, so this set is the only
        # record of what has already been spoken for.
        claimed: set[str] = set()
        adopted = 0

        for rowid, project_id, user_id, name, data in reader:
            board_id = self._adopt_board(cursor, project_id=project_id, user_id=user_id, data=data, claimed=claimed)

            if board_id is None:
                board_id = self._insert_board(cursor, user_id=user_id, name=name)
            else:
                adopted += 1
                # Un-archived as well as renamed, and for the same reason the claim path does it:
                # a project's board follows the project, and the generic board API refuses to
                # change `archived` on a claimed board. Someone who had archived their project's
                # gallery board before upgrading would otherwise end up with a project whose board
                # is invisible in every listing and unreachable by any route.
                cursor.execute(
                    """--sql
                    UPDATE boards SET board_name = ?, archived = FALSE WHERE board_id = ?;
                    """,
                    (name[:BOARD_NAME_MAX_LENGTH], board_id),
                )

            claimed.add(board_id)
            assignments.append((rowid, board_id))

        reader.close()
        self._logger.info(
            f"Project boards migration: {len(assignments)} project(s), {adopted} kept the board they were using."
        )

        return assignments

    def _adopt_board(
        self, cursor: sqlite3.Cursor, *, project_id: str, user_id: str, data: str, claimed: set[str]
    ) -> Optional[str]:
        """Return the project's existing board if exactly one candidate is safe to adopt.

        Every refusal is logged. A project that adopts nothing starts on a fresh empty board and its
        old gallery contents stay where they are — the right way to be wrong, but invisible: the
        person sees a project that has apparently lost its images, and without this an operator has
        no way to find out which projects were affected or why.
        """
        candidates = _collect_board_candidates(data)

        # Two gallery widgets disagreeing about which board is the project's leaves no way to pick
        # correctly, and guessing would hand a project someone else's media.
        if len(candidates) != 1:
            if candidates:
                self._log_refusal(project_id, f"its document names {len(candidates)} boards")
            return None

        board_id = next(iter(candidates))

        if board_id in claimed:
            self._log_refusal(project_id, f"board {board_id} was already claimed by an earlier project")
            return None

        cursor.execute(
            """--sql
            SELECT
                b.user_id,
                b.board_visibility,
                EXISTS(SELECT 1 FROM shared_boards s WHERE s.board_id = b.board_id)
            FROM boards b
            WHERE b.board_id = ?;
            """,
            (board_id,),
        )
        row = cursor.fetchone()

        if row is None:
            self._log_refusal(project_id, f"board {board_id} no longer exists")
            return None

        board_user_id, board_visibility, is_shared = row

        if board_user_id != user_id:
            self._log_refusal(project_id, f"board {board_id} belongs to another account")
            return None
        # A project board is private by definition; adopting a shared or public one would lock a
        # board other people can see into a single project's lifecycle.
        if board_visibility != "private":
            self._log_refusal(project_id, f"board {board_id} is {board_visibility}, not private")
            return None
        if is_shared:
            self._log_refusal(project_id, f"board {board_id} is shared with other accounts")
            return None

        return board_id

    def _log_refusal(self, project_id: str, reason: str) -> None:
        self._logger.warning(
            f"Project boards migration: project {project_id} starts on a new empty board because {reason}."
            " Its previous board and everything on it are untouched."
        )

    def _insert_board(self, cursor: sqlite3.Cursor, *, user_id: str, name: str) -> str:
        board_id = uuid_string()
        cursor.execute(
            """--sql
            INSERT INTO boards (board_id, board_name, user_id, board_visibility, archived)
            VALUES (?, ?, ?, 'private', FALSE);
            """,
            (board_id, name[:BOARD_NAME_MAX_LENGTH], user_id),
        )
        return board_id

    def _rebuild_projects_table(self, cursor: sqlite3.Cursor, assignments: list[tuple[int, str]]) -> None:
        """Rebuild `projects` with `board_id`.

        SQLite cannot `ALTER TABLE ... ADD COLUMN` a `UNIQUE`, `NOT NULL` column with a foreign key,
        so the table is rebuilt. `projects` has no inbound foreign keys, so this is safe with
        `PRAGMA foreign_keys = ON` — which matters because the pragma is a no-op inside the
        transaction the migrator wraps this callback in and so cannot be turned off here.
        """
        # Dropped first so that renaming the replacement into place cannot rewrite a live trigger's
        # body out from under us.
        cursor.execute("DROP TRIGGER IF EXISTS tg_projects_updated_at;")

        # The migrator runs this whole callback in one transaction, so a half-finished rebuild is
        # not reachable through it. This is for the database that met a build predating that
        # guarantee: without it such a database fails forever on `table projects_with_boards
        # already exists`, and the scratch table is worthless in every other case anyway.
        cursor.execute("DROP TABLE IF EXISTS projects_with_boards;")

        cursor.execute(
            """--sql
            CREATE TABLE projects_with_boards (
                -- Client-generated identifier; unique per user, not globally.
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                -- Opaque client-owned project document (JSON).
                data TEXT NOT NULL,
                -- The project's private board. UNIQUE is what makes "at most one project per board"
                -- a schema fact; RESTRICT is what stops a claimed board being deleted through the
                -- generic board API. Spelled out rather than left to the default: SQLite's default
                -- is NO ACTION, which behaves identically only while the constraint is immediate.
                board_id TEXT NOT NULL UNIQUE,
                -- Incremented on every update; used for optimistic concurrency.
                revision INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (user_id, project_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (board_id) REFERENCES boards(board_id) ON DELETE RESTRICT
            );
            """
        )

        # Inserted in rowid order so the rebuilt table keeps the original insertion order, which is
        # what `list()` falls back to when two projects share a created_at millisecond.
        for rowid, board_id in assignments:
            cursor.execute(
                """--sql
                INSERT INTO projects_with_boards
                    (project_id, user_id, name, data, board_id, revision, created_at, updated_at)
                SELECT project_id, user_id, name, data, ?, revision, created_at, updated_at
                FROM projects
                WHERE rowid = ?;
                """,
                (board_id, rowid),
            )

        cursor.execute("DROP TABLE projects;")
        cursor.execute("ALTER TABLE projects_with_boards RENAME TO projects;")

        cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_projects_updated_at
            AFTER UPDATE ON projects
            FOR EACH ROW
            BEGIN
              UPDATE projects
                SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
              WHERE user_id = OLD.user_id AND project_id = OLD.project_id;
            END;
            """
        )


def _table_exists(cursor: sqlite3.Cursor, name: str) -> bool:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cursor.fetchone() is not None


def _collect_board_candidates(data: str) -> set[str]:
    """Every distinct non-empty `projectBoardId` a project document names.

    Two shapes are read: the current `widgetInstances` map, and the `widgetStates.gallery` shape
    earlier builds wrote. A document that is unparseable or shaped unexpectedly yields nothing,
    which lands the project on a fresh board.
    """
    try:
        document = json.loads(data)
    except (TypeError, ValueError):
        return set()

    if not isinstance(document, dict):
        return set()

    candidates: set[str] = set()

    widget_instances = document.get("widgetInstances")
    if isinstance(widget_instances, dict):
        for instance in widget_instances.values():
            if not isinstance(instance, dict) or instance.get("typeId") != "gallery":
                continue
            state = instance.get("state")
            if not isinstance(state, dict):
                continue
            _add_candidate(candidates, state.get("values"))

    widget_states = document.get("widgetStates")
    if isinstance(widget_states, dict):
        gallery = widget_states.get("gallery")
        if isinstance(gallery, dict):
            _add_candidate(candidates, gallery.get("values"))

    return candidates


def _add_candidate(candidates: set[str], values: Any) -> None:
    if not isinstance(values, dict):
        return
    board_id = values.get("projectBoardId")
    if isinstance(board_id, str) and board_id != "":
        candidates.add(board_id)


def build_migration(logger: Logger) -> Migration:
    """Build the migration that gives every project its own board.

    Takes the logger because this migration can decline to adopt a board, and a project that
    silently starts on an empty one looks to its owner like a project that lost its images.
    """
    return Migration(
        id="2026_08_06_add_project_boards",
        # The repair migration is what guarantees `projects` exists at all on databases that were
        # ever opened by an upstream build; it transitively guarantees `boards` and `shared_boards`.
        depends_on="2026_07_30_repair_projects_table",
        callback=AddProjectBoardsMigrationCallback(logger),
    )
