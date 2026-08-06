"""Give every project a first-class board that SQLite owns.

Until now a "project board" was a fiction the client maintained: the Workbench created an ordinary
board on demand and stashed its id in the opaque project document at
`widgetInstances.<id>.state.values.projectBoardId`. Nothing on the server knew the board belonged to
a project, so the board could be renamed, archived or deleted out from under it, two projects could
name the same board, and a project restored on another install pointed at a board id that did not
exist there.

This migration moves the relationship into the schema: `projects.board_id` becomes `NOT NULL UNIQUE`
with a foreign key to `boards`. The `UNIQUE` makes "a board belongs to at most one project" a
database fact, and the FK's default `RESTRICT` makes "a claimed board cannot be deleted" one too,
backing the application's 409 rather than merely agreeing with it.

Existing projects keep the board they were already using **only when that is unambiguously safe**.
The document is opaque and client-written, so its `projectBoardId` is treated as a hint to be
verified, never as an assertion: a candidate is adopted only when exactly one distinct value survives
every ownership test. Anything ambiguous — two candidates, a board owned by someone else, a shared or
public board, one another project already took — gets a fresh empty board instead. That is the
conservative direction: a project that adopts nothing merely starts with an empty board, whereas a
project that adopts the wrong board takes ownership of media it does not own.

Nothing is moved between boards. `board_images` and `board_videos` are read but never written, so no
image or video changes hands here; only the projects table gains a column, and boards may gain rows
and a new name.
"""

import json
import sqlite3
from typing import Any, Optional

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.app.util.misc import uuid_string

# `BoardChanges.board_name` caps names at 300 characters. Project names are unbounded, so a board
# adopting a project's name is truncated to stay inside what the generic board API would accept.
BOARD_NAME_MAX_LENGTH = 300


class AddProjectBoardsMigrationCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        assignments = self._assign_boards(cursor)
        self._rebuild_projects_table(cursor, assignments)

    def _assign_boards(self, cursor: sqlite3.Cursor) -> list[tuple[int, str]]:
        """Resolve one board per project, creating boards as needed.

        Returns (rowid, board_id) pairs. Projects are visited in `rowid` order so that when two of
        them name the same board the earlier one wins, deterministically and repeatably.
        """
        cursor.execute(
            """--sql
            SELECT rowid, user_id, name, data FROM projects ORDER BY rowid ASC;
            """
        )
        projects = cursor.fetchall()

        assignments: list[tuple[int, str]] = []
        # Boards claimed earlier in this run. The column does not exist yet, so this set is the only
        # record of what has already been spoken for.
        claimed: set[str] = set()

        for rowid, user_id, name, data in projects:
            board_id = self._adopt_board(cursor, user_id=user_id, data=data, claimed=claimed)

            if board_id is None:
                board_id = self._insert_board(cursor, user_id=user_id, name=name)
            else:
                cursor.execute(
                    """--sql
                    UPDATE boards SET board_name = ? WHERE board_id = ?;
                    """,
                    (name[:BOARD_NAME_MAX_LENGTH], board_id),
                )

            claimed.add(board_id)
            assignments.append((rowid, board_id))

        return assignments

    def _adopt_board(
        self, cursor: sqlite3.Cursor, *, user_id: str, data: str, claimed: set[str]
    ) -> Optional[str]:
        """Return the project's existing board if exactly one candidate is safe to adopt."""
        candidates = _collect_board_candidates(data)

        # Two gallery widgets disagreeing about which board is the project's leaves no way to pick
        # correctly, and guessing would hand a project someone else's media.
        if len(candidates) != 1:
            return None

        board_id = next(iter(candidates))

        if board_id in claimed:
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
            return None

        board_user_id, board_visibility, is_shared = row

        if board_user_id != user_id:
            return None
        # A project board is private by definition; adopting a shared or public one would lock a
        # board other people can see into a single project's lifecycle.
        if board_visibility != "private":
            return None
        if is_shared:
            return None

        return board_id

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
                -- a schema fact; the FK's default RESTRICT is what stops a claimed board being
                -- deleted through the generic board API.
                board_id TEXT NOT NULL UNIQUE,
                -- Incremented on every update; used for optimistic concurrency.
                revision INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                PRIMARY KEY (user_id, project_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (board_id) REFERENCES boards(board_id)
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


def build_migration() -> Migration:
    """Build the migration that gives every project its own board."""
    return Migration(
        id="2026_08_06_add_project_boards",
        # The repair migration is what guarantees `projects` exists at all on databases that were
        # ever opened by an upstream build; it transitively guarantees `boards` and `shared_boards`.
        depends_on="2026_07_30_repair_projects_table",
        callback=AddProjectBoardsMigrationCallback(),
    )
