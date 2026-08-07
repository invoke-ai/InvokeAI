import json
import sqlite3
import uuid
from typing import Any, Optional

from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.project_records.project_records_base import ProjectRecordsStorageBase
from invokeai.app.services.project_records.project_records_common import (
    ProjectBoardItemDTO,
    ProjectBoardNotFoundError,
    ProjectBoardSnapshotDTO,
    ProjectBoardUnavailableError,
    ProjectRecordConflictError,
    ProjectRecordDTO,
    ProjectRecordExistsError,
    ProjectRecordNotFoundError,
    ProjectSummaryDTO,
)
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.util.misc import uuid_string

# `BoardChanges.board_name` caps names at 300 characters. Project names are unbounded, so the board
# carrying a project's name is truncated to stay inside what the generic board API would accept.
BOARD_NAME_MAX_LENGTH = 300


class ProjectRecordsSqlite(ProjectRecordsStorageBase):
    """SQLite implementation of per-user project document storage.

    A project and its board are one unit here. Every write that touches both — creating them,
    renaming them, deleting them — happens inside a single `self._db.transaction()` so the two can
    never disagree.

    That transaction is the reason this class talks to `boards` in raw SQL rather than through
    `BoardService`. `SqliteDatabase.transaction()` guards a re-entrant lock and commits on the way
    out of *every* level, so calling another service's method from inside an open transaction would
    commit this one's half-finished work. Nothing between the `with` and its close may call anything
    that opens a transaction of its own, including `self.get()`.
    """

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def create(
        self,
        user_id: str,
        name: str,
        data: dict[str, Any],
        project_id: str | None = None,
        board_id: str | None = None,
    ) -> ProjectRecordDTO:
        project_id = project_id or uuid.uuid4().hex

        try:
            with self._db.transaction() as cursor:
                if board_id is None:
                    resolved_board_id = self._insert_board(cursor, user_id=user_id, name=name)
                else:
                    self._claim_board(cursor, user_id=user_id, board_id=board_id, name=name)
                    resolved_board_id = board_id

                cursor.execute(
                    """--sql
                    INSERT INTO projects (project_id, user_id, name, data, board_id)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (project_id, user_id, name, json.dumps(data), resolved_board_id),
                )
        except sqlite3.IntegrityError as e:
            # The board insert/rename rolls back with the project insert, so a rejected create never
            # leaves an orphan board behind.
            raise self._classify_integrity_error(e, project_id=project_id, board_id=board_id) from e

        return self.get(user_id, project_id)

    def get(self, user_id: str, project_id: str) -> ProjectRecordDTO:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT project_id, board_id, name, data, revision, created_at, updated_at
                FROM projects
                WHERE user_id = ? AND project_id = ?;
                """,
                (user_id, project_id),
            )
            row = cursor.fetchone()

        if row is None:
            raise ProjectRecordNotFoundError(project_id)

        return ProjectRecordDTO(
            project_id=row[0],
            board_id=row[1],
            name=row[2],
            data=json.loads(row[3]),
            revision=row[4],
            created_at=row[5],
            updated_at=row[6],
        )

    def list(self, user_id: str) -> list[ProjectSummaryDTO]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT project_id, board_id, name, revision, created_at, updated_at
                FROM projects
                WHERE user_id = ?
                -- rowid breaks ties between rows created in the same millisecond,
                -- keeping the listing in true insertion order.
                ORDER BY created_at ASC, rowid ASC;
                """,
                (user_id,),
            )
            rows = cursor.fetchall()

        return [
            ProjectSummaryDTO(
                project_id=row[0],
                board_id=row[1],
                name=row[2],
                revision=row[3],
                created_at=row[4],
                updated_at=row[5],
            )
            for row in rows
        ]

    def update(
        self, user_id: str, project_id: str, expected_revision: int, name: str, data: dict[str, Any]
    ) -> ProjectRecordDTO:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                UPDATE projects
                SET name = ?, data = ?, revision = revision + 1
                WHERE user_id = ? AND project_id = ? AND revision = ?;
                """,
                (name, json.dumps(data), user_id, project_id, expected_revision),
            )

            if cursor.rowcount == 0:
                # Distinguish "gone" from "someone else saved first".
                cursor.execute(
                    """--sql
                    SELECT revision FROM projects
                    WHERE user_id = ? AND project_id = ?;
                    """,
                    (user_id, project_id),
                )
                row = cursor.fetchone()

                if row is None:
                    raise ProjectRecordNotFoundError(project_id)

                # Raised before the rename, so a save that lost the race renames nothing.
                raise ProjectRecordConflictError(project_id, expected_revision, row[0])

            cursor.execute(
                """--sql
                UPDATE boards
                SET board_name = ?
                WHERE board_id = (SELECT board_id FROM projects WHERE user_id = ? AND project_id = ?);
                """,
                (name[:BOARD_NAME_MAX_LENGTH], user_id, project_id),
            )

        return self.get(user_id, project_id)

    def delete(self, user_id: str, project_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT board_id FROM projects WHERE user_id = ? AND project_id = ?;
                """,
                (user_id, project_id),
            )
            row = cursor.fetchone()

            if row is None:
                return

            cursor.execute(
                """--sql
                DELETE FROM projects
                WHERE user_id = ? AND project_id = ?;
                """,
                (user_id, project_id),
            )
            # Only after the project is gone: the board FK is RESTRICT, so a claimed board cannot be
            # deleted. Deleting the board cascades its memberships, returning the media to
            # Uncategorized; the image and video records and their files are untouched.
            cursor.execute(
                """--sql
                DELETE FROM boards WHERE board_id = ?;
                """,
                (row[0],),
            )

    def get_board_snapshot(self, user_id: str, project_id: str) -> ProjectBoardSnapshotDTO:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT board_id FROM projects WHERE user_id = ? AND project_id = ?;
                """,
                (user_id, project_id),
            )
            row = cursor.fetchone()

            if row is None:
                raise ProjectRecordNotFoundError(project_id)

            # One union over both namespaces, shaped like `get_board_media_summaries`. The filters
            # are what make the result match what the gallery shows for this board: intermediates
            # are hidden, and `other` — the canvas's private category — is in neither
            # IMAGE_CATEGORIES nor ASSETS_CATEGORIES and so appears in no board view.
            #
            # `deleted_at` is deliberately *not* filtered. Soft delete is unused and no other
            # board query consults it, so filtering here would make the snapshot disagree with the
            # counts shown beside the board.
            cursor.execute(
                """--sql
                SELECT 'image' AS kind, images.image_name AS name,
                       images.image_category AS category, images.starred AS starred
                FROM board_images
                INNER JOIN images ON board_images.image_name = images.image_name
                WHERE board_images.board_id = ?
                  AND images.is_intermediate = FALSE
                  AND images.image_category IN ('general', 'control', 'mask', 'user')
                UNION ALL
                SELECT 'video' AS kind, videos.video_name AS name,
                       videos.video_category AS category, videos.starred AS starred
                FROM board_videos
                INNER JOIN videos ON board_videos.video_name = videos.video_name
                WHERE board_videos.board_id = ?
                  AND videos.is_intermediate = FALSE
                  AND videos.video_category IN ('general', 'control', 'mask', 'user')
                ORDER BY kind ASC, name ASC;
                """,
                (row[0], row[0]),
            )
            rows = cursor.fetchall()

        return ProjectBoardSnapshotDTO(
            items=[ProjectBoardItemDTO(kind=r[0], name=r[1], category=r[2], starred=bool(r[3])) for r in rows]
        )

    def get_board_id(self, user_id: str, project_id: str) -> str:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT board_id FROM projects WHERE user_id = ? AND project_id = ?;
                """,
                (user_id, project_id),
            )
            row = cursor.fetchone()

        if row is None:
            raise ProjectRecordNotFoundError(project_id)

        return row[0]

    def _insert_board(self, cursor: sqlite3.Cursor, *, user_id: str, name: str) -> str:
        board_id = uuid_string()
        cursor.execute(
            """--sql
            INSERT INTO boards (board_id, board_name, user_id, board_visibility, archived)
            VALUES (?, ?, ?, ?, FALSE);
            """,
            (board_id, name[:BOARD_NAME_MAX_LENGTH], user_id, BoardVisibility.Private.value),
        )
        return board_id

    def _claim_board(self, cursor: sqlite3.Cursor, *, user_id: str, board_id: str, name: str) -> None:
        """Take ownership of an existing private board, or explain why it cannot be taken.

        One query answers every way a claim can fail. The `UNIQUE` constraint on `projects.board_id`
        settles the race this check cannot see: two imports claiming the same board both pass here,
        and exactly one of them survives the insert.
        """
        cursor.execute(
            """--sql
            SELECT
                b.user_id,
                b.board_visibility,
                EXISTS(SELECT 1 FROM shared_boards s WHERE s.board_id = b.board_id),
                EXISTS(SELECT 1 FROM projects p WHERE p.board_id = b.board_id)
            FROM boards b
            WHERE b.board_id = ?;
            """,
            (board_id,),
        )
        row = cursor.fetchone()

        if row is None or row[0] != user_id:
            raise ProjectBoardNotFoundError(board_id)

        if row[1] != BoardVisibility.Private.value or row[2] or row[3]:
            raise ProjectBoardUnavailableError(board_id)

        # Un-archived as well as renamed. A project's board takes its archived state from the
        # project, and `PATCH /boards/{id}` refuses to set it on a claimed board — so a board that
        # arrived archived would be invisible in every listing with no API left to fix it.
        cursor.execute(
            """--sql
            UPDATE boards SET board_name = ?, archived = FALSE WHERE board_id = ?;
            """,
            (name[:BOARD_NAME_MAX_LENGTH], board_id),
        )

    @staticmethod
    def _classify_integrity_error(
        error: sqlite3.IntegrityError, *, project_id: str, board_id: Optional[str]
    ) -> Exception:
        """Name which uniqueness a rejected insert violated.

        Two constraints can reject the same statement: the composite primary key, meaning the user
        already has this project id, and `projects.board_id`, meaning another project claimed the
        board after `_claim_board` looked.
        """
        if board_id is not None and "projects.board_id" in str(error):
            return ProjectBoardUnavailableError(board_id)
        return ProjectRecordExistsError(project_id)
