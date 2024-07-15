import sqlite3
import threading
from dataclasses import dataclass
from typing import Union, cast

from invokeai.app.services.board_records.board_records_base import BoardRecordStorageBase
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordDeleteException,
    BoardRecordNotFoundException,
    BoardRecordSaveException,
    UncategorizedImageCounts,
    deserialize_board_record,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.util.misc import uuid_string

# This query is missing a GROUP BY clause, which is required for the query to be valid.
BASE_UNTERMINATED_AND_MISSING_GROUP_BY_BOARD_RECORDS_QUERY = """
    SELECT b.board_id,
        b.board_name,
        b.created_at,
        b.updated_at,
        b.archived,
        COUNT(
            CASE
                WHEN i.image_category in ('general')
                AND i.is_intermediate = 0 THEN 1
            END
        ) AS image_count,
        COUNT(
            CASE
                WHEN i.image_category in ('control', 'mask', 'user', 'other')
                AND i.is_intermediate = 0 THEN 1
            END
        ) AS asset_count,
        (
            SELECT bi.image_name
            FROM board_images bi
                JOIN images i ON bi.image_name = i.image_name
            WHERE bi.board_id = b.board_id
                AND i.is_intermediate = 0
            ORDER BY i.created_at DESC
            LIMIT 1
        ) AS cover_image_name
    FROM boards b
        LEFT JOIN board_images bi ON b.board_id = bi.board_id
        LEFT JOIN images i ON bi.image_name = i.image_name
    """


@dataclass
class PaginatedBoardRecordsQueries:
    main_query: str
    total_count_query: str


def get_paginated_list_board_records_queries(include_archived: bool) -> PaginatedBoardRecordsQueries:
    """Gets a query to retrieve a paginated list of board records."""

    archived_condition = "WHERE b.archived = 0" if not include_archived else ""

    # The GROUP BY must be added _after_ the WHERE clause!
    main_query = f"""
        {BASE_UNTERMINATED_AND_MISSING_GROUP_BY_BOARD_RECORDS_QUERY}
        {archived_condition}
        GROUP BY b.board_id,
            b.board_name,
            b.created_at,
            b.updated_at
        ORDER BY b.created_at DESC
        LIMIT ? OFFSET ?;
        """

    total_count_query = f"""
        SELECT COUNT(*)
        FROM boards b
        {archived_condition};
        """

    return PaginatedBoardRecordsQueries(main_query=main_query, total_count_query=total_count_query)


def get_list_all_board_records_query(include_archived: bool) -> str:
    """Gets a query to retrieve all board records."""

    archived_condition = "WHERE b.archived = 0" if not include_archived else ""

    # The GROUP BY must be added _after_ the WHERE clause!
    return f"""
        {BASE_UNTERMINATED_AND_MISSING_GROUP_BY_BOARD_RECORDS_QUERY}
        {archived_condition}
        GROUP BY b.board_id,
            b.board_name,
            b.created_at,
            b.updated_at
        ORDER BY b.created_at DESC;
        """


def get_board_record_query() -> str:
    """Gets a query to retrieve a board record."""

    return f"""
        {BASE_UNTERMINATED_AND_MISSING_GROUP_BY_BOARD_RECORDS_QUERY}
        WHERE b.board_id = ?;
        """


class SqliteBoardRecordStorage(BoardRecordStorageBase):
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.RLock

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._lock = db.lock
        self._conn = db.conn
        self._cursor = self._conn.cursor()

    def delete(self, board_id: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM boards
                WHERE board_id = ?;
                """,
                (board_id,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordDeleteException from e
        except Exception as e:
            self._conn.rollback()
            raise BoardRecordDeleteException from e
        finally:
            self._lock.release()

    def save(
        self,
        board_name: str,
    ) -> BoardRecord:
        try:
            board_id = uuid_string()
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO boards (board_id, board_name)
                VALUES (?, ?);
                """,
                (board_id, board_name),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordSaveException from e
        finally:
            self._lock.release()
        return self.get(board_id)

    def get(
        self,
        board_id: str,
    ) -> BoardRecord:
        try:
            self._lock.acquire()
            self._cursor.execute(
                get_board_record_query(),
                (board_id,),
            )

            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordNotFoundException from e
        finally:
            self._lock.release()
        if result is None:
            raise BoardRecordNotFoundException
        return deserialize_board_record(dict(result))

    def update(
        self,
        board_id: str,
        changes: BoardChanges,
    ) -> BoardRecord:
        try:
            self._lock.acquire()

            # Change the name of a board
            if changes.board_name is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE boards
                    SET board_name = ?
                    WHERE board_id = ?;
                    """,
                    (changes.board_name, board_id),
                )

            # Change the cover image of a board
            if changes.cover_image_name is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE boards
                    SET cover_image_name = ?
                    WHERE board_id = ?;
                    """,
                    (changes.cover_image_name, board_id),
                )

            # Change the archived status of a board
            if changes.archived is not None:
                self._cursor.execute(
                    """--sql
                    UPDATE boards
                    SET archived = ?
                    WHERE board_id = ?;
                    """,
                    (changes.archived, board_id),
                )

            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordSaveException from e
        finally:
            self._lock.release()
        return self.get(board_id)

    def get_many(
        self, offset: int = 0, limit: int = 10, include_archived: bool = False
    ) -> OffsetPaginatedResults[BoardRecord]:
        try:
            self._lock.acquire()

            queries = get_paginated_list_board_records_queries(include_archived=include_archived)

            self._cursor.execute(
                queries.main_query,
                (limit, offset),
            )

            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            boards = [deserialize_board_record(dict(r)) for r in result]

            self._cursor.execute(queries.total_count_query)
            count = cast(int, self._cursor.fetchone()[0])

            return OffsetPaginatedResults[BoardRecord](items=boards, offset=offset, limit=limit, total=count)

        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

    def get_all(self, include_archived: bool = False) -> list[BoardRecord]:
        try:
            self._lock.acquire()
            self._cursor.execute(get_list_all_board_records_query(include_archived=include_archived))
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            boards = [deserialize_board_record(dict(r)) for r in result]
            return boards

        except sqlite3.Error as e:
            self._conn.rollback()
            raise e
        finally:
            self._lock.release()

    def get_uncategorized_image_counts(self) -> UncategorizedImageCounts:
        try:
            self._lock.acquire()
            query = """
                SELECT
                    CASE
                        WHEN i.image_category = 'general' THEN 'images'
                        ELSE 'assets'
                    END AS category_type,
                    COUNT(*) AS unassigned_count
                FROM images i
                LEFT JOIN board_images bi ON i.image_name = bi.image_name
                WHERE i.image_category IN ('general', 'control', 'mask', 'user', 'other')
                AND bi.board_id IS NULL
                AND i.is_intermediate = 0
                GROUP BY category_type;
                """
            self._cursor.execute(query)
            results = self._cursor.fetchall()
            image_count = results[0][1]
            asset_count = results[1][1]
            return UncategorizedImageCounts(image_count=image_count, asset_count=asset_count)
        finally:
            self._lock.release()
