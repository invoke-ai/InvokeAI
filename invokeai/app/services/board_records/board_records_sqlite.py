import sqlite3
import threading
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

_BASE_BOARD_RECORD_QUERY = """
    -- This query retrieves board records, joining with the board_images and images tables to get image counts and cover image names.
    -- It is not a complete query, as it is missing a GROUP BY or WHERE clause (and is unterminated).
    SELECT b.board_id,
        b.board_name,
        b.created_at,
        b.updated_at,
        b.archived,
        -- Count the number of images in the board, alias image_count
        COUNT(
            CASE
                WHEN i.image_category in ('general') -- "Images" are images in the 'general' category
                AND i.is_intermediate = 0 THEN 1 -- Intermediates are not counted
            END
        ) AS image_count,
        -- Count the number of assets in the board, alias asset_count
        COUNT(
            CASE
                WHEN i.image_category in ('control', 'mask', 'user', 'other') -- "Assets" are images in any of the other categories ('control', 'mask', 'user', 'other')
                AND i.is_intermediate = 0 THEN 1 -- Intermediates are not counted
            END
        ) AS asset_count,
        -- Get the name of the the most recent image in the board, alias cover_image_name
        (
            SELECT bi.image_name
            FROM board_images bi
                JOIN images i ON bi.image_name = i.image_name
            WHERE bi.board_id = b.board_id
                AND i.is_intermediate = 0 -- Intermediates cannot be cover images
            ORDER BY i.created_at DESC -- Sort by created_at to get the most recent image
            LIMIT 1
        ) AS cover_image_name
    FROM boards b
        LEFT JOIN board_images bi ON b.board_id = bi.board_id
        LEFT JOIN images i ON bi.image_name = i.image_name
    """


def get_paginated_list_board_records_queries(include_archived: bool) -> str:
    """Gets a query to retrieve a paginated list of board records. The query has placeholders for limit and offset.

    Args:
        include_archived: Whether to include archived board records in the results.

    Returns:
        A query to retrieve a paginated list of board records.
    """

    archived_condition = "WHERE b.archived = 0" if not include_archived else ""

    # The GROUP BY must be added _after_ the WHERE clause!
    query = f"""
        {_BASE_BOARD_RECORD_QUERY}
        {archived_condition}
        GROUP BY b.board_id,
            b.board_name,
            b.created_at,
            b.updated_at
        ORDER BY b.created_at DESC
        LIMIT ? OFFSET ?;
        """

    return query


def get_total_boards_count_query(include_archived: bool) -> str:
    """Gets a query to retrieve the total count of board records.

    Args:
        include_archived: Whether to include archived board records in the count.

    Returns:
        A query to retrieve the total count of board records.
    """

    archived_condition = "WHERE b.archived = 0" if not include_archived else ""

    return f"SELECT COUNT(*) FROM boards {archived_condition};"


def get_list_all_board_records_query(include_archived: bool) -> str:
    """Gets a query to retrieve all board records.

    Args:
        include_archived: Whether to include archived board records in the results.

    Returns:
        A query to retrieve all board records.
    """

    archived_condition = "WHERE b.archived = 0" if not include_archived else ""

    return f"""
        {_BASE_BOARD_RECORD_QUERY}
        {archived_condition}
        GROUP BY b.board_id,
            b.board_name,
            b.created_at,
            b.updated_at
        ORDER BY b.created_at DESC;
        """


def get_board_record_query() -> str:
    """Gets a query to retrieve a board record. The query has a placeholder for the board_id."""

    return f"{_BASE_BOARD_RECORD_QUERY} WHERE b.board_id = ?;"


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

            main_query = get_paginated_list_board_records_queries(include_archived=include_archived)

            self._cursor.execute(main_query, (limit, offset))

            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            boards = [deserialize_board_record(dict(r)) for r in result]

            total_query = get_total_boards_count_query(include_archived=include_archived)
            self._cursor.execute(total_query)
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
            query = get_list_all_board_records_query(include_archived=include_archived)
            self._cursor.execute(query)
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
                -- Get the count of uncategorized images and assets.
                SELECT
                    CASE
                        WHEN i.image_category = 'general' THEN 'image_count' -- "Images" are images in the 'general' category
                        ELSE 'asset_count' -- "Assets" are images in any of the other categories ('control', 'mask', 'user', 'other')
                    END AS category_type,
                    COUNT(*) AS unassigned_count
                FROM images i
                LEFT JOIN board_images bi ON i.image_name = bi.image_name
                WHERE bi.board_id IS NULL -- Uncategorized images have no board association
                AND i.is_intermediate = 0 -- Omit intermediates from the counts
                GROUP BY category_type; -- Group by category_type alias, as derived from the image_category column earlier
                """
            self._cursor.execute(query)
            results = self._cursor.fetchall()
            image_count = dict(results)["image_count"]
            asset_count = dict(results)["asset_count"]
            return UncategorizedImageCounts(image_count=image_count, asset_count=asset_count)
        finally:
            self._lock.release()

    def get_uncategorized_image_names(self) -> list[str]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT image_name
                FROM images
                WHERE image_name NOT IN (
                    SELECT image_name
                    FROM board_images
                );
                """
            )
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            image_names = [r[0] for r in result]
            return image_names
        finally:
            self._lock.release()
