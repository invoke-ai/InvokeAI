from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, Optional, TypeVar, cast
import sqlite3
import threading
from typing import Optional, Union
import uuid
from invokeai.app.services.image_record_storage import OffsetPaginatedResults

from pydantic import BaseModel, Field, Extra
from pydantic.generics import GenericModel

T = TypeVar("T", bound=BaseModel)

class BoardRecord(BaseModel):
    """Deserialized board record."""

    id: str = Field(description="The unique ID of the board.")
    name: str = Field(description="The name of the board.")
    """The name of the board."""
    created_at: Union[datetime, str] = Field(
        description="The created timestamp of the board."
    )
    """The created timestamp of the image."""
    updated_at: Union[datetime, str] = Field(
        description="The updated timestamp of the board."
    )

class BoardRecordInList(BaseModel):
    """Deserialized board record in a list."""

    id: str = Field(description="The unique ID of the board.")
    name: str = Field(description="The name of the board.")
    most_recent_image_url: Optional[str] = Field(
        description="The URL of the most recent image in the board."
    )
    """The name of the board."""
    created_at: Union[datetime, str] = Field(
        description="The created timestamp of the board."
    )
    """The created timestamp of the image."""
    updated_at: Union[datetime, str] = Field(
        description="The updated timestamp of the board."
    )

class BoardRecordChanges(BaseModel, extra=Extra.forbid):
    name: Optional[str] = Field(
        description="The board's new name."
    )

class BoardRecordNotFoundException(Exception):
    """Raised when an board record is not found."""

    def __init__(self, message="Board record not found"):
        super().__init__(message)


class BoardRecordSaveException(Exception):
    """Raised when an board record cannot be saved."""

    def __init__(self, message="Board record not saved"):
        super().__init__(message)


class BoardRecordDeleteException(Exception):
    """Raised when an board record cannot be deleted."""

    def __init__(self, message="Board record not deleted"):
        super().__init__(message)

class BoardStorageBase(ABC):
    """Low-level service responsible for interfacing with the board record store."""

    @abstractmethod
    def delete(self, board_id: str) -> None:
        """Deletes a board record."""
        pass

    @abstractmethod
    def save(
        self,
        board_name: str,
    ):
        """Saves a board record."""
        pass

    def get_cover_photo(self, board_id: str) -> Optional[str]:
        """Gets the cover photo for a board."""
        pass

    def get_many(
        self,
        offset: int,
        limit: int,
    ):
        """Gets many board records."""
        pass


class SqliteBoardStorage(BoardStorageBase):
    _filename: str
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.Lock

    def __init__(self, filename: str) -> None:
        super().__init__()
        self._filename = filename
        self._conn = sqlite3.connect(filename, check_same_thread=False)
        # Enable row factory to get rows as dictionaries (must be done before making the cursor!)
        self._conn.row_factory = sqlite3.Row
        self._cursor = self._conn.cursor()
        self._lock = threading.Lock()

        try:
            self._lock.acquire()
            # Enable foreign keys
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._create_tables()
            self._conn.commit()
        finally:
            self._lock.release()

    def _create_tables(self) -> None:
        """Creates the `board` table."""

        # Create the `images` table.
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS boards (
                id TEXT NOT NULL PRIMARY KEY,
                name TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))
            );
            """
        )

        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_boards_created_at ON boards(created_at);
            """
        )

        # Add trigger for `updated_at`.
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_boards_updated_at
            AFTER UPDATE
            ON boards FOR EACH ROW
            BEGIN
                UPDATE boards SET updated_at = current_timestamp
                    WHERE board_name = old.board_name;
            END;
            """
        )


    def delete(self, board_id: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM boards
                WHERE id = ?;
                """,
                (board_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordDeleteException from e
        finally:
            self._lock.release()

    def save(
        self,
        board_name: str,
    ):
        try:
            board_id = str(uuid.uuid4())
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO boards (id, name)
                VALUES (?, ?);
                """,
                (board_id, board_name),
            )
            self._conn.commit()

            self._cursor.execute(
                """--sql
                SELECT *
                FROM boards
                WHERE id = ?;
                """,
                (board_id,),
            )

            result = self._cursor.fetchone()
            return result
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordSaveException from e
        finally:
            self._lock.release()
    

    def get_many(
        self,
        offset: int,
        limit: int,
    ) -> OffsetPaginatedResults[BoardRecord]:
        try:

            self._lock.acquire()

            count_query = f"""SELECT COUNT(*) FROM images WHERE 1=1\n"""
            images_query = f"""SELECT * FROM images WHERE 1=1\n"""

            query_conditions = ""
            query_params = []

            query_pagination = f"""ORDER BY created_at DESC LIMIT ? OFFSET ?\n"""

            # Final images query with pagination
            images_query += query_conditions + query_pagination + ";"
            # Add all the parameters
            images_params = query_params.copy()
            images_params.append(limit)
            images_params.append(offset)
            # Build the list of images, deserializing each row
            self._cursor.execute(images_query, images_params)
            result = cast(list[sqlite3.Row], self._cursor.fetchall())
            boards = [BoardRecord(**dict(row)) for row in result]

            # Set up and execute the count query, without pagination
            count_query += query_conditions + ";"
            count_params = query_params.copy()
            self._cursor.execute(count_query, count_params)
            count = self._cursor.fetchone()[0]

        except sqlite3.Error as e:
            self._conn.rollback()
            raise BoardRecordSaveException from e
        finally:
            self._lock.release()
        
        return OffsetPaginatedResults(
            items=boards, offset=offset, limit=limit, total=count
        )