import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union, cast

from pydantic import BaseModel, Extra, Field, StrictFloat, StrictInt, StrictStr, parse_raw_as, validator

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.services.graph import Graph

BatchDataType = Union[StrictStr, StrictInt, StrictFloat, ImageField]


class BatchData(BaseModel):
    """
    A batch data collection.
    """

    node_id: str = Field(description="The node into which this batch data collection will be substituted.")
    field_name: str = Field(description="The field into which this batch data collection will be substituted.")
    items: list[BatchDataType] = Field(
        default_factory=list, description="The list of items to substitute into the node/field."
    )


class Batch(BaseModel):
    """
    A batch, consisting of a list of a list of batch data collections.

    First, each inner list[BatchData] is zipped into a single batch data collection.

    Then, the final batch collection is created by taking the Cartesian product of all batch data collections.
    """

    data: list[list[BatchData]] = Field(default_factory=list, description="The list of batch data collections.")

    @validator("data")
    def validate_len(cls, v: list[list[BatchData]]):
        for batch_data in v:
            if any(len(batch_data[0].items) != len(i.items) for i in batch_data):
                raise ValueError("Zipped batch items must have all have same length")
        return v

    @validator("data")
    def validate_types(cls, v: list[list[BatchData]]):
        for batch_data in v:
            for datum in batch_data:
                for item in datum.items:
                    if not all(isinstance(item, type(i)) for i in datum.items):
                        raise TypeError("All items in a batch must have have same type")
        return v

    @validator("data")
    def validate_unique_field_mappings(cls, v: list[list[BatchData]]):
        paths: set[tuple[str, str]] = set()
        count: int = 0
        for batch_data in v:
            for datum in batch_data:
                paths.add((datum.node_id, datum.field_name))
                count += 1
        if len(paths) != count:
            raise ValueError("Each batch data must have unique node_id and field_name")
        return v


class BatchSession(BaseModel):
    batch_id: str = Field(description="The Batch to which this BatchSession is attached.")
    session_id: str = Field(description="The Session to which this BatchSession is attached.")
    state: Literal["created", "completed", "inprogress", "error"] = Field(description="The state of this BatchSession")


def uuid_string():
    res = uuid.uuid4()
    return str(res)


class BatchProcess(BaseModel):
    batch_id: str = Field(default_factory=uuid_string, description="Identifier for this batch.")
    batch: Batch = Field(description="The Batch to apply to this session.")
    canceled: bool = Field(description="Whether or not to run sessions from this batch.", default=False)
    graph: Graph = Field(description="The graph into which batch data will be inserted before being executed.")


class BatchSessionChanges(BaseModel, extra=Extra.forbid):
    state: Literal["created", "completed", "inprogress", "error"] = Field(description="The state of this BatchSession")


class BatchProcessNotFoundException(Exception):
    """Raised when an Batch Process record is not found."""

    def __init__(self, message="BatchProcess record not found"):
        super().__init__(message)


class BatchProcessSaveException(Exception):
    """Raised when an Batch Process record cannot be saved."""

    def __init__(self, message="BatchProcess record not saved"):
        super().__init__(message)


class BatchProcessDeleteException(Exception):
    """Raised when an Batch Process record cannot be deleted."""

    def __init__(self, message="BatchProcess record not deleted"):
        super().__init__(message)


class BatchSessionNotFoundException(Exception):
    """Raised when an Batch Session record is not found."""

    def __init__(self, message="BatchSession record not found"):
        super().__init__(message)


class BatchSessionSaveException(Exception):
    """Raised when an Batch Session record cannot be saved."""

    def __init__(self, message="BatchSession record not saved"):
        super().__init__(message)


class BatchSessionDeleteException(Exception):
    """Raised when an Batch Session record cannot be deleted."""

    def __init__(self, message="BatchSession record not deleted"):
        super().__init__(message)


class BatchProcessStorageBase(ABC):
    """Low-level service responsible for interfacing with the Batch Process record store."""

    @abstractmethod
    def delete(self, batch_id: str) -> None:
        """Deletes a BatchProcess record."""
        pass

    @abstractmethod
    def save(
        self,
        batch_process: BatchProcess,
    ) -> BatchProcess:
        """Saves a BatchProcess record."""
        pass

    @abstractmethod
    def get(
        self,
        batch_id: str,
    ) -> BatchProcess:
        """Gets a BatchProcess record."""
        pass

    @abstractmethod
    def start(
        self,
        batch_id: str,
    ):
        """Starts a BatchProcess record by marking its `canceled` attribute to False."""
        pass

    @abstractmethod
    def cancel(
        self,
        batch_id: str,
    ):
        """Cancel BatchProcess record by setting its `canceled` attribute to True."""
        pass

    @abstractmethod
    def create_session(
        self,
        session: BatchSession,
    ) -> BatchSession:
        """Creates a BatchSession attached to a BatchProcess."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> BatchSession:
        """Gets a BatchSession by session_id"""
        pass

    @abstractmethod
    def get_created_session(self, batch_id: str) -> BatchSession:
        """Gets the latest BatchSession with state `created`, for a given BatchProcess id."""
        pass

    @abstractmethod
    def get_created_sessions(self, batch_id: str) -> List[BatchSession]:
        """Gets all BatchSession's with state `created`, for a given BatchProcess id."""
        pass

    @abstractmethod
    def update_session_state(
        self,
        batch_id: str,
        session_id: str,
        changes: BatchSessionChanges,
    ) -> BatchSession:
        """Updates the state of a BatchSession record."""
        pass


class SqliteBatchProcessStorage(BatchProcessStorageBase):
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: threading.Lock

    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__()
        self._conn = conn
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
        """Creates the `batch_process` table and `batch_session` junction table."""

        # Create the `batch_process` table.
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS batch_process (
                batch_id TEXT NOT NULL PRIMARY KEY,
                batches TEXT NOT NULL,
                graph TEXT NOT NULL,
                canceled BOOLEAN NOT NULL DEFAULT(0),
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME
            );
            """
        )

        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_batch_process_created_at ON batch_process (created_at);
            """
        )

        # Add trigger for `updated_at`.
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_batch_process_updated_at
            AFTER UPDATE
            ON batch_process FOR EACH ROW
            BEGIN
                UPDATE batch_process SET updated_at = current_timestamp
                    WHERE batch_id = old.batch_id;
            END;
            """
        )

        # Create the `batch_session` junction table.
        self._cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS batch_session (
                batch_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- updated via trigger
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                -- Soft delete, currently unused
                deleted_at DATETIME,
                -- enforce one-to-many relationship between batch_process and batch_session using PK
                -- (we can extend this to many-to-many later)
                PRIMARY KEY (batch_id,session_id),
                FOREIGN KEY (batch_id) REFERENCES batch_process (batch_id) ON DELETE CASCADE
            );
            """
        )

        # Add index for batch id
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_batch_session_batch_id ON batch_session (batch_id);
            """
        )

        # Add index for batch id, sorted by created_at
        self._cursor.execute(
            """--sql
            CREATE INDEX IF NOT EXISTS idx_batch_session_batch_id_created_at ON batch_session (batch_id,created_at);
            """
        )

        # Add trigger for `updated_at`.
        self._cursor.execute(
            """--sql
            CREATE TRIGGER IF NOT EXISTS tg_batch_session_updated_at
            AFTER UPDATE
            ON batch_session FOR EACH ROW
            BEGIN
                UPDATE batch_session SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                    WHERE batch_id = old.batch_id AND session_id = old.session_id;
            END;
            """
        )

    def delete(self, batch_id: str) -> None:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                DELETE FROM batch_process
                WHERE batch_id = ?;
                """,
                (batch_id,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchProcessDeleteException from e
        except Exception as e:
            self._conn.rollback()
            raise BatchProcessDeleteException from e
        finally:
            self._lock.release()

    def save(
        self,
        batch_process: BatchProcess,
    ) -> BatchProcess:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO batch_process (batch_id, batches, graph)
                VALUES (?, ?, ?);
                """,
                (batch_process.batch_id, batch_process.batch.json(), batch_process.graph.json()),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchProcessSaveException from e
        finally:
            self._lock.release()
        return self.get(batch_process.batch_id)

    def _deserialize_batch_process(self, session_dict: dict) -> BatchProcess:
        """Deserializes a batch session."""

        # Retrieve all the values, setting "reasonable" defaults if they are not present.

        batch_id = session_dict.get("batch_id", "unknown")
        batch_raw = session_dict.get("batches", "unknown")
        graph_raw = session_dict.get("graph", "unknown")
        canceled = session_dict.get("canceled", 0)
        return BatchProcess(
            batch_id=batch_id,
            batch=parse_raw_as(Batch, batch_raw),
            graph=parse_raw_as(Graph, graph_raw),
            canceled=canceled == 1,
        )

    def get(
        self,
        batch_id: str,
    ) -> BatchProcess:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT *
                FROM batch_process
                WHERE batch_id = ?;
                """,
                (batch_id,),
            )

            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchProcessNotFoundException from e
        finally:
            self._lock.release()
        if result is None:
            raise BatchProcessNotFoundException
        return self._deserialize_batch_process(dict(result))

    def start(
        self,
        batch_id: str,
    ):
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""--sql
                UPDATE batch_process
                SET canceled = 0
                WHERE batch_id = ?;
                """,
                (batch_id,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchSessionSaveException from e
        finally:
            self._lock.release()

    def cancel(
        self,
        batch_id: str,
    ):
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""--sql
                UPDATE batch_process
                SET canceled = 1
                WHERE batch_id = ?;
                """,
                (batch_id,),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchSessionSaveException from e
        finally:
            self._lock.release()

    def create_session(
        self,
        session: BatchSession,
    ) -> BatchSession:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                INSERT OR IGNORE INTO batch_session (batch_id, session_id, state)
                VALUES (?, ?, ?);
                """,
                (session.batch_id, session.session_id, session.state),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchSessionSaveException from e
        finally:
            self._lock.release()
        return self.get_session(session.session_id)

    def get_session(self, session_id: str) -> BatchSession:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT *
                FROM batch_session
                WHERE session_id= ?;
                """,
                (session_id,),
            )

            result = cast(Union[sqlite3.Row, None], self._cursor.fetchone())
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchSessionNotFoundException from e
        finally:
            self._lock.release()
        if result is None:
            raise BatchSessionNotFoundException
        return self._deserialize_batch_session(dict(result))

    def _deserialize_batch_session(self, session_dict: dict) -> BatchSession:
        """Deserializes a batch session."""

        return BatchSession.parse_obj(session_dict)

    def get_created_session(self, batch_id: str) -> BatchSession:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT *
                FROM batch_session
                WHERE batch_id = ? AND state = 'created';
                """,
                (batch_id,),
            )

            result = cast(Optional[sqlite3.Row], self._cursor.fetchone())
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchSessionNotFoundException from e
        finally:
            self._lock.release()
        if result is None:
            raise BatchSessionNotFoundException
        session = self._deserialize_batch_session(dict(result))
        return session

    def get_created_sessions(self, batch_id: str) -> List[BatchSession]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT *
                FROM batch_session
                WHERE batch_id = ? AND state = created;
                """,
                (batch_id,),
            )

            result = cast(list[sqlite3.Row], self._cursor.fetchall())
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchSessionNotFoundException from e
        finally:
            self._lock.release()
        if result is None:
            raise BatchSessionNotFoundException
        sessions = list(map(lambda r: self._deserialize_batch_session(dict(r)), result))
        return sessions

    def update_session_state(
        self,
        batch_id: str,
        session_id: str,
        changes: BatchSessionChanges,
    ) -> BatchSession:
        try:
            self._lock.acquire()

            # Change the state of a batch session
            if changes.state is not None:
                self._cursor.execute(
                    f"""--sql
                    UPDATE batch_session
                    SET state = ?
                    WHERE batch_id = ? AND session_id = ?;
                    """,
                    (changes.state, batch_id, session_id),
                )

            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            raise BatchSessionSaveException from e
        finally:
            self._lock.release()
        return self.get_session(session_id)
