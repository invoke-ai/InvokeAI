from enum import Enum

from abc import ABC, abstractmethod
import json
import sqlite3
from threading import Lock
from typing import Any, Union

from pydantic import BaseModel, Field, parse_obj_as, parse_raw_as
from invokeai.app.invocations.image import ImageOutput
from invokeai.app.services.graph import GraphExecutionState
from invokeai.app.invocations.latent import LatentsOutput
from invokeai.app.services.item_storage import PaginatedResults


class ResultType(str, Enum):
    image_output = "image_output"
    latents_output = "latents_output"


class Result(BaseModel):
    """A result from a session, stored in the `results` table."""

    id: str = Field(description="Result ID")
    session_id: str = Field(description="Session ID")
    node_id: str = Field(description="Node ID")
    data: Union[LatentsOutput, ImageOutput] = Field(description="The result data")


class ResultWithSession(BaseModel):
    """A result with its session, returned by the Results service."""

    result: Result = Field(description="The result")
    session: GraphExecutionState = Field(description="The session")


class ResultsServiceABC(ABC):
    """The Results service is responsible for retrieving results."""

    @abstractmethod
    def get(
        self, result_id: str, result_type: ResultType
    ) -> Union[ResultWithSession, None]:
        pass

    @abstractmethod
    def get_many(
        self, result_type: ResultType, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[ResultWithSession]:
        pass

    @abstractmethod
    def search(
        self, query: str, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[ResultWithSession]:
        pass

    @abstractmethod
    def handle_graph_execution_state_change(self, session: GraphExecutionState) -> None:
        pass


class SqliteResultsService(ResultsServiceABC):
    """SQLite implementation of the Results service."""

    _filename: str
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: Lock

    def __init__(self, filename: str):
        super().__init__()

        self._filename = filename
        self._lock = Lock()

        self._conn = sqlite3.connect(
            self._filename, check_same_thread=False
        )  # TODO: figure out a better threading solution
        self._cursor = self._conn.cursor()

        self._create_table()

    def _create_table(self):
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                CREATE TABLE IF NOT EXISTS results (
                  id TEXT PRIMARY KEY, -- the result's name
                  result_type TEXT, -- `image_output` | `latents_output`
                  node_id TEXT, -- the node that produced this result
                  session_id TEXT, -- the session that produced this result
                  data TEXT -- the result itself
                );
                """
            )
            self._cursor.execute(
                """--sql
                CREATE UNIQUE INDEX IF NOT EXISTS result_id ON results(id);
                """
            )
        finally:
            self._lock.release()

    def _parse_joined_result(self, result_row: Any, column_names: list[str]):
        result_raw = {}
        session_raw = {}
        for idx, name in enumerate(column_names):
            if name == "session":
                session_raw = json.loads(result_row[idx])
            elif name == "data":
                result_raw[name] = json.loads(result_row[idx])
            else:
                result_raw[name] = result_row[idx]

        return ResultWithSession(
            result=parse_obj_as(Result, result_raw),
            session=parse_obj_as(GraphExecutionState, session_raw),
        )

    def get(
        self, result_id: str, result_type: ResultType
    ) -> Union[ResultWithSession, None]:
        """Retrieves a result by ID and type."""
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT
                    results.id AS id,
                    results.result_type AS result_type,
                    results.node_id AS node_id,
                    results.session_id AS session_id,
                    results.data AS data,
                    graph_executions.item AS session
                FROM results
                JOIN graph_executions ON results.session_id = graph_executions.id
                WHERE results.id = ? AND results.result_type = ?
                """,
                (result_id, result_type),
            )

            result_row = self._cursor.fetchone()

            if result_row is None:
                return None

            column_names = list(map(lambda x: x[0], self._cursor.description))
            result_parsed = self._parse_joined_result(result_row, column_names)
        finally:
            self._lock.release()

        if not result_parsed:
            return None

        return result_parsed

    def get_many(
        self,
        result_type: ResultType,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ResultWithSession]:
        """Lists results of a given type."""
        try:
            self._lock.acquire()

            self._cursor.execute(
                f"""--sql
                SELECT
                    results.id AS id,
                    results.result_type AS result_type,
                    results.node_id AS node_id,
                    results.session_id AS session_id,
                    results.data AS data,
                    graph_executions.item AS session
                FROM results
                JOIN graph_executions ON results.session_id = graph_executions.id
                WHERE results.result_type = ?
                LIMIT ? OFFSET ?;
                """,
                (result_type.value, per_page, page * per_page),
            )

            result_rows = self._cursor.fetchall()
            column_names = list(map(lambda c: c[0], self._cursor.description))

            result_parsed = []

            for result_row in result_rows:
                result_parsed.append(
                    self._parse_joined_result(result_row, column_names)
                )

            self._cursor.execute("""SELECT count(*) FROM results;""")
            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedResults[ResultWithSession](
            items=result_parsed,
            page=page,
            pages=pageCount,
            per_page=per_page,
            total=count,
        )

    def search(
        self,
        query: str,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ResultWithSession]:
        """Finds results by query."""
        try:
            self._lock.acquire()
            self._cursor.execute(
                """--sql
                SELECT results.data, graph_executions.item
                FROM results
                JOIN graph_executions ON results.session_id = graph_executions.id
                WHERE item LIKE ?
                LIMIT ? OFFSET ?;
                """,
                (f"%{query}%", per_page, page * per_page),
            )

            result_rows = self._cursor.fetchall()

            items = list(
                map(
                    lambda r: ResultWithSession(
                        result=parse_raw_as(Result, r[0]),
                        session=parse_raw_as(GraphExecutionState, r[1]),
                    ),
                    result_rows,
                )
            )
            self._cursor.execute(
                """--sql
                SELECT count(*) FROM results WHERE item LIKE ?;
                """,
                (f"%{query}%",),
            )
            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedResults[ResultWithSession](
            items=items, page=page, pages=pageCount, per_page=per_page, total=count
        )

    def handle_graph_execution_state_change(self, session: GraphExecutionState) -> None:
        """Updates the results table with the results from the session."""
        with self._conn as conn:
            for node_id, result in session.results.items():
                # We'll only process 'image_output' or 'latents_output'
                if result.type not in ["image_output", "latents_output"]:
                    continue

                # The id depends on the result type
                if result.type == "image_output":
                    id = result.image.image_name
                    result_type = "image_output"
                else:
                    id = result.latents.latents_name
                    result_type = "latents_output"

                # Insert the result into the results table, ignoring if it already exists
                conn.execute(
                    """--sql
                    INSERT OR IGNORE INTO results (id, result_type, node_id, session_id, data)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (id, result_type, node_id, session.id, result.json()),
                )
