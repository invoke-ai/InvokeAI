from __future__ import annotations
from abc import ABC, abstractmethod
import json
import sqlite3
from threading import Lock
from typing import Union

from pydantic import BaseModel, Field, parse_raw_as

from invokeai.app.models.image import ImageField
from invokeai.app.invocations.latent import LatentsField
from invokeai.app.services.graph import GraphExecutionState
from invokeai.app.services.item_storage import PaginatedResults


class Result(BaseModel):
    id: str = Field(description="Result ID")
    session_id: str = Field(description="Session ID")
    node_id: str = Field(description="Node ID")
    data: Union[LatentsField, ImageField] = Field(description="The result data")


class ResultWithSession(BaseModel):
    result: Result = Field(description="The result")
    session: GraphExecutionState = Field(description="The session")


class ResultsServiceABC(ABC):
    @abstractmethod
    def get(self, output_id: str) -> str:
        pass

    @abstractmethod
    def list(
        self, page: int = 0, per_page: int = 10
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
                """
                CREATE TABLE IF NOT EXISTS results (
                  id TEXT PRIMARY KEY,
                  node_id TEXT,
                  session_id TEXT,
                  data TEXT
                );
                """
            )
            self._cursor.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS result_id ON result(id);"""
            )
        finally:
            self._lock.release()

    def get(self, id: str) -> Union[ResultWithSession, None]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """
                SELECT results.data, graph_executions.state
                FROM results
                JOIN graph_executions ON results.session_id = graph_executions.id
                WHERE results.id = ?
                """,
                (id,),
            )

            result_row = self._cursor.fetchone()

            if result_row is None:
                return None

            result_raw, graph_execution_state_raw = result_row
            result = parse_raw_as(Result, result_raw)
            graph_execution_state = parse_raw_as(
                GraphExecutionState, graph_execution_state_raw
            )
        finally:
            self._lock.release()

        if not result:
            return None

        return ResultWithSession(result=result, session=graph_execution_state)

    def list(
        self, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[ResultWithSession]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """
                SELECT results.data, graph_executions.state
                FROM results
                JOIN graph_executions ON results.session_id = graph_executions.id
                LIMIT ? OFFSET ?;
                """,
                (per_page, page * per_page),
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

            self._cursor.execute("""SELECT count(*) FROM results;""")
            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedResults[ResultWithSession](
            items=items, page=page, pages=pageCount, per_page=per_page, total=count
        )

    def search(
        self, query: str, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[ResultWithSession]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                """
                SELECT results.data, graph_executions.state
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
                f"""SELECT count(*) FROM results WHERE item LIKE ?;""",
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
        with self._conn as conn:
            for node_id, result in session.results.items():
                # We'll only process 'image_output' or 'latents_output'
                if result["type"] not in ["image_output", "latents_output"]:
                    continue

                # The id depends on the result type
                if result["type"] == "image_output":
                    id = result["image"]["image_name"]
                else:  # 'latents_output'
                    id = result["latents"]["latents_name"]

                # Stringify the entire result object for the data column
                data = json.dumps(result)

                # Insert the result into the results table, ignoring if it already exists
                conn.execute(
                    """
                    INSERT OR IGNORE INTO results (id, node_id, session_id, data)
                    VALUES (?, ?, ?, ?)
                """,
                    (id, node_id, session.id, data),
                )
