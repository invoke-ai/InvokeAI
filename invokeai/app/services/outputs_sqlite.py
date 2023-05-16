import json
import sqlite3
from threading import Lock
from typing import Union

from invokeai.app.services.outputs_session_storage import (
    OutputsSessionStorageABC,
    PaginatedStringResults,
)

sqlite_memory = ":memory:"


class OutputsSqliteItemStorage(OutputsSessionStorageABC):
    _filename: str
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _lock: Lock

    def __init__(
        self,
        filename: str,
    ):
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
                f"""CREATE TABLE IF NOT EXISTS outputs (
                id TEXT NOT NULL PRIMARY KEY,
                session_id TEXT NOT NULL
                );"""
            )
            self._cursor.execute(
                f"""CREATE UNIQUE INDEX IF NOT EXISTS outputs_id ON outputs(id);"""
            )
        finally:
            self._lock.release()

    def set(self, output_id: str, session_id: str):
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""INSERT OR REPLACE INTO outputs (id, session_id) VALUES (?, ?);""",
                (output_id, session_id),
            )
            self._conn.commit()
        finally:
            self._lock.release()
        self._on_changed(output_id)

    def get(self, output_id: str) -> Union[str, None]:
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""
                    SELECT graph_executions.item session
                    FROM graph_executions
                    INNER JOIN outputs ON outputs.session_id = graph_executions.id
                    WHERE outputs.id = ?;
                """,
                (output_id,),
            )
            result = self._cursor.fetchone()
        finally:
            self._lock.release()

        if not result:
            return None

        return result[0]

    def delete(self, output_id: str):
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""DELETE FROM outputs WHERE id = ?;""", (str(id),)
            )
            self._conn.commit()
        finally:
            self._lock.release()
        self._on_deleted(output_id)

    def list(self, page: int = 0, per_page: int = 10) -> PaginatedStringResults:
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""
                    SELECT graph_executions.item session
                    FROM graph_executions
                    INNER JOIN outputs ON outputs.session_id = graph_executions.id
                    LIMIT ? OFFSET ?;
                """,
                (per_page, page * per_page),
            )
            result = self._cursor.fetchall()

            items = list(map(lambda r: r[0], result))

            self._cursor.execute(
                f"""
                    SELECT count(*)
                    FROM graph_executions
                    INNER JOIN outputs ON outputs.session_id = graph_executions.id;
                """)
            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedStringResults(
            items=items, page=page, pages=pageCount, per_page=per_page, total=count
        )

    def search(
        self, query: str, page: int = 0, per_page: int = 10
    ) -> PaginatedStringResults:
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""
                    SELECT graph_executions.item session
                    FROM graph_executions
                    INNER JOIN outputs ON outputs.session_id = graph_executions.id
                    WHERE outputs.id LIKE ? LIMIT ? OFFSET ?;
                """,
                (f"%{query}%", per_page, page * per_page),
            )
            result = self._cursor.fetchall()

            items = list(map(lambda r: r[0], result))

            self._cursor.execute(
                f"""
                    SELECT count(*)
                    FROM graph_executions
                    INNER JOIN outputs ON outputs.session_id = graph_executions.id
                    WHERE outputs.id LIKE ?;
                """,                
                (f"%{query}%",),
            )
            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedStringResults(
            items=items, page=page, pages=pageCount, per_page=per_page, total=count
        )
