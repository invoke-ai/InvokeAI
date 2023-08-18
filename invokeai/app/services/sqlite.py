import json
import sqlite3
from threading import Lock
from typing import Generic, Optional, TypeVar, get_args

from pydantic import BaseModel, parse_obj_as, parse_raw_as

from .item_storage import ItemStorageABC, PaginatedDictResults, PaginatedResults

T = TypeVar("T", bound=BaseModel)

sqlite_memory = ":memory:"


class SqliteItemStorage(ItemStorageABC, Generic[T]):
    _filename: str
    _table_name: str
    _conn: sqlite3.Connection
    _cursor: sqlite3.Cursor
    _id_field: str
    _lock: Lock

    def __init__(self, filename: str, table_name: str, id_field: str = "id"):
        super().__init__()

        self._filename = filename
        self._table_name = table_name
        self._id_field = id_field  # TODO: validate that T has this field
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
                f"""CREATE TABLE IF NOT EXISTS {self._table_name} (
                item TEXT,
                id TEXT GENERATED ALWAYS AS (json_extract(item, '$.{self._id_field}')) VIRTUAL NOT NULL);"""
            )
            self._cursor.execute(
                f"""CREATE UNIQUE INDEX IF NOT EXISTS {self._table_name}_id ON {self._table_name}(id);"""
            )
        finally:
            self._lock.release()

    def _parse_item_from_dict(self, item: dict) -> T:
        item_type = get_args(self.__orig_class__)[0]
        parsed = parse_obj_as(item_type, item)
        return parsed

    def set(self, item: T):
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""INSERT OR REPLACE INTO {self._table_name} (item) VALUES (?);""",
                (item.json(),),
            )
            self._conn.commit()
        finally:
            self._lock.release()
        self._on_changed(item)

    def get_as_dict(self, id: str) -> Optional[dict]:
        try:
            self._lock.acquire()
            self._cursor.execute(f"""SELECT item FROM {self._table_name} WHERE id = ?;""", (str(id),))
            result = self._cursor.fetchone()
        finally:
            self._lock.release()

        if not result:
            return None

        return json.loads(result[0])

    def get(self, id: str) -> Optional[T]:
        item = self.get_as_dict(id)
        if not item:
            return None

        return self._parse_item_from_dict(item)

    def delete(self, id: str):
        try:
            self._lock.acquire()
            self._cursor.execute(f"""DELETE FROM {self._table_name} WHERE id = ?;""", (str(id),))
            self._conn.commit()
        finally:
            self._lock.release()
        self._on_deleted(id)

    def list_as_dict(self, page: int = 0, per_page: int = 10) -> PaginatedDictResults:
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""SELECT item FROM {self._table_name} LIMIT ? OFFSET ?;""",
                (per_page, page * per_page),
            )
            result = self._cursor.fetchall()

            items = list(map(lambda r: json.loads(r[0]), result))

            self._cursor.execute(f"""SELECT count(*) FROM {self._table_name};""")
            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedDictResults(items=items, page=page, pages=pageCount, per_page=per_page, total=count)

    def list(self, page: int = 0, per_page: int = 10) -> PaginatedResults[T]:
        paginated_raw_results = self.list_as_dict(page, per_page)
        items = list(map(lambda r: self._parse_item_from_dict(r), paginated_raw_results.items))
        return PaginatedResults[T](
            items=items,
            page=page,
            pages=paginated_raw_results.pages,
            per_page=per_page,
            total=paginated_raw_results.total,
        )

    def search_as_dict(self, query: str, page: int = 0, per_page: int = 10) -> PaginatedDictResults:
        try:
            self._lock.acquire()
            self._cursor.execute(
                f"""SELECT item FROM {self._table_name} WHERE item LIKE ? LIMIT ? OFFSET ?;""",
                (f"%{query}%", per_page, page * per_page),
            )
            result = self._cursor.fetchall()

            items = list(map(lambda r: json.loads(r[0]), result))

            self._cursor.execute(
                f"""SELECT count(*) FROM {self._table_name} WHERE item LIKE ?;""",
                (f"%{query}%",),
            )
            count = self._cursor.fetchone()[0]
        finally:
            self._lock.release()

        pageCount = int(count / per_page) + 1

        return PaginatedDictResults(items=items, page=page, pages=pageCount, per_page=per_page, total=count)

    def search(self, query: str, page: int = 0, per_page: int = 10) -> PaginatedResults[T]:
        paginated_raw_results = self.search_as_dict(query, page, per_page)
        items = list(map(lambda r: self._parse_item_from_dict(r), paginated_raw_results.items))
        return PaginatedResults[T](
            items=items,
            page=page,
            pages=paginated_raw_results.pages,
            per_page=per_page,
            total=paginated_raw_results.total,
        )
