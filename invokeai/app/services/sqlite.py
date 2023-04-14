from typing import Any, Generic, TypeVar, Union, get_args
from pydantic import BaseModel, parse_raw_as

from .item_storage import ItemStorageABC, PaginatedResults

from sqlalchemy import create_engine, String, TEXT, Engine, select
from sqlalchemy.orm import DeclarativeBase, mapped_column, Session


T = TypeVar("T", bound=BaseModel)


class Base(DeclarativeBase):
    pass


class SqliteItemStorage(ItemStorageABC, Generic[T]):
    _filename: str
    _table_name: str
    _id_field: str
    _engine: Engine
    _table: Any # TODO: figure out how to type this

    def __init__(self, filename: str, table_name: str, id_field: str = "id"):
        super().__init__()

        self._filename = filename
        self._table_name = table_name
        self._id_field = id_field  # TODO: validate that T has this field

        self._engine = create_engine(f"sqlite+pysqlite:///{self._filename}")
        self._create_table()

    def _create_table(self):
        # dynamically create the ORM model class to avoid name collisions

        # cannot access `self.__orig_class__` in `__init__` or `__new__` so
        # format the table name into the class name
        pascal_table_name = self._table_name.replace("_", " ").title()
        pascal_table_name = pascal_table_name.replace(" ", "")

        table_dict = dict(
            __tablename__=self._table_name,
            id=mapped_column(String, primary_key=True),
            item=mapped_column(TEXT, nullable=False),
        )

        self._table = type(pascal_table_name, (Base,), table_dict)

        Base.metadata.create_all(self._engine)

    def _parse_item(self, item: str) -> T:
        item_type = get_args(self.__orig_class__)[0]
        return parse_raw_as(item_type, item)

    def set(self, item: T):
        session = Session(self._engine)

        item_id = str(getattr(item, self._id_field))
        new_item = self._table(id=item_id, item=item.json())

        session.merge(new_item)

        session.commit()
        session.close()

        self._on_changed(item)

    def get(self, id: str) -> Union[T, None]:
        session = Session(self._engine)

        item = session.get(self._table, str(id))

        session.close()

        if not item:
            return None

        return self._parse_item(item.item)

    def delete(self, id: str):
        session = Session(self._engine)

        item = session.get(self._table, id)
        session.delete(item)

        session.commit()
        session.close()

        self._on_deleted(id)

    def list(self, page: int = 0, per_page: int = 10) -> PaginatedResults[T]:
        session = Session(self._engine)

        stmt = select(self._table.item).limit(per_page).offset(page * per_page)
        result = session.execute(stmt)

        items = list(map(lambda r: self._parse_item(r[0]), result))
        count = session.query(self._table.item).count()

        session.commit()
        session.close()

        pageCount = int(count / per_page) + 1

        return PaginatedResults[T](
            items=items, page=page, pages=pageCount, per_page=per_page, total=count
        )

    def search(
        self, query: str, page: int = 0, per_page: int = 10
    ) -> PaginatedResults[T]:
        session = Session(self._engine)

        stmt = (
            session.query(self._table)
            .where(self._table.item.like(f"%{query}%"))
            .limit(per_page)
            .offset(page * per_page)
        )

        result = session.execute(stmt)

        items = list(map(lambda r: self._parse_item(r[0].item), result))
        count = session.query(self._table.item).count()

        pageCount = int(count / per_page) + 1

        return PaginatedResults[T](
            items=items, page=page, pages=pageCount, per_page=per_page, total=count
        )
