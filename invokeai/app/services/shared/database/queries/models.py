"""Installed models: each one's config, a JSON document whose members the other columns are generated from."""

import functools
from collections.abc import Sequence
from typing import Any, Optional

from sqlalchemy import ColumnElement, Connection, Select, bindparam, delete, insert, literal, select, update

from invokeai.app.services.shared.database.dialect import fold_case
from invokeai.app.services.shared.database.queries.base import QueryModule, locking, mapped, read, write
from invokeai.app.services.shared.database.schema.models import models

_M = models.c

# The longest key and path a model can have: a server stores no longer one in their columns.
MAX_KEY_LENGTH: int = _M.id.type.length
MAX_PATH_LENGTH: int = _M.path.type.length

_GET = select(_M.config).where(_M.id == bindparam("key"))
_LOCK = _GET.with_for_update()
_EXISTS = select(literal(1)).where(_M.id == bindparam("key"))
_AT_PATH = select(_M.config).where(_M.path == bindparam("path"))
# A hash is not unique. The oldest model of a file comes first, as SQLite's table order had it, and the key breaks
# ties: the route that recalls a model by its hash takes the first.
_WITH_HASH = select(_M.config).where(_M.hash == bindparam("hash")).order_by(_M.created_at, _M.id)
_PATHS = select(_M.path)
_INSERT = insert(models)
_SAVE = update(models).where(_M.id == bindparam("key")).values(config=bindparam("config"))
_DELETE = delete(models).where(_M.id == bindparam("key"))

# The search filters, in the order `search` takes them.
_FILTERS = (_M.name, _M.base, _M.type, _M.format)
# Keyed by the values of `ModelRecordOrderBy`, which is not imported: its module loads the model manager. Orders by
# name ignore case, which SQL cannot do alike on every backend: `_by_name` sorts them in Python, from the columns
# `_SORTED_COLUMNS` names. A base is lowercase ASCII (`BaseModelType`), so it sorts as it is.
_ORDERINGS: dict[str, tuple[ColumnElement[Any], ...]] = {
    "type": (_M.type,),
    "base": (_M.base,),
    "format": (_M.format,),
    "size": (_M.file_size,),
    "created_at": (_M.created_at,),
    "updated_at": (_M.updated_at,),
    "path": (_M.path,),
}
_SORTED_COLUMNS = (_M.config, _M.id, _M.type, _M.base, _M.name, _M.format)
# The sort keys of the orders `_ORDERINGS` leaves out, from a row of `_SORTED_COLUMNS`; the key breaks ties.
_NAME_KEYS = {
    "default": lambda row: (row[2], row[3], fold_case(row[4]), row[5], row[1]),
    "name": lambda row: (fold_case(row[4]), row[1]),
}


@functools.cache
def _search(filtered: tuple[bool, ...], order_by: str, descending: bool) -> Select[Any]:
    """The configs of the models matching the filters that are set, in order; few combinations of filters and
    orders exist."""
    conditions = tuple(
        column == bindparam(f"filter_{column.name}") for column, on in zip(_FILTERS, filtered, strict=True) if on
    )
    if order_by in _NAME_KEYS:
        return select(*_SORTED_COLUMNS).where(*conditions)
    # The key breaks ties, so that equal values keep one order.
    ordering = (*_ORDERINGS[order_by], _M.id)
    return (
        select(_M.config)
        .where(*conditions)
        .order_by(*(column.desc() if descending else column.asc() for column in ordering))
    )


def _configs(result: tuple[Sequence[Any], str, bool]) -> list[str]:
    """The configs of rows `search` read, in order: sorted here, outside the transaction, by name ignoring case."""
    rows, order_by, descending = result
    if order_by in _NAME_KEYS:
        rows = sorted(rows, key=_NAME_KEYS[order_by], reverse=descending)
        return [row[0] for row in rows]
    return list(rows)


class ModelQueries(QueryModule):
    @read
    def get(self, conn: Connection, key: str) -> Optional[str]:
        """The model's config."""
        return conn.execute(_GET, {"key": key}).scalar()

    @locking
    def lock(self, conn: Connection, key: str) -> Optional[str]:
        """The model's config, its row locked until the transaction ends."""
        return conn.execute(_LOCK, {"key": key}).scalar()

    @read
    def exists(self, conn: Connection, key: str) -> bool:
        return conn.execute(_EXISTS, {"key": key}).first() is not None

    @mapped(_configs)
    @read
    def search(
        self,
        conn: Connection,
        *,
        name: Optional[str],
        base: Optional[str],
        model_type: Optional[str],
        model_format: Optional[str],
        order_by: str,
        descending: bool,
    ) -> tuple[Sequence[Any], str, bool]:
        """The configs of the models matching every filter given (an empty one matches all), ordered by
        `order_by`, a value of `ModelRecordOrderBy`: the rows `_configs` makes them from."""
        values = (name, base, model_type, model_format)
        statement = _search(tuple(bool(value) for value in values), order_by, descending)
        parameters = {f"filter_{column.name}": value for column, value in zip(_FILTERS, values, strict=True) if value}
        rows = (
            conn.execute(statement, parameters).all()
            if order_by in _NAME_KEYS
            else conn.execute(statement, parameters).scalars().all()
        )
        return rows, order_by, descending

    @read
    def at_path(self, conn: Connection, path: str) -> list[str]:
        """The configs of the models at the path: none or one, since a path is unique."""
        return list(conn.execute(_AT_PATH, {"path": path}).scalars().all())

    @read
    def with_hash(self, conn: Connection, hash: str) -> list[str]:
        """The configs of the models whose file has the hash."""
        return list(conn.execute(_WITH_HASH, {"hash": hash}).scalars().all())

    @read
    def paths(self, conn: Connection) -> list[str]:
        """The path of every model, also of one whose config no longer validates."""
        return list(conn.execute(_PATHS).scalars().all())

    @write
    def insert(self, conn: Connection, key: str, config: str) -> None:
        conn.execute(_INSERT, {"id": key, "config": config})

    @write
    def save(self, conn: Connection, key: str, config: str) -> bool:
        """Replaces the model's config; whether the model exists."""
        return conn.execute(_SAVE, {"key": key, "config": config}).rowcount > 0

    @write
    def delete(self, conn: Connection, key: str) -> bool:
        """Deletes the model, and its relationships with it; whether there was one."""
        return conn.execute(_DELETE, {"key": key}).rowcount > 0
