"""Per-user state of the web client: a value per key and account."""

import functools
import itertools
from typing import Any, Optional

from sqlalchemy import Connection, Insert, Select, bindparam, delete, select

from invokeai.app.services.shared.database.dialect import (
    CaseInsensitiveLike,
    like_parameters,
    like_prefix,
    upsert,
)
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, read, write
from invokeai.app.services.shared.database.schema.client_state import client_state
from invokeai.app.services.shared.database.types import now_text

_GET = select(client_state.c.value).where(
    client_state.c.user_id == bindparam("user_id"), client_state.c.key == bindparam("key")
)
_DELETE = delete(client_state).where(
    client_state.c.user_id == bindparam("user_id"), client_state.c.key == bindparam("key")
)
_LOCK_KEYS = select(client_state.c.key).where(client_state.c.user_id == bindparam("user_id")).with_for_update()
_DELETE_KEYS = delete(client_state).where(
    client_state.c.user_id == bindparam("user_id"), client_state.c.key.in_(bindparam("keys", expanding=True))
)


# Case-insensitive, as SQLite's LIKE always matched these prefixes. Most recently written first.
@functools.cache
def _keys_with_prefix(slots: int) -> Select[Any]:
    return (
        select(client_state.c.key)
        .where(
            client_state.c.user_id == bindparam("user_id"),
            CaseInsensitiveLike(client_state.c.key, "pattern", slots),
        )
        .order_by(client_state.c.updated_at.desc())
    )


@functools.cache
def _set(dialect_name: str) -> Insert:
    return upsert(dialect_name, client_state, update=("value", "updated_at"))


class ClientStateQueries(QueryModule):
    @read
    def get(self, conn: Connection, user_id: str, key: str) -> Optional[str]:
        return conn.execute(_GET, {"user_id": user_id, "key": key}).scalar_one_or_none()

    @read
    def keys_with_prefix(self, conn: Connection, user_id: str, prefix: str) -> list[str]:
        """The account's keys that start with `prefix`, ignoring case; `%` and `_` in it match only themselves."""
        pattern = like_prefix(prefix)
        parameters = {"user_id": user_id, **like_parameters("pattern", pattern, pattern.slots)}
        return list(conn.execute(_keys_with_prefix(pattern.slots), parameters).scalars().all())

    @write
    def set(self, conn: Connection, user_id: str, key: str, value: str) -> None:
        conn.execute(
            _set(conn.dialect.name), {"user_id": user_id, "key": key, "value": value, "updated_at": now_text()}
        )

    @write
    def delete(self, conn: Connection, user_id: str, key: str) -> bool:
        """Deletes the value; whether there was one."""
        return conn.execute(_DELETE, {"user_id": user_id, "key": key}).rowcount > 0

    @write
    def delete_all(self, conn: Connection, user_id: str) -> list[str]:
        """Deletes the account's values and returns their keys.

        Only values this transaction locked: on a server, a key that another transaction writes meanwhile, for
        the first time, is neither deleted nor returned.
        """
        keys: list[str] = list(conn.execute(_LOCK_KEYS, {"user_id": user_id}).scalars().all())
        for chunk in itertools.batched(keys, IN_CHUNK):
            conn.execute(_DELETE_KEYS, {"user_id": user_id, "keys": list(chunk)})
        return keys
