"""Wildcards: an account's named lists of values that a prompt can draw from."""

import functools
import itertools
import json
from collections.abc import Sequence
from typing import Any, Optional

from sqlalchemy import Connection, Row, Update, bindparam, delete, insert, select, update

from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, mapped, read, write
from invokeai.app.services.shared.database.schema.wildcards import wildcards
from invokeai.app.services.wildcard_records.wildcard_records_common import (
    WildcardChanges,
    WildcardRecordDTO,
    WildcardWithoutId,
)

_W = wildcards.c
_COLUMNS = (_W.id, _W.name, _W.values_json, _W.user_id)

_GET = select(*_COLUMNS).where(_W.id == bindparam("wildcard_id"))
# Names are unique per account, so name order has no ties.
_OWNED = select(*_COLUMNS).where(_W.user_id == bindparam("user_id")).order_by(_W.name)
_INSERT = insert(wildcards)
_DELETE = delete(wildcards).where(_W.id == bindparam("wildcard_id"))
_OWNED_AMONG = select(_W.id).where(
    _W.user_id == bindparam("user_id"), _W.id.in_(bindparam("wildcard_ids", expanding=True))
)
_DELETE_OWNED = delete(wildcards).where(
    _W.user_id == bindparam("user_id"), _W.id.in_(bindparam("wildcard_ids", expanding=True))
)


@functools.cache
def _update(fields: tuple[str, ...]) -> Update:
    """Sets these fields of the wildcard; one statement per set of fields, of which there are few."""
    return (
        update(wildcards)
        .where(_W.id == bindparam("target_wildcard_id"))
        .values({field: bindparam(f"new_{field}") for field in fields})
    )


def _wildcard(row: Sequence[Any]) -> WildcardRecordDTO:
    wildcard_id, name, values_json, user_id = row
    return WildcardRecordDTO.from_dict({"id": wildcard_id, "name": name, "values": values_json, "user_id": user_id})


def _wildcard_or_none(row: Optional[Sequence[Any]]) -> Optional[WildcardRecordDTO]:
    return _wildcard(row) if row is not None else None


def _wildcards(rows: Sequence[Sequence[Any]]) -> list[WildcardRecordDTO]:
    return [_wildcard(row) for row in rows]


class WildcardQueries(QueryModule):
    @mapped(_wildcard_or_none)
    @read
    def get(self, conn: Connection, wildcard_id: str) -> Optional[Row[Any]]:
        return conn.execute(_GET, {"wildcard_id": wildcard_id}).first()

    @mapped(_wildcards)
    @read
    def owned_by(self, conn: Connection, user_id: str) -> Sequence[Row[Any]]:
        """The account's wildcards, in name order."""
        return conn.execute(_OWNED, {"user_id": user_id}).all()

    @write
    def insert(self, conn: Connection, wildcard_id: str, wildcard: WildcardWithoutId, user_id: str) -> None:
        conn.execute(
            _INSERT,
            {"id": wildcard_id, "name": wildcard.name, "values_json": json.dumps(wildcard.values), "user_id": user_id},
        )

    @write
    def update(self, conn: Connection, wildcard_id: str, changes: WildcardChanges) -> None:
        """Applies the changes to the wildcard's name and values."""
        fields: dict[str, Any] = {}
        if changes.name is not None:
            fields["name"] = changes.name
        if changes.values is not None:
            fields["values_json"] = json.dumps(changes.values)
        if fields:
            values = {f"new_{field}": value for field, value in fields.items()}
            conn.execute(_update(tuple(sorted(fields))), {"target_wildcard_id": wildcard_id, **values})

    @write
    def delete(self, conn: Connection, wildcard_id: str) -> None:
        conn.execute(_DELETE, {"wildcard_id": wildcard_id})

    @write
    def delete_owned(self, conn: Connection, user_id: str, wildcard_ids: Sequence[str]) -> list[str]:
        """Deletes those of the wildcards that the account owns and returns their ids; the others are left alone."""
        deleted: list[str] = []
        for chunk in itertools.batched(wildcard_ids, IN_CHUNK):
            params = {"user_id": user_id, "wildcard_ids": list(chunk)}
            owned = [row[0] for row in conn.execute(_OWNED_AMONG, params).all()]
            if owned:
                conn.execute(_DELETE_OWNED, {"user_id": user_id, "wildcard_ids": owned})
                deleted.extend(owned)
        return deleted
