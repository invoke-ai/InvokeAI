"""Which installed models belong together: pairs of related models, each stored once."""

import functools

from sqlalchemy import Connection, Insert, bindparam, delete, select, union

from invokeai.app.services.shared.database.dialect import insert_ignore
from invokeai.app.services.shared.database.queries.base import QueryModule, read, write
from invokeai.app.services.shared.database.schema.models import model_relationships

_R = model_relationships.c

_DELETE = delete(model_relationships).where(
    _R.model_key_1 == bindparam("model_key_1"), _R.model_key_2 == bindparam("model_key_2")
)
# A pair is stored once, so a model's relatives are on either side of it. UNION drops duplicates.
_RELATED = union(
    select(_R.model_key_2.label("related_key")).where(_R.model_key_1 == bindparam("key")),
    select(_R.model_key_1.label("related_key")).where(_R.model_key_2 == bindparam("key")),
).order_by("related_key")


@functools.cache
def _add(dialect_name: str) -> Insert:
    return insert_ignore(dialect_name, model_relationships)


def _pair(model_key_1: str, model_key_2: str) -> dict[str, str]:
    # The smaller key first, so that a pair has one row whichever way round it is named.
    first, second = sorted((model_key_1, model_key_2))
    return {"model_key_1": first, "model_key_2": second}


class ModelRelationshipQueries(QueryModule):
    @write
    def add(self, conn: Connection, model_key_1: str, model_key_2: str) -> None:
        """Relates the two models, unless they are already."""
        conn.execute(_add(conn.dialect.name), _pair(model_key_1, model_key_2))

    @write
    def remove(self, conn: Connection, model_key_1: str, model_key_2: str) -> None:
        conn.execute(_DELETE, _pair(model_key_1, model_key_2))

    @read
    def related(self, conn: Connection, model_key: str) -> list[str]:
        """The keys of the models related to this one, in order."""
        return list(conn.execute(_RELATED, {"key": model_key}).scalars().all())
