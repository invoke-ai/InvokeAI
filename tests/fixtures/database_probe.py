"""Probe tables and a probe query module for exercising the database layer without the application schema."""

from collections.abc import Callable
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    Column,
    Connection,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    delete,
    func,
    insert,
    select,
    update,
)

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.base import QueryModule, QueryScope, read, write

probe_metadata = MetaData()

probe_items = Table(
    "probe_items",
    probe_metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("name", String(32), nullable=False, unique=True),
    Column("size", Integer, nullable=False),
    # Table-level: MariaDB rejects a named CHECK inside a column definition.
    CheckConstraint("size >= 0", name="ck_probe_items_size"),
    mysql_engine="InnoDB",
)

probe_links = Table(
    "probe_links",
    probe_metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("item_id", Integer, ForeignKey("probe_items.id", ondelete="CASCADE"), nullable=False),
    mysql_engine="InnoDB",
)


class ProbeItems(QueryModule):
    @write
    def add(self, conn: Connection, item_id: int, name: Optional[str], size: int = 0) -> None:
        conn.execute(insert(probe_items).values(id=item_id, name=name, size=size))

    @write
    def link(self, conn: Connection, link_id: int, item_id: int) -> None:
        conn.execute(insert(probe_links).values(id=link_id, item_id=item_id))

    @write
    def resize(self, conn: Connection, item_id: int, size: int) -> None:
        conn.execute(update(probe_items).where(probe_items.c.id == item_id).values(size=size))

    @write
    def resize_both(self, conn: Connection, first: int, second: int, between: Callable[[], None]) -> None:
        """Updates two rows in the given order, calling `between` in between (to interleave two writers)."""
        conn.execute(update(probe_items).where(probe_items.c.id == first).values(size=1))
        between()
        conn.execute(update(probe_items).where(probe_items.c.id == second).values(size=1))

    @read
    def read_then_write(self, conn: Connection, between: Callable[[], None]) -> None:
        """Reads, calls `between`, then writes, all in a read transaction -- which no real query method does.

        On SQLite a read transaction holds a snapshot; if another connection commits in `between`, the
        write cannot proceed on that stale snapshot and fails at once: a lost race, as a deadlock is.
        """
        conn.execute(select(func.count()).select_from(probe_items)).scalar_one()
        between()
        conn.execute(update(probe_items).values(size=0))

    @write
    def remove(self, conn: Connection, item_id: int) -> None:
        conn.execute(delete(probe_items).where(probe_items.c.id == item_id))

    @read
    def names(self, conn: Connection) -> list[str]:
        return list(conn.execute(select(probe_items.c.name).order_by(probe_items.c.id)).scalars())

    @read
    def link_ids(self, conn: Connection) -> list[int]:
        return list(conn.execute(select(probe_links.c.id).order_by(probe_links.c.id)).scalars())

    @read
    def count(self, conn: Connection) -> int:
        return conn.execute(select(func.count()).select_from(probe_items)).scalar_one()


class ProbeQueries(Queries):
    def __init__(self, database: Database, scope: Optional[QueryScope] = None) -> None:
        super().__init__(database, scope)
        self.items = ProbeItems(self._scope)


def create_probe_tables(database: Database) -> None:
    with database.begin(write=True) as conn:
        probe_metadata.create_all(conn)
