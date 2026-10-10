"""Index the queue-scoped reads that run on every queue refresh and status change.

Listing a queue's item ids (`WHERE queue_id = ? [AND user_id = ?] [AND origin LIKE ?] ORDER BY
created_at`) and counting its items by status (`... GROUP BY status`, globally and per user) had no
usable index, so each scanned the whole retained history. Every column they read except `queue_id`
is stored after the large `session` blob, so reading it per row walks that row's overflow pages:
tens of milliseconds per call at 10k history rows, under the database lock the queue's writers need.

One index answers all of them from the index alone. It carries the filtered and grouped columns so
no row is read. `origin` filters are prefix `LIKE`s, which cannot seek a BINARY-collated index, so
they are evaluated against index entries. `status` is included because without it the planner
prefers this index for the status counts and then reads every row.

The key is `created_at ASC, item_id DESC`: scanned backwards it yields the newest-first listing's
`created_at DESC, item_id ASC` without a sort, while every new row, being the newest, lands at the
index's right edge. The mirrored `created_at DESC` key gives the same plans, but there each insert
lands at the left edge and splits pages half-full: about 1.6x the index size once history grows.

The database keeps no table statistics, so this index attracts statements that filter only on
`queue_id`; the queue's queries keep such terms off it with `dialect.Unindexed`.

SQLite only. On MySQL and MariaDB `queue_id` and `origin` are long texts, which no index holds whole, and a server's
planner weighs its indexes by their statistics.
"""

from sqlalchemy import inspect, text

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_NAME = "idx_session_queue_listing"


def _add_session_queue_listing_index(context: PortableMigrationContext) -> None:
    if context.conn.dialect.name != "sqlite":
        return
    if any(index["name"] == _NAME for index in inspect(context.conn).get_indexes("session_queue")):
        return
    context.op.create_index(
        _NAME,
        "session_queue",
        ["queue_id", "created_at", text("item_id DESC"), "user_id", "origin", "status"],
    )


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_04_add_session_queue_listing_index",
        depends_on="2026_10_01_add_anima_variant",
        callback=_add_session_queue_listing_index,
    )
