"""Record who stages a queue item's results, so that browsers on different devices agree on one.

A retried item's results are staged into the project it ran for by one browser. A browser claims the delivery
(`delivery_claimant`, `delivery_claimed_at`) before staging and confirms it (`delivered_at`) after; a claim that
is not confirmed lapses, so a browser that went away does not keep the results from every other one.
"""

from sqlalchemy import Column, inspect

from invokeai.app.services.shared.database.types import LongText, Timestamp
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)


def _add_session_queue_delivery(context: PortableMigrationContext) -> None:
    existing = {column["name"] for column in inspect(context.conn).get_columns("session_queue")}
    for column in (
        Column("delivery_claimant", LongText()),
        Column("delivery_claimed_at", Timestamp()),
        Column("delivered_at", Timestamp()),
    ):
        if column.name not in existing:
            context.op.add_column("session_queue", column)


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_13_add_session_queue_delivery",
        depends_on="2026_10_08_add_board_project_membership",
        callback=_add_session_queue_delivery,
    )
