"""The session queue, and the receipts that make enqueueing idempotent."""

from sqlalchemy import Column, ForeignKey, Index

from invokeai.app.services.shared.database.schema.metadata import (
    ENUM_LENGTH,
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import BigInt, Key, LongText, Timestamp

session_queue = table(
    "session_queue",
    # Never reused (AUTOINCREMENT on SQLite): it orders the queue and pages through it.
    Column("item_id", BigInt(), primary_key=True, nullable=True),
    Column("batch_id", Key(), nullable=False),
    Column("queue_id", LongText(), nullable=False),
    Column("session_id", Key(), nullable=False, unique=True),
    Column("field_values", LongText()),
    Column("session", LongText(), nullable=False),
    Column("status", Key(ENUM_LENGTH), nullable=False, server_default=default("pending")),
    # Higher runs first.
    Column("priority", BigInt(), nullable=False, server_default=default(0)),
    Column("error_traceback", LongText()),
    inserted_at(),
    updated_at(),
    Column("started_at", Timestamp()),
    Column("completed_at", Timestamp()),
    Column("workflow", LongText()),
    Column("error_type", LongText()),
    Column("error_message", LongText()),
    Column("origin", LongText()),
    Column("destination", LongText()),
    Column("retried_from_item_id", BigInt()),
    Column("user_id", Key(USER_ID_LENGTH), server_default=default("system")),
    Column("status_sequence", BigInt(), server_default=default(0)),
    Column("device", LongText()),
    Column("workflow_call_id", Key()),
    Column("parent_item_id", BigInt()),
    Column("parent_session_id", Key()),
    Column("root_item_id", BigInt()),
    Column("workflow_call_depth", BigInt()),
    Column("project_id", LongText()),
    # Counts the writes of `session`, so a writer can tell whether it still holds the latest one.
    Column("session_revision", BigInt(), nullable=False, server_default=default(0)),
    sqlite_autoincrement=True,
)

Index("idx_session_queue_batch_id", session_queue.c.batch_id)
Index("idx_session_queue_created_priority", session_queue.c.priority)
# Covered by the longer indexes starting with the same column, as is `idx_session_queue_user_id`. Only SQLite
# has these, where migrations created them.
Index("idx_session_queue_created_status", session_queue.c.status).ddl_if(dialect="sqlite")
# The primary key and the unique constraint cover these. Only SQLite has them, where migrations created them.
Index("idx_session_queue_item_id", session_queue.c.item_id, unique=True).ddl_if(dialect="sqlite")
# A queue's listing and its status counts, answered from the index alone (2026_10_04_add_session_queue_listing_index).
# Only SQLite has it: on a server `queue_id` and `origin` are long texts, which no index holds whole.
Index(
    "idx_session_queue_listing",
    session_queue.c.queue_id,
    session_queue.c.created_at,
    session_queue.c.item_id.desc(),
    session_queue.c.user_id,
    session_queue.c.origin,
    session_queue.c.status,
).ddl_if(dialect="sqlite")
Index("idx_session_queue_session_id", session_queue.c.session_id, unique=True).ddl_if(dialect="sqlite")
Index("idx_session_queue_parent_item_id", session_queue.c.parent_item_id)
Index("idx_session_queue_parent_session_id", session_queue.c.parent_session_id)
Index("idx_session_queue_root_item_id", session_queue.c.root_item_id)
# The round-robin dequeue: pending items per user, best first.
Index(
    "idx_session_queue_round_robin_pending",
    session_queue.c.status,
    session_queue.c.user_id,
    session_queue.c.priority.desc(),
    session_queue.c.item_id.asc(),
)
Index("idx_session_queue_user_id", session_queue.c.user_id).ddl_if(dialect="sqlite")
Index("idx_session_queue_user_started_at", session_queue.c.user_id, session_queue.c.started_at)
Index("idx_session_queue_workflow_call_depth", session_queue.c.workflow_call_depth)
Index("idx_session_queue_workflow_call_id", session_queue.c.workflow_call_id)

# What an enqueue request with an idempotency key did, so that sending it again does not enqueue it twice.
session_queue_enqueue_receipts = table(
    "session_queue_enqueue_receipts",
    Column("queue_id", Key(), primary_key=True),
    Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True),
    Column("idempotency_key", Key(), primary_key=True),
    Column("payload_hash", LongText(), nullable=False),
    Column("batch_id", LongText(), nullable=False),
    Column("requested", BigInt(), nullable=False),
    Column("enqueued", BigInt(), nullable=False),
    Column("priority", BigInt(), nullable=False),
    Column("item_ids", LongText(), nullable=False),
    Column("byte_size", BigInt(), nullable=False),
    inserted_at(),
    Column("acknowledged_at", Timestamp()),
)

Index(
    "idx_session_queue_enqueue_receipts_owner_ack",
    session_queue_enqueue_receipts.c.user_id,
    session_queue_enqueue_receipts.c.acknowledged_at,
    session_queue_enqueue_receipts.c.byte_size,
)
