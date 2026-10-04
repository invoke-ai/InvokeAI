"""Leases of browser tabs on intermediate media they still show, which keep it from being cleaned up."""

from sqlalchemy import Column, Index

from invokeai.app.services.shared.database.schema.metadata import ENUM_LENGTH, USER_ID_LENGTH, table
from invokeai.app.services.shared.database.types import Key, Timestamp

intermediates_browser_holds = table(
    "intermediates_browser_holds",
    Column("user_id", Key(USER_ID_LENGTH), primary_key=True),
    Column("lease_id", Key(), primary_key=True),
    Column("media_kind", Key(ENUM_LENGTH), primary_key=True),
    Column("media_name", Key(), primary_key=True),
    Column("expires_at", Timestamp("TEXT"), nullable=False),
)

Index("idx_intermediates_browser_holds_expires_at", intermediates_browser_holds.c.expires_at)
Index(
    "idx_intermediates_browser_holds_media",
    intermediates_browser_holds.c.media_kind,
    intermediates_browser_holds.c.media_name,
    intermediates_browser_holds.c.expires_at,
)
