"""State of the intermediates cleanup: what keeps intermediate media from being cleaned up, and what cannot be measured.

The browser holds outlive the process. The session holds and the unmeasurable marks are the process's own
state, kept in tables so that every connection sees them; the cleanup service empties them when it starts.
"""

from sqlalchemy import Column, Index

from invokeai.app.services.shared.database.schema.metadata import ENUM_LENGTH, USER_ID_LENGTH, table
from invokeai.app.services.shared.database.types import Key, Timestamp

# Leases of browser tabs on intermediate media they still show.
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

# Cached outputs a running session reuses: held while the session is active, then for the recency grace from
# `released_at`, when the session was first seen finished.
intermediates_session_holds = table(
    "intermediates_session_holds",
    Column("session_id", Key(), primary_key=True),
    Column("media_kind", Key(ENUM_LENGTH), primary_key=True),
    Column("media_name", Key(), primary_key=True),
    Column("released_at", Timestamp("TEXT")),
)

Index(
    "idx_intermediates_session_holds_media",
    intermediates_session_holds.c.media_kind,
    intermediates_session_holds.c.media_name,
    intermediates_session_holds.c.released_at,
)

# Intermediates whose file could not be measured, so that the measurement moves on to later ones.
intermediates_unmeasurable = table(
    "intermediates_unmeasurable",
    Column("media_kind", Key(ENUM_LENGTH), primary_key=True),
    Column("media_name", Key(), primary_key=True),
)
