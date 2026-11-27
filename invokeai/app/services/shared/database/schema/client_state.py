"""Per-user state of the web client."""

from sqlalchemy import Column, ForeignKey, Index

from invokeai.app.services.shared.database.schema.metadata import (
    CURRENT_TIMESTAMP,
    USER_ID_LENGTH,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import Key, LongText

client_state = table(
    "client_state",
    Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True),
    Column("key", Key(), primary_key=True),
    Column("value", LongText(), nullable=False),
    updated_at(sqlite_default=CURRENT_TIMESTAMP),
)

# The primary key covers it. Only SQLite has it, where a migration created it.
Index("idx_client_state_user_id", client_state.c.user_id).ddl_if(dialect="sqlite")
