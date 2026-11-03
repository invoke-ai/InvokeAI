"""Wildcards: named lists of values that a prompt can draw from."""

from sqlalchemy import Column, ForeignKey, Index

from invokeai.app.services.shared.database.schema.metadata import (
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import Key, LongText

wildcards = table(
    "wildcards",
    Column("id", Key(), primary_key=True),
    # At most 128 characters (validated by the service).
    Column("name", Key(), nullable=False),
    Column("values_json", LongText(), nullable=False, server_default=default("[]")),
    Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
    inserted_at(),
    updated_at(),
)

Index("idx_wildcards_user_id_name", wildcards.c.user_id, wildcards.c.name, unique=True)
