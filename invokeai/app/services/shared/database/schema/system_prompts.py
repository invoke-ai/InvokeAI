"""System prompts for the prompt-expansion models."""

from sqlalchemy import Boolean, Column, Index

from invokeai.app.services.shared.database.schema.metadata import (
    LONG_TEXT_INDEX_PREFIX,
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import BigInt, Key, LongText

system_prompts = table(
    "system_prompts",
    Column("id", Key(), primary_key=True),
    Column("name", LongText(), nullable=False),
    Column("content", LongText(), nullable=False),
    Column("user_id", Key(USER_ID_LENGTH), nullable=False, server_default=default("system")),
    Column("is_public", Boolean(), nullable=False, server_default=default(False)),
    inserted_at(),
    updated_at(),
    Column("max_tokens", BigInt()),
)

Index("idx_system_prompts_name", system_prompts.c.name, **LONG_TEXT_INDEX_PREFIX)
Index("idx_system_prompts_user_id", system_prompts.c.user_id)
