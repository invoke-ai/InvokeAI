"""Style presets: saved prompt templates."""

from sqlalchemy import Boolean, Column, Index

from invokeai.app.services.shared.database.schema.metadata import (
    LONG_TEXT_INDEX_PREFIX,
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import Key, LongText

style_presets = table(
    "style_presets",
    Column("id", Key(), primary_key=True),
    Column("name", LongText(), nullable=False),
    Column("preset_data", LongText(), nullable=False),
    Column("type", LongText(), nullable=False, server_default=default("user")),
    inserted_at(),
    updated_at(),
    Column("user_id", Key(USER_ID_LENGTH), server_default=default("system")),
    Column("is_public", Boolean(), nullable=False, server_default=default(False)),
)

Index("idx_style_presets_is_public", style_presets.c.is_public)
Index("idx_style_presets_name", style_presets.c.name, **LONG_TEXT_INDEX_PREFIX)
Index("idx_style_presets_user_id", style_presets.c.user_id)
