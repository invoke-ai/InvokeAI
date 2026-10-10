"""Installed models, and which of them belong together."""

from sqlalchemy import Column, Computed, ForeignKey, Index, UniqueConstraint

from invokeai.app.services.shared.database.dialect import JsonValue
from invokeai.app.services.shared.database.schema.metadata import (
    LONG_TEXT_INDEX_PREFIX,
    PATH_LENGTH,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import BigInt, Key, LongText


def _config_member(name: str, column_type: Key | LongText | BigInt, *, nullable: bool = False) -> Column[object]:
    """A generated column holding the member of `config` with the same name."""
    return Column(
        name,
        column_type,
        Computed(JsonValue("config", f"$.{name}", integer=isinstance(column_type, BigInt))),
        nullable=nullable,
    )


models = table(
    "models",
    Column("id", Key(), primary_key=True),
    _config_member("hash", LongText()),
    _config_member("base", LongText()),
    _config_member("type", LongText()),
    # Unique, so a server indexes it whole, which bounds its length there.
    _config_member("path", Key(PATH_LENGTH)),
    _config_member("format", LongText()),
    _config_member("name", LongText()),
    _config_member("description", LongText(), nullable=True),
    _config_member("source", LongText()),
    _config_member("source_type", LongText()),
    _config_member("source_api_response", LongText(), nullable=True),
    _config_member("trigger_phrases", LongText(), nullable=True),
    _config_member("file_size", BigInt()),
    # The whole config (JSON), with the members of its model type's subclass.
    Column("config", LongText(), nullable=False),
    inserted_at(),
    updated_at(),
    UniqueConstraint("path"),
)

Index("base_index", models.c.base, **LONG_TEXT_INDEX_PREFIX)
Index("name_index", models.c.name, **LONG_TEXT_INDEX_PREFIX)
Index("type_index", models.c.type, **LONG_TEXT_INDEX_PREFIX)

model_relationships = table(
    "model_relationships",
    # Keys of related models: model_key_1 < model_key_2, so that each pair is stored once.
    Column("model_key_1", Key(), ForeignKey("models.id", ondelete="CASCADE"), primary_key=True),
    Column("model_key_2", Key(), ForeignKey("models.id", ondelete="CASCADE"), primary_key=True),
    inserted_at(declared="TEXT DATETIME"),
)

Index("keyx_model_relationships_model_key_2", model_relationships.c.model_key_2)
