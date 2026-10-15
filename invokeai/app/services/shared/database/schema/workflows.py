"""The workflow library."""

from sqlalchemy import Boolean, Column, Computed, Index

from invokeai.app.services.shared.database.dialect import JsonValue
from invokeai.app.services.shared.database.schema.metadata import (
    LONG_TEXT_INDEX_PREFIX,
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import BigInt, Key, LongText, Timestamp

workflow_library = table(
    "workflow_library",
    Column("workflow_id", Key(), primary_key=True),
    Column("workflow", LongText(), nullable=False),
    inserted_at(),
    updated_at(),
    # Members of `workflow`, for filtering and sorting.
    Column("category", LongText(), Computed(JsonValue("workflow", "$.meta.category")), nullable=False),
    Column("name", LongText(), Computed(JsonValue("workflow", "$.name")), nullable=False),
    Column("description", LongText(), Computed(JsonValue("workflow", "$.description")), nullable=False),
    Column("tags", LongText(), Computed(JsonValue("workflow", "$.tags"))),
    Column("opened_at", Timestamp()),
    Column("user_id", Key(USER_ID_LENGTH), server_default=default("system")),
    Column("is_public", Boolean(), nullable=False, server_default=default(False)),
    Column("last_run_at", Timestamp()),
    Column("revision", BigInt(), nullable=False, server_default=default(1)),
)

Index("idx_workflow_library_category", workflow_library.c.category, **LONG_TEXT_INDEX_PREFIX)
Index("idx_workflow_library_created_at", workflow_library.c.created_at)
Index("idx_workflow_library_description", workflow_library.c.description, **LONG_TEXT_INDEX_PREFIX)
Index("idx_workflow_library_is_public", workflow_library.c.is_public)
Index("idx_workflow_library_name", workflow_library.c.name, **LONG_TEXT_INDEX_PREFIX)
Index("idx_workflow_library_opened_at", workflow_library.c.opened_at)
Index("idx_workflow_library_updated_at", workflow_library.c.updated_at)
Index("idx_workflow_library_user_id", workflow_library.c.user_id)
