"""Image records; the files themselves are on disk."""

from sqlalchemy import Boolean, Column, Index, text

from invokeai.app.services.shared.database.schema.metadata import (
    ENUM_LENGTH,
    ON_SERVERS,
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import BigInt, Key, LongText, Timestamp

images = table(
    "images",
    Column("image_name", Key(), primary_key=True),
    Column("image_origin", Key(ENUM_LENGTH), nullable=False),
    Column("image_category", Key(ENUM_LENGTH), nullable=False),
    Column("width", BigInt(), nullable=False),
    Column("height", BigInt(), nullable=False),
    Column("session_id", LongText()),
    Column("node_id", LongText()),
    Column("metadata", LongText()),
    Column("is_intermediate", Boolean(), server_default=default(False)),
    inserted_at(),
    updated_at(),
    Column("deleted_at", Timestamp()),  # unused
    Column("starred", Boolean(), server_default=default(False)),
    Column("has_workflow", Boolean(), server_default=default(False)),
    Column("user_id", Key(USER_ID_LENGTH), server_default=default("system")),
    Column("image_subfolder", LongText(), nullable=False, server_default=default("")),
    Column("project_id", Key()),
    Column("file_size_bytes", BigInt()),
)

Index("idx_images_created_at", images.c.created_at)
Index("idx_images_image_category", images.c.image_category)
# The primary key covers it. Only SQLite has it, where a migration created it.
Index("idx_images_image_name", images.c.image_name, unique=True).ddl_if(dialect="sqlite")
Index("idx_images_image_origin", images.c.image_origin)
# The intermediates of a scope, all of them, and those whose size is still unknown, each in creation order (the
# keyset of the cleanup's windows). Partial indexes on SQLite; a server has none, so its variants lead with the
# condition's columns instead, which narrows a query with the same condition to the same rows.
Index(
    "idx_images_intermediate_scope",
    images.c.user_id,
    images.c.project_id,
    images.c.created_at,
    images.c.image_name,
    sqlite_where=text("is_intermediate = TRUE"),
).ddl_if(dialect="sqlite")
Index(
    "idx_images_intermediate_scope",
    images.c.is_intermediate,
    images.c.user_id,
    images.c.project_id,
    images.c.created_at,
    images.c.image_name,
).ddl_if(dialect=ON_SERVERS)
Index(
    "idx_images_intermediates_owner",
    images.c.user_id,
    images.c.created_at,
    images.c.image_name,
    sqlite_where=text("is_intermediate = TRUE"),
).ddl_if(dialect="sqlite")
Index(
    "idx_images_intermediates_owner",
    images.c.is_intermediate,
    images.c.user_id,
    images.c.created_at,
    images.c.image_name,
).ddl_if(dialect=ON_SERVERS)
Index(
    "idx_images_intermediates_created",
    images.c.created_at,
    images.c.image_name,
    sqlite_where=text("is_intermediate = TRUE"),
).ddl_if(dialect="sqlite")
Index("idx_images_intermediates_created", images.c.is_intermediate, images.c.created_at, images.c.image_name).ddl_if(
    dialect=ON_SERVERS
)
Index("idx_images_starred", images.c.starred)
Index(
    "idx_images_unmeasured_intermediates",
    images.c.created_at,
    images.c.image_name,
    sqlite_where=text("is_intermediate = TRUE AND file_size_bytes IS NULL"),
).ddl_if(dialect="sqlite")
Index(
    "idx_images_unmeasured_intermediates",
    images.c.is_intermediate,
    images.c.file_size_bytes,
    images.c.created_at,
    images.c.image_name,
).ddl_if(dialect=ON_SERVERS)
Index("idx_images_user_id", images.c.user_id)
