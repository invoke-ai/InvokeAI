"""Video records; the files themselves are on disk."""

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
from invokeai.app.services.shared.database.types import BigInt, Key, LongText, Real, Timestamp

videos = table(
    "videos",
    Column("video_name", Key(), primary_key=True),
    Column("video_origin", Key(ENUM_LENGTH), nullable=False),
    Column("video_category", Key(ENUM_LENGTH), nullable=False),
    Column("width", BigInt(), nullable=False),
    Column("height", BigInt(), nullable=False),
    Column("duration", Real(), nullable=False, server_default=default(0.0)),
    Column("fps", Real()),
    Column("session_id", LongText()),
    Column("node_id", LongText()),
    Column("metadata", LongText()),
    Column("is_intermediate", Boolean(), server_default=default(False)),
    Column("starred", Boolean(), server_default=default(False)),
    Column("has_workflow", Boolean(), server_default=default(False)),
    # No foreign key to `users`, as for images: deleting a user leaves their media for an admin to review.
    Column("user_id", Key(USER_ID_LENGTH), nullable=False, server_default=default("system")),
    Column("video_subfolder", LongText(), nullable=False, server_default=default("")),
    inserted_at(),
    updated_at(),
    Column("deleted_at", Timestamp()),  # unused
    Column("project_id", Key()),
    Column("file_size_bytes", BigInt()),
)

Index("idx_videos_created_at", videos.c.created_at)
# As for images: partial indexes on SQLite, and server variants that lead with the condition's columns.
Index(
    "idx_videos_intermediate_scope",
    videos.c.user_id,
    videos.c.project_id,
    videos.c.created_at,
    videos.c.video_name,
    sqlite_where=text("is_intermediate = TRUE"),
).ddl_if(dialect="sqlite")
Index(
    "idx_videos_intermediate_scope",
    videos.c.is_intermediate,
    videos.c.user_id,
    videos.c.project_id,
    videos.c.created_at,
    videos.c.video_name,
).ddl_if(dialect=ON_SERVERS)
Index(
    "idx_videos_intermediates_owner",
    videos.c.user_id,
    videos.c.created_at,
    videos.c.video_name,
    sqlite_where=text("is_intermediate = TRUE"),
).ddl_if(dialect="sqlite")
Index(
    "idx_videos_intermediates_owner",
    videos.c.is_intermediate,
    videos.c.user_id,
    videos.c.created_at,
    videos.c.video_name,
).ddl_if(dialect=ON_SERVERS)
Index(
    "idx_videos_intermediates_created",
    videos.c.created_at,
    videos.c.video_name,
    sqlite_where=text("is_intermediate = TRUE"),
).ddl_if(dialect="sqlite")
Index("idx_videos_intermediates_created", videos.c.is_intermediate, videos.c.created_at, videos.c.video_name).ddl_if(
    dialect=ON_SERVERS
)
Index("idx_videos_starred", videos.c.starred)
Index(
    "idx_videos_unmeasured_intermediates",
    videos.c.created_at,
    videos.c.video_name,
    sqlite_where=text("is_intermediate = TRUE AND file_size_bytes IS NULL"),
).ddl_if(dialect="sqlite")
Index(
    "idx_videos_unmeasured_intermediates",
    videos.c.is_intermediate,
    videos.c.file_size_bytes,
    videos.c.created_at,
    videos.c.video_name,
).ddl_if(dialect=ON_SERVERS)
Index("idx_videos_user_id", videos.c.user_id)
Index("idx_videos_video_category", videos.c.video_category)
# The primary key covers it. Only SQLite has it, where a migration created it.
Index("idx_videos_video_name", videos.c.video_name, unique=True).ddl_if(dialect="sqlite")
Index("idx_videos_video_origin", videos.c.video_origin)
