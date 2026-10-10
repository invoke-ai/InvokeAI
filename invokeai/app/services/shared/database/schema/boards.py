"""Boards, who they are shared with, and the media on them."""

from sqlalchemy import Boolean, Column, ForeignKey, Index

from invokeai.app.services.shared.database.schema.metadata import (
    ENUM_LENGTH,
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import Key, LongText, Timestamp

boards = table(
    "boards",
    Column("board_id", Key(), primary_key=True),
    Column("board_name", LongText(), nullable=False),
    Column("cover_image_name", Key(), ForeignKey("images.image_name", ondelete="SET NULL")),
    inserted_at(),
    updated_at(),
    Column("deleted_at", Timestamp()),  # unused
    Column("archived", Boolean(), server_default=default(False)),
    Column("user_id", Key(USER_ID_LENGTH), server_default=default("system")),
    Column("is_public", Boolean(), nullable=False, server_default=default(False)),
    Column("board_visibility", Key(ENUM_LENGTH), nullable=False, server_default=default("private")),
)

Index("idx_boards_board_visibility", boards.c.board_visibility)
Index("idx_boards_created_at", boards.c.created_at)
Index("idx_boards_is_public", boards.c.is_public)
Index("idx_boards_user_id", boards.c.user_id)

shared_boards = table(
    "shared_boards",
    Column("board_id", Key(), ForeignKey("boards.board_id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True),
    Column("can_edit", Boolean(), nullable=False, server_default=default(False)),
    inserted_at("shared_at"),
)

# The primary key covers it. Only SQLite has it, where a migration created it.
Index("idx_shared_boards_board_id", shared_boards.c.board_id).ddl_if(dialect="sqlite")
Index("idx_shared_boards_user_id", shared_boards.c.user_id)

# An image is on at most one board: its name is the primary key.
board_images = table(
    "board_images",
    Column("board_id", Key(), ForeignKey("boards.board_id", ondelete="CASCADE"), nullable=False),
    Column("image_name", Key(), ForeignKey("images.image_name", ondelete="CASCADE"), primary_key=True),
    inserted_at(),
    updated_at(),
    Column("deleted_at", Timestamp()),  # unused
)

# `idx_board_images_board_id_created_at` covers it. Only SQLite has it, where a migration created it.
Index("idx_board_images_board_id", board_images.c.board_id).ddl_if(dialect="sqlite")
Index("idx_board_images_board_id_created_at", board_images.c.board_id, board_images.c.created_at)

board_videos = table(
    "board_videos",
    Column("board_id", Key(), ForeignKey("boards.board_id", ondelete="CASCADE"), nullable=False),
    Column("video_name", Key(), ForeignKey("videos.video_name", ondelete="CASCADE"), primary_key=True),
    inserted_at(),
    updated_at(),
    Column("deleted_at", Timestamp()),  # unused
)

# `idx_board_videos_board_id_created_at` covers it. Only SQLite has it, where a migration created it.
Index("idx_board_videos_board_id", board_videos.c.board_id).ddl_if(dialect="sqlite")
Index("idx_board_videos_board_id_created_at", board_videos.c.board_id, board_videos.c.created_at)
