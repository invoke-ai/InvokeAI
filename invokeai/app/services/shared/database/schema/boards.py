"""Boards, who they are shared with, and the media on them."""

from sqlalchemy import Boolean, CheckConstraint, Column, ForeignKey, Index

from invokeai.app.services.shared.database.schema.metadata import (
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
    # The owner. No foreign key: a deleted account's boards stay, for admins to see and manage.
    Column("user_id", Key(USER_ID_LENGTH), server_default=default("system")),
    # The project the board lives in; NULL is the Library. Not a foreign key: `projects.board_id` already refers
    # here, and a key each way leaves no order to create or copy the tables in. Project and account deletion
    # clear it instead.
    Column("project_id", Key()),
    # The role every other account has on the board: 'restricted' (none), 'viewer' or 'contributor'.
    Column("general_access", LongText(), nullable=False, server_default=default("restricted")),
    CheckConstraint("general_access IN ('restricted', 'viewer', 'contributor')", name="general_access"),
)

Index("idx_boards_created_at", boards.c.created_at)
Index("idx_boards_user_id", boards.c.user_id)
Index("idx_boards_project_id", boards.c.project_id)

# The accounts a board is shared with, and their role. The owner is never a member: `boards.user_id` is.
board_members = table(
    "board_members",
    Column("board_id", Key(), ForeignKey("boards.board_id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True),
    Column("role", LongText(), nullable=False),
    Column("granted_by", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="SET NULL")),
    inserted_at("granted_at"),
    CheckConstraint("role IN ('viewer', 'contributor', 'editor')", name="role"),
)

Index("idx_board_members_user_id", board_members.c.user_id)

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
