"""Jobs that move images between subfolders, and what each did to each image."""

from sqlalchemy import Boolean, CheckConstraint, Column, ForeignKey, Index

from invokeai.app.services.shared.database.schema.metadata import (
    ENUM_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import BigInt, Key, LongText

image_subfolder_move_jobs = table(
    "image_subfolder_move_jobs",
    Column("id", BigInt(), primary_key=True, nullable=True),
    Column("state", LongText(), nullable=False),
    inserted_at(),
    updated_at(),
    Column("error_message", LongText()),
    CheckConstraint("state IN ('planned', 'moving', 'moved', 'committed', 'error')", name="state"),
)

image_subfolder_move_items = table(
    "image_subfolder_move_items",
    # No ON DELETE: the items are the audit trail of a job's moves, never deleted on their own.
    Column("job_id", BigInt(), ForeignKey("image_subfolder_move_jobs.id"), primary_key=True),
    # Deleting an image takes its move history with it.
    Column("image_name", Key(), ForeignKey("images.image_name", ondelete="CASCADE"), primary_key=True),
    Column("old_subfolder", LongText(), nullable=False),
    Column("new_subfolder", LongText(), nullable=False),
    Column("is_intermediate", Boolean(), nullable=False, server_default=default(False)),
    Column("old_path", LongText()),
    Column("new_path", LongText()),
    Column("old_thumbnail_path", LongText()),
    Column("new_thumbnail_path", LongText()),
    Column("state", Key(ENUM_LENGTH), nullable=False),
    Column("error_message", LongText()),
    CheckConstraint("state IN ('planned', 'moved', 'committed', 'error')", name="state"),
)

Index("idx_image_subfolder_move_items_image_name", image_subfolder_move_items.c.image_name)
Index(
    "idx_image_subfolder_move_items_job_state", image_subfolder_move_items.c.job_id, image_subfolder_move_items.c.state
)
