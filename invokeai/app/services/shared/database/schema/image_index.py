"""The image index: embeddings of images and videos, cached map projections, and custom vocabulary."""

from sqlalchemy import CheckConstraint, Column, ForeignKey, Index

from invokeai.app.services.shared.database.schema.metadata import (
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import BigInt, Blob, Key, LongText, NoCaseKey

image_embeddings = table(
    "image_embeddings",
    Column("image_name", Key(), ForeignKey("images.image_name", ondelete="CASCADE"), primary_key=True),
    # The embedding model's content hash, not its install key.
    Column("model_id", Key(), primary_key=True),
    Column("dim", BigInt(), nullable=False),
    # L2-normalized, in the little-endian float type `encoding` names: dim * 2 or dim * 4 bytes.
    Column("embedding", Blob(), nullable=False),
    inserted_at(),
    Column("encoding", LongText(), nullable=False, server_default=default("float32")),
    CheckConstraint("encoding IN ('float32', 'float16')", name="encoding"),
)

Index("idx_image_embeddings_model_id", image_embeddings.c.model_id)

video_embeddings = table(
    "video_embeddings",
    Column("video_name", Key(), ForeignKey("videos.video_name", ondelete="CASCADE"), primary_key=True),
    # The embedding model's content hash, not its install key.
    Column("model_id", Key(), primary_key=True),
    Column("dim", BigInt(), nullable=False),
    # L2-normalized, in the little-endian float type `encoding` names: dim * 2 or dim * 4 bytes.
    Column("embedding", Blob(), nullable=False),
    inserted_at(),
    Column("encoding", LongText(), nullable=False, server_default=default("float32")),
    CheckConstraint("encoding IN ('float32', 'float16')", name="encoding"),
)

Index("idx_video_embeddings_model_id", video_embeddings.c.model_id)

image_projections = table(
    "image_projections",
    Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True),
    Column("model_id", Key(), primary_key=True),
    # Fingerprint of the media the projection was computed over: a different one makes it stale.
    Column("scope_hash", LongText(), nullable=False),
    # The projection parameters (JSON).
    Column("params", LongText(), nullable=False),
    Column("point_count", BigInt(), nullable=False),
    # Media names (JSON array), row-aligned with `coords`.
    Column("image_names", LongText(), nullable=False),
    # float32, shape (point_count, 2).
    Column("coords", Blob(), nullable=False),
    inserted_at(),
    updated_at(),
    Column("item_kinds", LongText()),
)

image_index_vocab_terms = table(
    "image_index_vocab_terms",
    Column("term", NoCaseKey(), primary_key=True),
    inserted_at(),
)
