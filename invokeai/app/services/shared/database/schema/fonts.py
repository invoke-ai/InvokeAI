"""The font catalog: fonts users uploaded, and fonts found in the configured font directory."""

from sqlalchemy import CheckConstraint, Column, ForeignKey, Index, text

from invokeai.app.services.shared.database.schema.metadata import (
    ENUM_LENGTH,
    PATH_LENGTH,
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
)
from invokeai.app.services.shared.database.types import BigInt, Key, LongText

fonts = table(
    "fonts",
    Column("id", Key(), primary_key=True),
    Column("owner_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE")),
    Column("scope", Key(ENUM_LENGTH), nullable=False),
    Column("source", Key(ENUM_LENGTH), nullable=False),
    Column("filename", LongText(), nullable=False),
    Column("storage_path", LongText()),
    # Relative to the font directory.
    Column("source_path", Key(PATH_LENGTH)),
    # From the font file's name table.
    Column("family", LongText(), nullable=False),
    Column("label", LongText(), nullable=False),
    Column("style", LongText(), nullable=False),
    Column("weight", BigInt(), nullable=False),
    # SHA-256, hex.
    Column("content_hash", Key(64), nullable=False),
    Column("byte_size", BigInt(), nullable=False),
    Column("axes_json", LongText(), nullable=False, server_default=default("[]")),
    Column("instances_json", LongText(), nullable=False, server_default=default("[]")),
    inserted_at(),
    # Set by the font service; no trigger ever maintained it.
    inserted_at("updated_at"),
    CheckConstraint("scope IN ('private', 'shared')", name="scope"),
    CheckConstraint("source IN ('uploaded', 'directory')", name="source"),
    CheckConstraint("weight >= 1 AND weight <= 1000", name="weight"),
    CheckConstraint("length(content_hash) = 64", name="content_hash"),
    CheckConstraint("byte_size > 0", name="byte_size"),
    CheckConstraint(
        """
        (source = 'directory' AND owner_id IS NULL AND scope = 'shared' AND storage_path IS NULL AND source_path IS NOT NULL)
        OR
        (source = 'uploaded' AND storage_path IS NOT NULL AND source_path IS NULL AND
            ((scope = 'private' AND owner_id IS NOT NULL) OR (scope = 'shared' AND owner_id IS NULL)))
        """,
        name="source_fields",
    ),
)

# Partial on SQLite only. On a server it covers every row, which is the same constraint: by the CHECK above, only
# directory fonts have a source path, and NULLs never collide.
Index(
    "idx_fonts_directory_source_path",
    fonts.c.source_path,
    unique=True,
    sqlite_where=text("source = 'directory'"),
)
Index("idx_fonts_uploaded_hash", fonts.c.source, fonts.c.scope, fonts.c.owner_id, fonts.c.content_hash)
# For listing by family, case-insensitively. Only SQLite has it: a server can neither index a column under
# another collation nor sort by a prefix index, and `idx_fonts_uploaded_hash` serves the filter there.
Index(
    "idx_fonts_visible_private",
    fonts.c.source,
    fonts.c.scope,
    fonts.c.owner_id,
    fonts.c.family.collate("NOCASE"),
).ddl_if(dialect="sqlite")
