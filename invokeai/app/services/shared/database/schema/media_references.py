"""Which documents (projects, workflows, client state) reference which media, so that referenced media is kept."""

from sqlalchemy import Column, Index

from invokeai.app.services.shared.database.schema.metadata import ENUM_LENGTH, USER_ID_LENGTH, table
from invokeai.app.services.shared.database.types import Key

media_references = table(
    "media_references",
    # A MediaReferenceOwnerKind.
    Column("owner_kind", Key(ENUM_LENGTH), primary_key=True),
    # The owning account: project ids are unique per user, not globally.
    Column("user_id", Key(USER_ID_LENGTH), primary_key=True),
    # A project id, a workflow id or a client state key.
    Column("owner_id", Key(), primary_key=True),
    # 'image' or 'video'.
    Column("media_kind", Key(ENUM_LENGTH), primary_key=True),
    Column("media_name", Key(), primary_key=True),
)

Index("idx_media_references_media", media_references.c.media_kind, media_references.c.media_name)
