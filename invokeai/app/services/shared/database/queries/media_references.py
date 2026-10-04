"""The index of which saved documents reference which media (see `invokeai.app.services.shared.media_references`).

A document's references are written in the transaction that saves or deletes the document, after the document's
own row. On a server, that row's lock keeps two saves of one document from interleaving their references. A
delete removes the references only of documents whose rows it deleted: a document written for the first time
while the delete runs has no row the delete could have locked, and must keep the references it writes.
"""

import itertools
from collections.abc import Collection

from sqlalchemy import Connection, bindparam, delete, insert

from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, write
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.shared.media_references import MediaReferenceOwnerKind, MediaReferences

_DELETE_OWNER = delete(media_references).where(
    media_references.c.owner_kind == bindparam("owner_kind"),
    media_references.c.user_id == bindparam("user_id"),
    media_references.c.owner_id == bindparam("owner_id"),
)
_DELETE_OWNERS = delete(media_references).where(
    media_references.c.owner_kind == bindparam("owner_kind"),
    media_references.c.user_id == bindparam("user_id"),
    media_references.c.owner_id.in_(bindparam("owner_ids", expanding=True)),
)
_DELETE_OWNED_BY = delete(media_references).where(
    media_references.c.user_id == bindparam("user_id"),
    media_references.c.owner_kind.in_(bindparam("owner_kinds", expanding=True)),
)
_INSERT = insert(media_references)


class MediaReferenceQueries(QueryModule):
    @write
    def replace(
        self,
        conn: Connection,
        *,
        owner_kind: MediaReferenceOwnerKind,
        user_id: str,
        owner_id: str,
        references: MediaReferences,
    ) -> None:
        """Makes the index for one document equal to `references`."""
        owner = {"owner_kind": owner_kind, "user_id": user_id, "owner_id": owner_id}
        conn.execute(_DELETE_OWNER, owner)
        rows = [{**owner, "media_kind": "image", "media_name": name} for name in sorted(references.images)]
        rows.extend({**owner, "media_kind": "video", "media_name": name} for name in sorted(references.videos))
        if rows:
            conn.execute(_INSERT, rows)

    @write
    def delete(self, conn: Connection, *, owner_kind: MediaReferenceOwnerKind, user_id: str, owner_id: str) -> None:
        conn.execute(_DELETE_OWNER, {"owner_kind": owner_kind, "user_id": user_id, "owner_id": owner_id})

    @write
    def delete_many(
        self, conn: Connection, *, owner_kind: MediaReferenceOwnerKind, user_id: str, owner_ids: Collection[str]
    ) -> None:
        for chunk in itertools.batched(owner_ids, IN_CHUNK):
            conn.execute(_DELETE_OWNERS, {"owner_kind": owner_kind, "user_id": user_id, "owner_ids": list(chunk)})

    @write
    def delete_owned_by(self, conn: Connection, user_id: str, *owner_kinds: MediaReferenceOwnerKind) -> None:
        """Deletes the references of every document of these kinds that the account owns, for documents deleted
        with the account: deleting its row locks theirs, and keeps new ones from being written meanwhile."""
        conn.execute(_DELETE_OWNED_BY, {"user_id": user_id, "owner_kinds": list(owner_kinds)})
