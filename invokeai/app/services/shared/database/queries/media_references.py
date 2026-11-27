"""The index of which saved documents reference which media (see `invokeai.app.services.shared.media_references`).

A document's references are written in the transaction that saves or deletes the document, after the document's
own row. On a server, that row's lock keeps two saves of one document from interleaving their references. A
delete removes the references only of documents whose rows it deleted: a document written for the first time
while the delete runs has no row the delete could have locked, and must keep the references it writes.
"""

import itertools
from collections.abc import Collection
from typing import Optional

from sqlalchemy import Connection, String, bindparam, delete, insert, null, select

from invokeai.app.services.shared.database.dialect import InBoundSet, bound_set
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, read, write
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.shared.database.schema.projects import orphaned_projects_2026_08_06, projects
from invokeai.app.services.shared.database.schema.workflows import workflow_library
from invokeai.app.services.shared.media_references import MediaKind, MediaReferenceOwnerKind, MediaReferences

_R = media_references.c
_P = projects.c

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

_OWNERS = select(_R.media_name, _R.owner_kind, _R.user_id, _R.owner_id).where(
    _R.media_kind == bindparam("media_kind"), InBoundSet(_R.media_name, bindparam("names", type_=String()))
)

_DOCUMENT_NAMES = {
    "project": select(_P.user_id, _P.project_id, _P.name).where(_P.project_id.in_(bindparam("ids", expanding=True))),
    # A workspace is named after its project, whose owner need not be the workspace's.
    "project_workspace": select(null(), _P.project_id, _P.name).where(
        _P.project_id.in_(bindparam("ids", expanding=True))
    ),
    "quarantined_project": select(
        orphaned_projects_2026_08_06.c.user_id,
        orphaned_projects_2026_08_06.c.project_id,
        orphaned_projects_2026_08_06.c.name,
    ).where(orphaned_projects_2026_08_06.c.project_id.in_(bindparam("ids", expanding=True))),
    # Workflow ids are unique across accounts, so the owner is not part of the match.
    "workflow": select(null(), workflow_library.c.workflow_id, workflow_library.c.name).where(
        workflow_library.c.workflow_id.in_(bindparam("ids", expanding=True))
    ),
}


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
    def delete_many_owners(
        self, conn: Connection, *, owner_kind: MediaReferenceOwnerKind, user_ids: Collection[str], owner_id: str
    ) -> None:
        """Deletes the references several accounts hold through one document, such as each member's workspace in a
        project: by key, not by a scan of every such document's references."""
        for user_id in user_ids:
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

    @read
    def owners(self, conn: Connection, kind: MediaKind, names: Collection[str]) -> list[tuple[str, str, str, str]]:
        """(media, owner kind, owner account, owner id) of each document naming one of the named media."""
        return [
            (str(r[0]), str(r[1]), str(r[2]), str(r[3]))
            for r in conn.execute(_OWNERS, {"media_kind": kind, "names": bound_set(names)}).all()
        ]

    @read
    def document_names(
        self, conn: Connection, owners: Collection[tuple[str, str, str]]
    ) -> dict[tuple[str, str, str], str]:
        """The names of the documents behind these (owner kind, owner account, owner id) keys, one query per kind and
        chunk. Client state has none, nor has a document that no longer exists."""
        wanted: dict[str, set[tuple[str, str, str]]] = {}
        for owner in owners:
            if owner[0] in _DOCUMENT_NAMES:
                wanted.setdefault(owner[0], set()).add(owner)
        found: dict[tuple[str, str, str], str] = {}
        for owner_kind, kind_owners in wanted.items():
            ids = sorted({owner_id for _, _, owner_id in kind_owners})
            named: dict[tuple[Optional[str], str], str] = {}
            for chunk in itertools.batched(ids, IN_CHUNK):
                for r in conn.execute(_DOCUMENT_NAMES[owner_kind], {"ids": list(chunk)}).all():
                    if r[2] is not None:
                        named[(r[0], str(r[1]))] = r[2]
            unowned = owner_kind in ("workflow", "project_workspace")
            for owner in kind_owners:
                name = named.get((None if unowned else owner[1], owner[2]))
                if name is not None:
                    found[owner] = name
        return found
