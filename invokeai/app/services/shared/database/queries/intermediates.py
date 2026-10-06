"""Intermediates cleanup: what every intermediate is under the cleanup policy, and the state that decides it.

One classification expression decides it, the same whether it aggregates a summary, describes a preview, pages
an operation or guards a delete:

- ``active``: produced or referenced by a pending, waiting or running queue item, or a completed child of an
  active root workflow; held by a running session that reuses it as a cached output; or leased by a browser tab.
- ``recent``: created inside the grace window, or held by a session that ended inside it.
- ``referenced``: named by a saved document (`media_references`).
- ``safe``: none of the above.

The caller binds the clock (`now`, `recent_cutoff`) and the names its scan of active queue sessions found
(`active_names`, a `bound_set`), so that every statement of one classification sees the same instant and the
same active work.
"""

import functools
from collections.abc import Collection, Sequence
from typing import Any, Literal, NamedTuple, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    Select,
    String,
    Subquery,
    Table,
    and_,
    bindparam,
    case,
    delete,
    exists,
    false,
    func,
    literal,
    null,
    or_,
    select,
    union_all,
    update,
)

from invokeai.app.services.shared.database.dialect import (
    InBoundSet,
    KeysetAfter,
    OrderedJoin,
    bound_set,
    fixed_limit,
    insert_ignore,
    sql_true,
    upsert,
)
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, read, write
from invokeai.app.services.shared.database.schema.boards import board_images
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.intermediates import (
    intermediates_browser_holds,
    intermediates_session_holds,
    intermediates_unmeasurable,
)
from invokeai.app.services.shared.database.schema.media_references import media_references
from invokeai.app.services.shared.database.schema.projects import orphaned_projects_2026_08_06, projects
from invokeai.app.services.shared.database.schema.session_queue import session_queue
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.shared.database.schema.workflows import workflow_library

MediaKind = Literal["image", "video"]
Classification = Literal["safe", "referenced", "active", "recent"]
CleanupMode = Literal["safe", "force"]

ACTIVE_QUEUE_STATUSES = ("pending", "in_progress", "waiting")
# The selected rows of a preview, per statement. A statement has a power of two of slots, the spare ones bound to
# NULL, so that a few statements serve every selection while each row is checked against few slots.
TARGET_SLOTS = 32

_MEDIA: dict[MediaKind, tuple[Table, str, str]] = {
    "image": (images, "image_name", "image_subfolder"),
    "video": (videos, "video_name", "video_subfolder"),
}
_Q = session_queue.c
_H = intermediates_session_holds.c
_B = intermediates_browser_holds.c
_U = intermediates_unmeasurable.c
_R = media_references.c
_P = projects.c


def _protected_items() -> Subquery:
    """The active queue items, and the completed items of an active root, which keep their outputs until it ends.

    A child is completed before its outputs reach its waiting parent; the whole completed subtree stays protected
    until the root finishes. The join starts from the active roots, then looks the children up by index, rather
    than scanning all completed history.
    """
    statuses = [literal(status) for status in ACTIVE_QUEUE_STATUSES]
    root = session_queue.alias("root")
    child = session_queue.alias("child")
    direct = select(_Q.item_id, _Q.session_id, _Q.session_revision).where(_Q.status.in_(statuses))
    nested = (
        select(child.c.item_id, child.c.session_id, child.c.session_revision)
        .select_from(OrderedJoin(root, child, child.c.root_item_id == root.c.item_id))
        .where(root.c.status.in_(statuses), child.c.status == literal("completed"))
    )
    return union_all(direct, nested).subquery("protected")


_PROTECTED = _protected_items()
_ACTIVE_ITEMS = select(_PROTECTED.c.item_id, func.coalesce(_PROTECTED.c.session_revision, 0))
_SESSIONS = select(_Q.item_id, func.coalesce(_Q.session_revision, 0), _Q.session).where(
    _Q.item_id.in_(bindparam("item_ids", expanding=True))
)
_SESSION_IS_ACTIVE = select(literal(1)).where(
    _Q.session_id == bindparam("session_id"), _Q.status.in_([literal(status) for status in ACTIVE_QUEUE_STATUSES])
)

# A session hold is released the first time its session is seen finished, and dropped once its grace is over.
_RELEASE_SESSION_HOLDS = (
    update(intermediates_session_holds)
    .where(_H.released_at.is_(None), _H.session_id.not_in(select(_protected_items().c.session_id)))
    .values(released_at=bindparam("now"))
)
_DROP_RELEASED_SESSION_HOLDS = delete(intermediates_session_holds).where(_H.released_at <= bindparam("recent_cutoff"))
_CLEAR_SESSION_HOLDS = delete(intermediates_session_holds)

_SWEEP_BROWSER_HOLDS = delete(intermediates_browser_holds).where(_B.expires_at <= bindparam("now"))
_RELEASE_LEASE = delete(intermediates_browser_holds).where(
    _B.user_id == bindparam("user_id"), _B.lease_id == bindparam("lease_id")
)
_LEASES = select(_B.lease_id, func.max(_B.expires_at)).where(_B.user_id == bindparam("user_id")).group_by(_B.lease_id)
_RELEASE_LEASES = delete(intermediates_browser_holds).where(
    _B.user_id == bindparam("user_id"), _B.lease_id.in_(bindparam("lease_ids", expanding=True))
)

_CLEAR_UNMEASURABLE = delete(intermediates_unmeasurable)

_DOCUMENT_NAMES = {
    "project": select(_P.user_id, _P.project_id, _P.name).where(_P.project_id.in_(bindparam("ids", expanding=True))),
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


def _is_intermediate(media: Table) -> ColumnElement[bool]:
    # As the partial indexes on intermediates spell it, so that SQLite uses them.
    return media.c.is_intermediate == sql_true()


def _classification(kind: MediaKind) -> ColumnElement[str]:
    media, name_column, _ = _MEDIA[kind]
    name = media.c[name_column]
    held = select(_H.media_name).where(_H.media_kind == literal(kind))
    leased = exists(
        select(literal(1)).where(
            _B.media_kind == literal(kind), _B.media_name == name, _B.expires_at > bindparam("now")
        )
    )
    active = or_(
        and_(media.c.session_id.is_not(None), media.c.session_id.in_(select(_protected_items().c.session_id))),
        InBoundSet(name, bindparam("active_names", type_=String())),
        name.in_(held.where(_H.released_at.is_(None))),
        leased,
    )
    recent = or_(media.c.created_at > bindparam("recent_cutoff"), name.in_(held.where(_H.released_at.is_not(None))))
    referenced = exists(select(literal(1)).where(_R.media_kind == literal(kind), _R.media_name == name))
    return case(
        (active, literal("active")),
        (recent, literal("recent")),
        (referenced, literal("referenced")),
        else_=literal("safe"),
    )


ScopeKind = Literal["all", "owner", "owners", "project", "unassigned", "targets"]


class ScopeShape(NamedTuple):
    """What a statement's scope looks like: its kind, and for selected rows how many slots it has."""

    kind: ScopeKind
    slots: int = 0


class ScopeSpec(NamedTuple):
    """Which intermediates a statement covers. Statements are built per kind of scope, its values are bound.

    - "all";
    - "owner": `user_id`'s;
    - "owners": those of the accounts in `users`;
    - "project": the media of `targets[0]`, an (owner, project) of a project that exists;
    - "unassigned": `user_id`'s media of no existing project;
    - "targets": up to `TARGET_SLOTS` (owner, project) rows, a None project for the owner's unassigned row.
    """

    kind: ScopeKind
    user_id: Optional[str] = None
    users: frozenset[str] = frozenset()
    targets: tuple[tuple[str, Optional[str]], ...] = ()

    @property
    def shape(self) -> ScopeShape:
        if self.kind != "targets":
            return ScopeShape(self.kind)
        if not 0 < len(self.targets) <= TARGET_SLOTS:
            raise ValueError(f"1 to {TARGET_SLOTS} rows a statement")
        return ScopeShape("targets", 1 << (len(self.targets) - 1).bit_length())

    def parameters(self) -> dict[str, Any]:
        if self.kind in ("owner", "unassigned"):
            return {"scope_user": self.user_id}
        if self.kind == "project":
            owner, project = self.targets[0]
            return {"scope_user": owner, "scope_project": project}
        if self.kind == "owners":
            return {"scope_users": bound_set(self.users)}
        if self.kind == "targets":
            slots = [*self.targets, *[(None, None)] * (self.shape.slots - len(self.targets))]
            parameters: dict[str, Any] = {}
            for i, (owner, project) in enumerate(slots):
                parameters[f"target_user_{i}"] = owner
                parameters[f"target_project_{i}"] = project
            return parameters
        return {}


def _scope_conditions(shape: ScopeShape, media: Table) -> list[ColumnElement[bool]]:
    scope = shape.kind
    project_exists = _P.project_id.is_not(None)
    if scope == "owner":
        return [media.c.user_id == bindparam("scope_user")]
    if scope == "owners":
        return [InBoundSet(media.c.user_id, bindparam("scope_users", type_=String()))]
    if scope == "project":
        return [
            media.c.user_id == bindparam("scope_user"),
            media.c.project_id == bindparam("scope_project"),
            project_exists,
        ]
    if scope == "unassigned":
        return [media.c.user_id == bindparam("scope_user"), _P.project_id.is_(None)]
    if scope == "targets":
        slots = []
        for i in range(shape.slots):
            target_project = bindparam(f"target_project_{i}", type_=String())
            slots.append(
                and_(
                    media.c.user_id == bindparam(f"target_user_{i}", type_=String()),
                    or_(
                        and_(target_project.is_(None), _P.project_id.is_(None)),
                        and_(media.c.project_id == target_project, project_exists),
                    ),
                )
            )
        return [or_(*slots)]
    return []


def _scoped(kind: MediaKind, scope: ScopeShape) -> Subquery:
    """Every intermediate of `kind` in `scope`, with its (owner, project) row and its classification.

    Media whose project no longer exists, or that never had one, belong to the owner's unassigned row.
    """
    media, name_column, _ = _MEDIA[kind]
    owned_project = and_(_P.user_id == media.c.user_id, _P.project_id == media.c.project_id)
    return (
        select(
            media.c[name_column].label("name"),
            media.c.user_id.label("user_id"),
            case((_P.project_id.is_(None), null()), else_=media.c.project_id).label("project_key"),
            media.c.file_size_bytes.label("file_size_bytes"),
            media.c.created_at.label("created_at"),
            _classification(kind).label("cls"),
        )
        .select_from(media.outerjoin(projects, owned_project))
        .where(_is_intermediate(media), *_scope_conditions(scope, media))
        .subquery("scoped")
    )


def _deletable(kind: MediaKind, classified: Subquery, mode: CleanupMode, is_admin: bool) -> ColumnElement[int]:
    """1 for what a cleanup of `mode` may delete, else 0.

    Force mode adds referenced items, but another account's document is never the caller's to break unless the
    caller administers the instance (`caller_user_id`).
    """
    cls = classified.c.cls
    if mode == "safe":
        allowed: ColumnElement[bool] = cls == literal("safe")
    elif is_admin:
        allowed = cls.in_([literal("safe"), literal("referenced")])
    else:
        others_document = exists(
            select(literal(1)).where(
                _R.media_kind == literal(kind),
                _R.media_name == classified.c.name,
                _R.user_id != bindparam("caller_user_id"),
            )
        )
        allowed = or_(cls == literal("safe"), and_(cls == literal("referenced"), ~others_document))
    return case((allowed, 1), else_=0)


def _with_deletable(kind: MediaKind, scope: ScopeShape, mode: CleanupMode, is_admin: bool) -> Subquery:
    scoped = _scoped(kind, scope)
    return select(*scoped.c, _deletable(kind, scoped, mode, is_admin).label("deletable")).subquery("classified")


@functools.cache
def _aggregate(kind: MediaKind, scope: ScopeShape) -> Select[Any]:
    s = _scoped(kind, scope)
    return select(
        s.c.user_id,
        s.c.project_key,
        s.c.cls,
        func.count(),
        func.sum(func.coalesce(s.c.file_size_bytes, 0)),
        func.sum(case((s.c.file_size_bytes.is_(None), 1), else_=0)),
    ).group_by(s.c.user_id, s.c.project_key, s.c.cls)


@functools.cache
def _preview(kind: MediaKind, scope: ScopeShape, mode: CleanupMode, is_admin: bool) -> Select[Any]:
    c = _with_deletable(kind, scope, mode, is_admin)
    return select(
        c.c.user_id,
        c.c.project_key,
        c.c.cls,
        c.c.deletable,
        func.count(),
        func.sum(func.coalesce(c.c.file_size_bytes, 0)),
        func.sum(case((c.c.file_size_bytes.is_(None), 1), else_=0)),
    ).group_by(c.c.user_id, c.c.project_key, c.c.cls, c.c.deletable)


@functools.cache
def _acknowledged(kind: MediaKind, scope: ScopeShape, is_admin: bool) -> Select[Any]:
    """The documents naming the referenced items a force cleanup may delete, with how many each names."""
    c = _with_deletable(kind, scope, "force", is_admin)
    return (
        select(_R.owner_kind, _R.user_id, _R.owner_id, func.count())
        .select_from(c.join(media_references, and_(_R.media_kind == literal(kind), _R.media_name == c.c.name)))
        .where(c.c.cls == literal("referenced"), c.c.deletable == 1)
        .group_by(_R.owner_kind, _R.user_id, _R.owner_id)
    )


@functools.cache
def _window(kind: MediaKind, scope: ScopeShape, mode: CleanupMode, is_admin: bool) -> Select[Any]:
    """The next rows of a scope in creation order, after (`after_created_at`, `after_name`), classified.

    The page is chosen from the plain rows, by an index seek, before anything is classified: a server cannot merge a
    derived table whose columns have subqueries, and classifying the whole remaining scope on every window would
    make a cleanup quadratic in its scope.
    """
    media, name_column, _ = _MEDIA[kind]
    name = media.c[name_column]
    after = KeysetAfter(
        media.c.created_at,
        name,
        bindparam("after_created_at", type_=String()),
        bindparam("after_name", type_=String()),
    )
    owned_project = and_(_P.user_id == media.c.user_id, _P.project_id == media.c.project_id)
    page = (
        select(name.label("name"))
        .select_from(media.outerjoin(projects, owned_project))
        .where(_is_intermediate(media), *_scope_conditions(scope, media), after)
        .order_by(media.c.created_at, name)
        .limit(bindparam("limit"))
        .subquery("page")
    )
    classified = (
        select(
            name.label("name"),
            media.c.file_size_bytes.label("file_size_bytes"),
            media.c.created_at.label("created_at"),
            _classification(kind).label("cls"),
        )
        .select_from(media.join(page, page.c.name == name))
        .subquery("classified")
    )
    return select(
        classified.c.name,
        classified.c.file_size_bytes,
        _deletable(kind, classified, mode, is_admin),
        classified.c.created_at,
    ).order_by(classified.c.created_at, classified.c.name)


@functools.cache
def _still_deletable(kind: MediaKind, mode: CleanupMode, is_admin: bool) -> Select[Any]:
    """Which of the named intermediates a cleanup may delete now, with their owners."""
    media, name_column, _ = _MEDIA[kind]
    name = media.c[name_column]
    classified = (
        select(name.label("name"), media.c.user_id.label("user_id"), _classification(kind).label("cls"))
        .where(InBoundSet(name, bindparam("names", type_=String())), _is_intermediate(media))
        .subquery("named")
    )
    return select(classified.c.name, classified.c.user_id).where(_deletable(kind, classified, mode, is_admin) == 1)


@functools.cache
def _count_existing(kind: MediaKind) -> Select[Any]:
    media, name_column, _ = _MEDIA[kind]
    return select(func.count()).where(InBoundSet(media.c[name_column], bindparam("names", type_=String())))


@functools.cache
def _hold_for_lease(kind: MediaKind, dialect_name: str) -> Any:
    """Leases the named intermediates the account owns. A refresh of the same lease that ran meanwhile has inserted
    the same rows on a server, where writers do not wait for each other here: they get the later expiry."""
    media, name_column, _ = _MEDIA[kind]
    leased = select(
        bindparam("user_id", type_=String()),
        bindparam("lease_id", type_=String()),
        literal(kind),
        media.c[name_column],
        bindparam("expires_at", type_=String()),
    ).where(
        InBoundSet(media.c[name_column], bindparam("names", type_=String())),
        media.c.user_id == bindparam("user_id", type_=String()),
        _is_intermediate(media),
    )
    return upsert(dialect_name, intermediates_browser_holds, update=["expires_at"]).from_select(
        ["user_id", "lease_id", "media_kind", "media_name", "expires_at"], leased
    )


@functools.cache
def _reference_owners(kind: MediaKind) -> Select[Any]:
    return select(_R.media_name, _R.owner_kind, _R.user_id, _R.owner_id).where(
        _R.media_kind == literal(kind), InBoundSet(_R.media_name, bindparam("names", type_=String()))
    )


def _unmeasured(kind: MediaKind) -> tuple[Table, list[ColumnElement[bool]]]:
    media, name_column, _ = _MEDIA[kind]
    marked = exists(select(literal(1)).where(_U.media_kind == literal(kind), _U.media_name == media.c[name_column]))
    return media, [_is_intermediate(media), media.c.file_size_bytes.is_(None), ~marked]


@functools.cache
def _any_unmeasured(kind: MediaKind) -> Select[Any]:
    media, conditions = _unmeasured(kind)
    return select(literal(1)).select_from(media).where(*conditions).limit(fixed_limit(1))


@functools.cache
def _next_unmeasured(kind: MediaKind) -> Select[Any]:
    media, conditions = _unmeasured(kind)
    _, name_column, subfolder_column = _MEDIA[kind]
    return (
        select(media.c[name_column], media.c[subfolder_column])
        .where(*conditions, media.c.created_at <= bindparam("created_before"))
        .order_by(media.c.created_at, media.c[name_column])
        .limit(bindparam("limit"))
    )


@functools.cache
def _hold(dialect_name: str) -> Any:
    # A hold taken again starts over: held while its session runs.
    return upsert(dialect_name, intermediates_session_holds, update=["released_at"])


@functools.cache
def _mark_unmeasurable(dialect_name: str) -> Any:
    return insert_ignore(dialect_name, intermediates_unmeasurable)


_COVER = (
    select(board_images.c.image_name)
    .join(images, images.c.image_name == board_images.c.image_name)
    .where(board_images.c.board_id == _P.board_id, images.c.is_intermediate == false())
    .order_by(board_images.c.created_at.desc(), board_images.c.image_name.desc())
    .limit(fixed_limit(1))
    .scalar_subquery()
)
_PROJECTS = select(_P.user_id, _P.project_id, _P.name, _COVER)
_OWNERS_PROJECTS = _PROJECTS.where(_P.user_id == bindparam("user_id"))


class AggregateRow(NamedTuple):
    user_id: str
    project_key: Optional[str]
    classification: Classification
    count: int
    measured_bytes: int
    unmeasured: int


class PreviewRow(NamedTuple):
    user_id: str
    project_key: Optional[str]
    classification: Classification
    deletable: bool
    count: int
    measured_bytes: int
    unmeasured: int


class WindowRow(NamedTuple):
    name: str
    file_size_bytes: Optional[int]
    deletable: bool
    created_at: str


def _clock_parameters(now: str, recent_cutoff: str, active_names: Collection[str]) -> dict[str, Any]:
    return {"now": now, "recent_cutoff": recent_cutoff, "active_names": bound_set(active_names)}


class IntermediateQueries(QueryModule):
    # region active work

    @read
    def active_items(self, conn: Connection) -> dict[int, int]:
        """The protected queue items, each with the revision of its stored session."""
        return {int(row[0]): int(row[1]) for row in conn.execute(_ACTIVE_ITEMS)}

    @read
    def sessions(self, conn: Connection, item_ids: Collection[int]) -> list[tuple[int, int, Optional[str]]]:
        """(item, revision, session JSON) of each named queue item."""
        rows: list[tuple[int, int, Optional[str]]] = []
        ordered = sorted(item_ids)
        for start in range(0, len(ordered), IN_CHUNK):
            chunk = ordered[start : start + IN_CHUNK]
            rows.extend((int(r[0]), int(r[1]), r[2]) for r in conn.execute(_SESSIONS, {"item_ids": chunk}))
        return rows

    @read
    def is_session_active(self, conn: Connection, session_id: str) -> bool:
        return conn.execute(_SESSION_IS_ACTIVE, {"session_id": session_id}).first() is not None

    @read
    def count_existing(self, conn: Connection, kind: MediaKind, names: Collection[str]) -> int:
        """How many of the named media exist."""
        return int(conn.execute(_count_existing(kind), {"names": bound_set(names)}).scalar_one())

    # endregion

    # region holds

    @write
    def expire_session_holds(self, conn: Connection, *, now: str, recent_cutoff: str) -> None:
        """Releases the holds of sessions no longer active, and drops those whose grace is over."""
        conn.execute(_RELEASE_SESSION_HOLDS, {"now": now})
        conn.execute(_DROP_RELEASED_SESSION_HOLDS, {"recent_cutoff": recent_cutoff})

    @write
    def hold_session_media(self, conn: Connection, session_id: str, held: Sequence[tuple[MediaKind, str]]) -> None:
        if held:
            conn.execute(
                _hold(conn.dialect.name),
                [
                    {"session_id": session_id, "media_kind": kind, "media_name": name, "released_at": None}
                    for kind, name in held
                ],
            )

    @write
    def clear_process_state(self, conn: Connection) -> None:
        """Empties the session holds and the unmeasurable marks, which belong to the process that wrote them."""
        conn.execute(_CLEAR_SESSION_HOLDS)
        conn.execute(_CLEAR_UNMEASURABLE)

    @write
    def sweep_browser_holds(self, conn: Connection, now: str) -> None:
        conn.execute(_SWEEP_BROWSER_HOLDS, {"now": now})

    @write
    def release_lease(self, conn: Connection, user_id: str, lease_id: str) -> None:
        conn.execute(_RELEASE_LEASE, {"user_id": user_id, "lease_id": lease_id})

    @write
    def hold_for_lease(
        self, conn: Connection, kind: MediaKind, *, user_id: str, lease_id: str, names: Collection[str], expires_at: str
    ) -> None:
        """Leases those of the named media that are the account's intermediates."""
        conn.execute(
            _hold_for_lease(kind, conn.dialect.name),
            {"user_id": user_id, "lease_id": lease_id, "names": bound_set(names), "expires_at": expires_at},
        )

    @read
    def leases(self, conn: Connection, user_id: str) -> list[tuple[str, str]]:
        """The account's leases, each with when it was last refreshed (its latest expiry)."""
        return [(str(r[0]), str(r[1])) for r in conn.execute(_LEASES, {"user_id": user_id})]

    @write
    def release_leases(self, conn: Connection, user_id: str, lease_ids: Sequence[str]) -> None:
        for start in range(0, len(lease_ids), IN_CHUNK):
            conn.execute(_RELEASE_LEASES, {"user_id": user_id, "lease_ids": list(lease_ids[start : start + IN_CHUNK])})

    # endregion

    # region classification

    @read
    def aggregate(
        self,
        conn: Connection,
        kind: MediaKind,
        scope: ScopeSpec,
        *,
        now: str,
        recent_cutoff: str,
        active_names: Collection[str],
    ) -> list[AggregateRow]:
        """(owner, project, classification) groups of a scope's intermediates, with counts and sizes."""
        parameters = {**_clock_parameters(now, recent_cutoff, active_names), **scope.parameters()}
        return [
            AggregateRow(str(r[0]), r[1], r[2], int(r[3]), int(r[4] or 0), int(r[5] or 0))
            for r in conn.execute(_aggregate(kind, scope.shape), parameters)
        ]

    @read
    def preview(
        self,
        conn: Connection,
        kind: MediaKind,
        scope: ScopeSpec,
        *,
        mode: CleanupMode,
        is_admin: bool,
        caller_user_id: str,
        now: str,
        recent_cutoff: str,
        active_names: Collection[str],
    ) -> list[PreviewRow]:
        parameters = {
            **_clock_parameters(now, recent_cutoff, active_names),
            **scope.parameters(),
            "caller_user_id": caller_user_id,
        }
        return [
            PreviewRow(str(r[0]), r[1], r[2], bool(r[3]), int(r[4]), int(r[5] or 0), int(r[6] or 0))
            for r in conn.execute(_preview(kind, scope.shape, mode, is_admin), parameters)
        ]

    @read
    def acknowledged(
        self,
        conn: Connection,
        kind: MediaKind,
        scope: ScopeSpec,
        *,
        is_admin: bool,
        caller_user_id: str,
        now: str,
        recent_cutoff: str,
        active_names: Collection[str],
    ) -> list[tuple[str, str, str, int]]:
        """(owner kind, owner account, owner id, items) of the documents a force cleanup of the scope breaks."""
        parameters = {
            **_clock_parameters(now, recent_cutoff, active_names),
            **scope.parameters(),
            "caller_user_id": caller_user_id,
        }
        return [
            (str(r[0]), str(r[1]), str(r[2]), int(r[3]))
            for r in conn.execute(_acknowledged(kind, scope.shape, is_admin), parameters)
        ]

    @read
    def window(
        self,
        conn: Connection,
        kind: MediaKind,
        scope: ScopeSpec,
        *,
        mode: CleanupMode,
        is_admin: bool,
        caller_user_id: str,
        after: tuple[str, str],
        limit: int,
        now: str,
        recent_cutoff: str,
        active_names: Collection[str],
    ) -> list[WindowRow]:
        parameters = {
            **_clock_parameters(now, recent_cutoff, active_names),
            **scope.parameters(),
            "caller_user_id": caller_user_id,
            "after_created_at": after[0],
            "after_name": after[1],
            "limit": limit,
        }
        return [
            WindowRow(str(r[0]), r[1], bool(r[2]), str(r[3]))
            for r in conn.execute(_window(kind, scope.shape, mode, is_admin), parameters)
        ]

    @read
    def still_deletable(
        self,
        conn: Connection,
        kind: MediaKind,
        names: Collection[str],
        *,
        mode: CleanupMode,
        is_admin: bool,
        caller_user_id: str,
        now: str,
        recent_cutoff: str,
        active_names: Collection[str],
    ) -> list[tuple[str, str]]:
        """(name, owner) of those of the named intermediates a cleanup may delete now."""
        parameters = {
            **_clock_parameters(now, recent_cutoff, active_names),
            "names": bound_set(names),
            "caller_user_id": caller_user_id,
        }
        return [(str(r[0]), str(r[1])) for r in conn.execute(_still_deletable(kind, mode, is_admin), parameters)]

    @read
    def reference_owners(
        self, conn: Connection, kind: MediaKind, names: Collection[str]
    ) -> list[tuple[str, str, str, str]]:
        """(media, owner kind, owner account, owner id) of each document naming one of the named media."""
        return [
            (str(r[0]), str(r[1]), str(r[2]), str(r[3]))
            for r in conn.execute(_reference_owners(kind), {"names": bound_set(names)})
        ]

    # endregion

    # region documents and projects

    @read
    def projects(self, conn: Connection, user_id: Optional[str]) -> list[tuple[str, str, str, Optional[str]]]:
        """(owner, project, name, cover) of the account's projects (None: everyone's); the cover is the newest image
        on the project's board that is not an intermediate."""
        statement = _OWNERS_PROJECTS if user_id is not None else _PROJECTS
        return [(str(r[0]), str(r[1]), str(r[2]), r[3]) for r in conn.execute(statement, {"user_id": user_id})]

    @read
    def document_names(
        self, conn: Connection, owner_kind: str, ids: Sequence[str]
    ) -> list[tuple[Optional[str], str, Optional[str]]]:
        """(owner, id, name) of the documents of `owner_kind` with the given ids; workflows have no owner here."""
        rows: list[tuple[Optional[str], str, Optional[str]]] = []
        for start in range(0, len(ids), IN_CHUNK):
            chunk = list(ids[start : start + IN_CHUNK])
            rows.extend((r[0], str(r[1]), r[2]) for r in conn.execute(_DOCUMENT_NAMES[owner_kind], {"ids": chunk}))
        return rows

    # endregion

    # region measurement

    @read
    def any_unmeasured(self, conn: Connection, kind: MediaKind) -> bool:
        return conn.execute(_any_unmeasured(kind)).first() is not None

    @read
    def next_unmeasured(
        self, conn: Connection, kind: MediaKind, *, limit: int, created_before: str
    ) -> list[tuple[str, str]]:
        """(name, subfolder) of unmeasured intermediates created by `created_before`, oldest first."""
        statement = _next_unmeasured(kind)
        return [
            (str(r[0]), str(r[1])) for r in conn.execute(statement, {"limit": limit, "created_before": created_before})
        ]

    @write
    def mark_unmeasurable(self, conn: Connection, kind: MediaKind, names: Sequence[str]) -> None:
        if names:
            conn.execute(
                _mark_unmeasurable(conn.dialect.name), [{"media_kind": kind, "media_name": name} for name in names]
            )

    # endregion
