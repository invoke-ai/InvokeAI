"""Key projects by `project_id` alone, and clear board memberships that name no project.

Project ids were unique per account; they become unique across accounts so that a project can be named without
its owner, which sharing needs. Clients have always generated them at random, so two accounts holding the same
id is not expected; where it happens anyway, the oldest project keeps the id and every other one gets a fresh
`project-<uuid>`, with everything that names it rewritten: its boards' membership, the provenance of its images
and videos, its queue items and their origins, its media references, and the open-project state the web client
keeps for its owner. Each re-keyed project is logged.

Boards of accounts deleted earlier still name projects that went with the account; those memberships are
cleared, which leaves the boards in their owner's Library, and their number is logged. From now on deleting an
account clears its boards' memberships with it.

All data changes run before the first DDL: on a server every DDL statement commits on its own, and a run that
fails among the data changes then rolls them back whole instead of leaving half a project re-keyed. Re-keying
runs only while `projects` is still keyed per account, so a resumed run never re-keys twice.
"""

import json
from functools import partial
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    column,
    exists,
    func,
    inspect,
    select,
    table,
)

from invokeai.app.services.shared.database.schema.metadata import USER_ID_LENGTH, default, inserted_at, updated_at
from invokeai.app.services.shared.database.types import BigInt, Key, LongText
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_PER_ACCOUNT_TABLE = "projects_per_account"
_ORIGIN_PREFIX = "webv2:p:"
# As the application and the web client serialize these documents: compact, with non-ASCII text as itself. A
# document near the size limit, serialized any wider, would come out over it.
_dumps = partial(json.dumps, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
# Client state is the web client's own string, stored as sent and never bounded: a draft cut mid-emoji arrives as
# the escape `\ud83d`, which parses to a lone surrogate that no UTF-8 column can hold as itself. Escaped, every
# non-ASCII character is written as an escape, so the text differs from the client's but reads as the same value
# to any JSON reader; the value has no size limit a wider serialization could cross.
_dumps_client_state = partial(json.dumps, ensure_ascii=True, allow_nan=False, separators=(",", ":"))

_projects = table("projects", column("project_id"), column("user_id"), column("created_at"), column("data"))
_boards = table("boards", column("board_id"), column("user_id"), column("project_id"))
_images = table("images", column("user_id"), column("project_id"))
_videos = table("videos", column("user_id"), column("project_id"))
_session_queue = table("session_queue", column("item_id"), column("user_id"), column("project_id"), column("origin"))
_media_references = table("media_references", column("owner_kind"), column("user_id"), column("owner_id"))
_client_state = table("client_state", column("user_id"), column("key"), column("value"))


def _is_keyed_by_project_id(context: PortableMigrationContext) -> bool:
    return inspect(context.conn).get_pk_constraint("projects")["constrained_columns"] == ["project_id"]


def _replace_id(value: Any, old_id: str, new_id: str) -> Any:
    """The JSON value with every string equal to the old id, and every key equal to it, replaced."""
    if isinstance(value, str):
        return new_id if value == old_id else value
    if isinstance(value, list):
        return [_replace_id(item, old_id, new_id) for item in value]
    if isinstance(value, dict):
        return {(new_id if key == old_id else key): _replace_id(item, old_id, new_id) for key, item in value.items()}
    return value


def _rekey(context: PortableMigrationContext, user_id: str, old_id: str, new_id: str) -> None:
    conn = context.conn
    # The client's document names its project too; a client opening it under the old id would save to nothing.
    document = conn.execute(
        select(_projects.c.data).where(_projects.c.user_id == user_id, _projects.c.project_id == old_id)
    ).scalar_one()
    try:
        rewritten_document = _dumps(_replace_id(json.loads(document), old_id, new_id))
    except ValueError:
        rewritten_document = document
    conn.execute(
        _projects.update()
        .where(_projects.c.user_id == user_id, _projects.c.project_id == old_id)
        .values(data=rewritten_document)
    )
    for target in (_projects, _boards, _images, _videos, _session_queue):
        conn.execute(
            target.update().where(target.c.user_id == user_id, target.c.project_id == old_id).values(project_id=new_id)
        )
    old_prefix = f"{_ORIGIN_PREFIX}{old_id}:"
    for item_id, origin in conn.execute(
        select(_session_queue.c.item_id, _session_queue.c.origin).where(
            _session_queue.c.user_id == user_id, _session_queue.c.origin.startswith(old_prefix, autoescape=True)
        )
    ).all():
        conn.execute(
            _session_queue.update()
            .where(_session_queue.c.item_id == item_id)
            .values(origin=f"{_ORIGIN_PREFIX}{new_id}:{origin[len(old_prefix) :]}")
        )
    conn.execute(
        _media_references.update()
        .where(
            _media_references.c.owner_kind == "project",
            _media_references.c.user_id == user_id,
            _media_references.c.owner_id == old_id,
        )
        .values(owner_id=new_id)
    )
    # The web client's open projects, active project and covers name projects by id.
    for key, value in conn.execute(
        select(_client_state.c.key, _client_state.c.value).where(
            _client_state.c.user_id == user_id, _client_state.c.value.contains(old_id, autoescape=True)
        )
    ).all():
        try:
            document = json.loads(value)
            rewritten = _replace_id(document, old_id, new_id)
            rewritten_value = _dumps_client_state(rewritten)
        except ValueError:
            continue
        if rewritten != document:
            conn.execute(
                _client_state.update()
                .where(_client_state.c.user_id == user_id, _client_state.c.key == key)
                .values(value=rewritten_value)
            )


def _rekey_duplicates(context: PortableMigrationContext) -> None:
    shared_ids = (
        select(_projects.c.project_id).group_by(_projects.c.project_id).having(func.count() > 1).scalar_subquery()
    )
    holders = context.conn.execute(
        select(_projects.c.project_id, _projects.c.user_id)
        .where(_projects.c.project_id.in_(shared_ids))
        .order_by(_projects.c.project_id, _projects.c.created_at, _projects.c.user_id)
    ).all()
    keeper: dict[str, str] = {}
    for project_id, user_id in holders:
        if keeper.setdefault(project_id, user_id) == user_id:
            continue
        new_id = f"project-{uuid4()}"
        _rekey(context, user_id, project_id, new_id)
        context.logger.warning(
            f"Project id {project_id} was held by more than one account; the copy owned by {user_id} is now {new_id}"
        )


def _clear_unresolvable_memberships(context: PortableMigrationContext) -> None:
    """Boards name projects of their own owner; a membership no such project resolves goes back to the Library."""
    resolves = exists().where(_projects.c.project_id == _boards.c.project_id, _projects.c.user_id == _boards.c.user_id)
    cleared = context.conn.execute(
        _boards.update().where(_boards.c.project_id.is_not(None), ~resolves).values(project_id=None)
    ).rowcount
    if cleared:
        context.logger.warning(f"{cleared} boards named a project that no longer exists; they are in the Library now")


def _rebuild_projects_on_sqlite(context: PortableMigrationContext) -> None:
    """SQLite changes a primary key only by rebuilding the table, inside the migration's transaction. The old table
    is moved aside first, so the new one is created under its own name and its constraints are named for it. No
    table refers to `projects` by foreign key, so the move rewrites none."""
    context.op.rename_table("projects", _PER_ACCOUNT_TABLE)
    # The tables its foreign keys name, as far as the CREATE TABLE needs them.
    Table("users", context.metadata, Column("user_id", Key(USER_ID_LENGTH), primary_key=True))
    Table("boards", context.metadata, Column("board_id", Key(), primary_key=True))
    rekeyed = context.create_table(
        "projects",
        Column("project_id", Key(), nullable=False),
        Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
        Column("name", LongText(), nullable=False),
        Column("data", LongText(), nullable=False),
        Column("board_id", Key(), ForeignKey("boards.board_id", ondelete="RESTRICT"), nullable=False, unique=True),
        Column("revision", BigInt(), nullable=False, server_default=default(1)),
        inserted_at(),
        updated_at(),
        Column("minimum_canvas_schema_version", BigInt(), nullable=False, server_default=default(2)),
        PrimaryKeyConstraint("project_id"),
        CheckConstraint("minimum_canvas_schema_version >= 1", name="minimum_canvas_schema_version"),
    )
    old = Table(_PER_ACCOUNT_TABLE, MetaData(), autoload_with=context.conn)
    names = [entry.name for entry in rekeyed.columns]
    context.conn.execute(rekeyed.insert().from_select(names, select(*(old.c[name] for name in names))))
    context.op.drop_table(_PER_ACCOUNT_TABLE)


def _create_user_index(context: PortableMigrationContext) -> None:
    if "idx_projects_user_id" not in {index["name"] for index in inspect(context.conn).get_indexes("projects")}:
        context.op.create_index("idx_projects_user_id", "projects", ["user_id"])


def _global_project_ids(context: PortableMigrationContext) -> None:
    if not _is_keyed_by_project_id(context):
        _rekey_duplicates(context)
    _clear_unresolvable_memberships(context)

    if not _is_keyed_by_project_id(context):
        if context.conn.dialect.name == "sqlite":
            _rebuild_projects_on_sqlite(context)
        else:
            # The per-account key is also the index the `user_id` foreign key relies on, so the replacement comes
            # first. One statement then swaps the key, so a failed run never leaves the table without one.
            _create_user_index(context)
            context.conn.exec_driver_sql("ALTER TABLE projects DROP PRIMARY KEY, ADD PRIMARY KEY (project_id)")
    _create_user_index(context)


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_10_global_project_ids",
        depends_on="2026_10_10_board_roles",
        callback=_global_project_ids,
    )
