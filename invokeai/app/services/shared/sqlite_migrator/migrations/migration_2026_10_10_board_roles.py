"""Share boards by role: members with a role each, and the role everyone else gets.

`board_members` holds the accounts a board is shared with and their role (viewer, contributor or editor); the
owner is never a member. `boards.general_access` is the role every other account gets: none ('restricted'),
'viewer' or 'contributor'. They replace `board_visibility`, the `is_public` flag it superseded, and
`shared_boards`, which no code ever wrote and whose `can_edit` nothing ever read.

Visibility maps to general access: private to restricted, shared to viewer, public to contributor. Each
`shared_boards` row becomes a viewer member, whatever its `can_edit`, unless it names the board's own owner. A
board without an owner (`user_id` NULL, as some left by a deleted account are) keeps every share.
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Table,
    case,
    column,
    exists,
    inspect,
    literal,
    null,
    or_,
    select,
    table,
)

from invokeai.app.services.shared.database.schema.metadata import USER_ID_LENGTH, default, inserted_at
from invokeai.app.services.shared.database.types import Key, LongText
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

_GENERAL_ACCESS_CHECK = "general_access IN ('restricted', 'viewer', 'contributor')"

_boards = table("boards", column("board_id"), column("user_id"), column("board_visibility"), column("general_access"))
_shared_boards = table("shared_boards", column("board_id"), column("user_id"), column("shared_at"))
_board_members = table(
    "board_members", column("board_id"), column("user_id"), column("role"), column("granted_by"), column("granted_at")
)


def _create_board_members(context: PortableMigrationContext) -> None:
    # The tables its foreign keys name, as far as the CREATE TABLE needs them.
    Table("boards", context.metadata, Column("board_id", Key(), primary_key=True))
    Table("users", context.metadata, Column("user_id", Key(USER_ID_LENGTH), primary_key=True))
    context.create_table(
        "board_members",
        Column("board_id", Key(), ForeignKey("boards.board_id", ondelete="CASCADE"), primary_key=True),
        Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True),
        Column("role", LongText(), nullable=False),
        Column("granted_by", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="SET NULL")),
        inserted_at("granted_at"),
        CheckConstraint("role IN ('viewer', 'contributor', 'editor')", name="role"),
    )


def _copy_shared_boards(context: PortableMigrationContext) -> None:
    """Every share becomes a viewer member. Rows already copied by an interrupted run are skipped."""
    already_member = exists().where(
        _board_members.c.board_id == _shared_boards.c.board_id, _board_members.c.user_id == _shared_boards.c.user_id
    )
    shares = (
        select(
            _shared_boards.c.board_id, _shared_boards.c.user_id, literal("viewer"), null(), _shared_boards.c.shared_at
        )
        .select_from(_shared_boards.join(_boards, _boards.c.board_id == _shared_boards.c.board_id))
        # `!=` alone is NULL, and so false, for a board without an owner.
        .where(or_(_boards.c.user_id.is_(None), _shared_boards.c.user_id != _boards.c.user_id), ~already_member)
    )
    context.conn.execute(
        _board_members.insert().from_select(["board_id", "user_id", "role", "granted_by", "granted_at"], shares)
    )


def _board_roles(context: PortableMigrationContext) -> None:
    inspector = inspect(context.conn)
    tables = set(inspector.get_table_names())

    if "board_members" not in tables:
        _create_board_members(context)
    if "shared_boards" in tables:
        _copy_shared_boards(context)

    columns = {entry["name"] for entry in inspect(context.conn).get_columns("boards")}
    if "general_access" not in columns:
        context.op.add_column(
            "boards", Column("general_access", LongText(), nullable=False, server_default=default("restricted"))
        )
    if "board_visibility" in columns:
        # Repeated by a run that resumes before the column is dropped, with the same result.
        context.conn.execute(
            _boards.update().values(
                general_access=case(
                    (_boards.c.board_visibility == "shared", "viewer"),
                    (_boards.c.board_visibility == "public", "contributor"),
                    else_="restricted",
                )
            )
        )

    indexes = {index["name"] for index in inspect(context.conn).get_indexes("boards")}
    for name in ("idx_boards_board_visibility", "idx_boards_is_public"):
        if name in indexes:
            context.op.drop_index(name, table_name="boards")

    columns = {entry["name"] for entry in inspect(context.conn).get_columns("boards")}
    has_check = "ck_boards_general_access" in {
        check["name"] for check in inspect(context.conn).get_check_constraints("boards")
    }
    dropped = [name for name in ("board_visibility", "is_public") if name in columns]
    if dropped or not has_check:
        # One rebuild on SQLite for every change; plain ALTERs on a server.
        with context.op.batch_alter_table("boards") as batch:
            for name in dropped:
                batch.drop_column(name)
            if not has_check:
                batch.create_check_constraint("general_access", _GENERAL_ACCESS_CHECK)

    if "idx_board_members_user_id" not in {
        index["name"] for index in inspect(context.conn).get_indexes("board_members")
    }:
        context.op.create_index("idx_board_members_user_id", "board_members", ["user_id"])
    if "shared_boards" in set(inspect(context.conn).get_table_names()):
        context.op.drop_table("shared_boards")


def build_migration() -> PortableMigration:
    return PortableMigration(
        id="2026_10_10_board_roles",
        depends_on="2026_10_08_add_board_project_membership",
        callback=_board_roles,
    )
