"""What each role may do to a board and to the items on it, decided over a real database."""

from typing import Optional

import pytest
from sqlalchemy import insert

from invokeai.app.services.board_access.board_access_common import (
    BoardAccess,
    BoardAccessDeniedError,
    BoardAction,
    BoardAudience,
    BoardRole,
    GeneralAccess,
    ItemAccessDeniedError,
    ItemAction,
)
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import board_images, board_members
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.projects import projects
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.pagination import SQLiteDirection

OWNER, OTHER = "owner", "other"

# What each role may do, written out from the role table rather than read from the code under test.
EXPECTED: dict[BoardRole, set[BoardAction]] = {
    BoardRole.Viewer: {BoardAction.Read},
    BoardRole.Contributor: {BoardAction.Read, BoardAction.AddMedia},
    BoardRole.Editor: {
        BoardAction.Read,
        BoardAction.AddMedia,
        BoardAction.ManageItems,
        BoardAction.Rename,
        BoardAction.SetCover,
        BoardAction.Archive,
    },
    BoardRole.Owner: set(BoardAction),
}


@pytest.mark.parametrize("role", list(BoardRole))
def test_each_role_allows_exactly_its_actions(role: BoardRole) -> None:
    access = BoardAccess(role=role, is_admin=False, is_inbox=False)

    assert set(access.allowed_actions()) == EXPECTED[role]


def test_admins_may_do_everything_but_manage_an_inbox_apart_from_its_project() -> None:
    assert set(BoardAccess(role=None, is_admin=True, is_inbox=False).allowed_actions()) == set(BoardAction)
    assert set(BoardAccess(role=None, is_admin=True, is_inbox=True).allowed_actions()) == {
        BoardAction.Read,
        BoardAction.AddMedia,
        BoardAction.ManageItems,
        BoardAction.SetCover,
    }


def test_admin_override_is_reported_only_when_admin_rights_add_to_the_role() -> None:
    assert BoardAccess(role=BoardRole.Viewer, is_admin=True, is_inbox=False).admin_override
    assert not BoardAccess(role=BoardRole.Owner, is_admin=True, is_inbox=False).admin_override
    assert not BoardAccess(role=BoardRole.Viewer, is_admin=False, is_inbox=False).admin_override


@pytest.fixture
def board_id(mock_invoker: Invoker, mock_sqlite_database: Database) -> str:
    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(
            insert(users),
            [{"user_id": user, "email": f"{user}@test.com", "password_hash": "-"} for user in (OWNER, OTHER)],
        )
    return mock_invoker.services.board_records.save("Board", OWNER).board_id


def _grant(database: Database, board_id: str, role: BoardRole) -> None:
    with database.begin(write=True) as conn:
        conn.execute(insert(board_members).values(board_id=board_id, user_id=OTHER, role=role.value))


def _set_general_access(invoker: Invoker, board_id: str, general_access: GeneralAccess) -> None:
    invoker.services.board_records.update(board_id, BoardChanges(general_access=general_access))


@pytest.mark.parametrize(
    ("member_role", "general_access", "expected"),
    [
        (None, GeneralAccess.Restricted, None),
        (None, GeneralAccess.Viewer, BoardRole.Viewer),
        (BoardRole.Viewer, GeneralAccess.Contributor, BoardRole.Contributor),
        (BoardRole.Editor, GeneralAccess.Viewer, BoardRole.Editor),
    ],
)
def test_a_caller_gets_the_higher_of_their_membership_and_general_access(
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    board_id: str,
    member_role: Optional[BoardRole],
    general_access: GeneralAccess,
    expected: Optional[BoardRole],
) -> None:
    if member_role is not None:
        _grant(mock_sqlite_database, board_id, member_role)
    _set_general_access(mock_invoker, board_id, general_access)

    assert mock_invoker.services.board_access.get(board_id, OTHER, False).role == expected
    assert mock_invoker.services.board_access.get(board_id, OWNER, False).role == BoardRole.Owner


def test_require_refuses_what_the_role_does_not_allow_and_tells_a_missing_board_apart(
    mock_invoker: Invoker, mock_sqlite_database: Database, board_id: str
) -> None:
    access = mock_invoker.services.board_access

    with pytest.raises(BoardAccessDeniedError):
        access.require(board_id, OTHER, False, BoardAction.Read)
    _grant(mock_sqlite_database, board_id, BoardRole.Viewer)
    assert access.require(board_id, OTHER, False, BoardAction.Read).role is BoardRole.Viewer
    with pytest.raises(BoardAccessDeniedError):
        access.require(board_id, OTHER, False, BoardAction.AddMedia)
    with pytest.raises(BoardRecordNotFoundException):
        access.require("missing", OTHER, False, BoardAction.Read)


def test_a_projects_inbox_is_managed_through_its_project(
    mock_invoker: Invoker, mock_sqlite_database: Database, board_id: str
) -> None:
    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(insert(projects).values(project_id="p", user_id=OWNER, name="P", data="{}", board_id=board_id))

    inbox = mock_invoker.services.board_access.get(board_id, OWNER, False)

    assert inbox.is_inbox
    assert set(inbox.allowed_actions()) == {
        BoardAction.Read,
        BoardAction.AddMedia,
        BoardAction.ManageItems,
        BoardAction.SetCover,
    }


def _put_image(database: Database, name: str, owner: str, board_id: Optional[str]) -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(images).values(
                image_name=name, image_origin="internal", image_category="general", width=1, height=1, user_id=owner
            )
        )
        if board_id is not None:
            conn.execute(insert(board_images).values(board_id=board_id, image_name=name))


def _allowed_item_actions(invoker: Invoker, name: str, user_id: str, is_admin: bool = False) -> set[ItemAction]:
    allowed = set()
    for action in ItemAction:
        try:
            invoker.services.board_access.require_image(name, user_id, is_admin, action)
        except ItemAccessDeniedError:
            continue
        allowed.add(action)
    return allowed


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (None, set()),
        (BoardRole.Viewer, {ItemAction.Read}),
        (BoardRole.Contributor, {ItemAction.Read}),
        (BoardRole.Editor, {ItemAction.Read, ItemAction.RemoveFromBoard, ItemAction.Star}),
    ],
)
def test_someone_elses_image_is_managed_by_role_and_deleted_by_no_member(
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    board_id: str,
    role: Optional[BoardRole],
    expected: set[ItemAction],
) -> None:
    _put_image(mock_sqlite_database, "theirs.png", OWNER, board_id)
    if role is not None:
        _grant(mock_sqlite_database, board_id, role)

    assert _allowed_item_actions(mock_invoker, "theirs.png", OTHER) == expected


def test_an_items_owner_keeps_every_right_to_it_on_a_board_they_cannot_see(
    mock_invoker: Invoker, mock_sqlite_database: Database, board_id: str
) -> None:
    _put_image(mock_sqlite_database, "mine.png", OTHER, board_id)

    assert _allowed_item_actions(mock_invoker, "mine.png", OTHER) == set(ItemAction)
    # The board's owner only manages it there.
    assert _allowed_item_actions(mock_invoker, "mine.png", OWNER) == {
        ItemAction.Read,
        ItemAction.RemoveFromBoard,
        ItemAction.Star,
    }
    assert _allowed_item_actions(mock_invoker, "mine.png", OWNER, is_admin=True) == set(ItemAction)


def test_an_image_on_no_board_is_its_owners_alone(
    mock_invoker: Invoker, mock_sqlite_database: Database, board_id: str
) -> None:
    _put_image(mock_sqlite_database, "loose.png", OWNER, None)

    assert _allowed_item_actions(mock_invoker, "loose.png", OTHER) == set()


def test_the_audience_is_everyone_or_the_owner_and_members(
    mock_invoker: Invoker, mock_sqlite_database: Database, board_id: str
) -> None:
    access = mock_invoker.services.board_access
    _grant(mock_sqlite_database, board_id, BoardRole.Viewer)

    assert access.audience(board_id) == BoardAudience(everyone=False, user_ids=(OWNER, OTHER))
    _set_general_access(mock_invoker, board_id, GeneralAccess.Viewer)
    assert access.audience(board_id) == BoardAudience(everyone=True, user_ids=())
    assert access.audience("missing") is None


@pytest.mark.parametrize("member_role", [None, BoardRole.Viewer, BoardRole.Editor])
@pytest.mark.parametrize("general_access", list(GeneralAccess))
def test_listings_show_exactly_the_boards_the_service_lets_a_caller_read(
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    board_id: str,
    member_role: Optional[BoardRole],
    general_access: GeneralAccess,
) -> None:
    """The SQL that filters listings and the rules that decide single boards are two copies of one rule."""
    if member_role is not None:
        _grant(mock_sqlite_database, board_id, member_role)
    _set_general_access(mock_invoker, board_id, general_access)

    listed = {
        board.board_id
        for board in mock_invoker.services.board_records.get_all(
            OTHER, False, BoardRecordOrderBy.CreatedAt, SQLiteDirection.Descending, False
        )
    }
    readable = mock_invoker.services.board_access.get(board_id, OTHER, False).allows(BoardAction.Read)

    assert (board_id in listed) == readable


def test_a_membership_gives_its_role_to_its_member_alone(
    mock_invoker: Invoker, mock_sqlite_database: Database, board_id: str
) -> None:
    """With members on the board, an account that is not one gets nothing, and each member gets their own role."""
    third = "third"
    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(insert(users).values(user_id=third, email="third@test.com", password_hash="-"))
        conn.execute(insert(board_members).values(board_id=board_id, user_id=OTHER, role=BoardRole.Editor.value))
    access = mock_invoker.services.board_access

    assert access.get(board_id, OTHER, False).role is BoardRole.Editor
    assert access.get(board_id, third, False).role is None
    with pytest.raises(BoardAccessDeniedError):
        access.require(board_id, third, False, BoardAction.Read)

    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(insert(board_members).values(board_id=board_id, user_id=third, role=BoardRole.Viewer.value))
    roles = {
        user: {dto.board_id: dto.my_role for dto in _listing(mock_invoker, user)}[board_id] for user in (OTHER, third)
    }
    assert roles == {OTHER: BoardRole.Editor, third: BoardRole.Viewer}


def _listing(invoker: Invoker, user_id: str) -> list:
    return invoker.services.boards.get_all(user_id, False, BoardRecordOrderBy.CreatedAt, SQLiteDirection.Descending)


def test_a_caller_sees_only_their_own_projects_among_media_provenance(
    mock_invoker: Invoker, mock_sqlite_database: Database, board_id: str
) -> None:
    """Media carry the project they were made in; a viewer learns it only for their own projects, an admin for all."""
    others_inbox = mock_invoker.services.board_records.save("Their inbox", OTHER).board_id
    with mock_sqlite_database.begin(write=True) as conn:
        conn.execute(insert(projects).values(project_id="mine", user_id=OWNER, name="M", data="{}", board_id=board_id))
        conn.execute(
            insert(projects).values(project_id="theirs", user_id=OTHER, name="T", data="{}", board_id=others_inbox)
        )
    access = mock_invoker.services.board_access
    seen = ["mine", "theirs", "gone"]

    assert access.visible_project_ids(seen, OWNER, False) == {"mine"}
    assert access.visible_project_ids(seen, OTHER, False) == {"theirs"}
    assert access.visible_project_ids(seen, "third", False) == set()
    assert access.visible_project_ids(seen, "third", True) == set(seen)
    assert access.visible_project_ids([], OWNER, False) == set()
