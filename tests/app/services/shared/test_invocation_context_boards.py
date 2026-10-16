from unittest.mock import MagicMock, create_autospec

import pytest

from invokeai.app.services.board_access.board_access_common import (
    BoardAccessDeniedError,
    BoardAction,
    BoardViewer,
    ItemAccessDeniedError,
    ItemAction,
)
from invokeai.app.services.board_records.board_records_common import BoardRecordOrderBy
from invokeai.app.services.boards.boards_base import BoardServiceABC
from invokeai.app.services.shared.invocation_context import BoardsInterface
from invokeai.app.services.shared.pagination import SQLiteDirection


def _make_interface(multiuser: bool, user: MagicMock | None) -> tuple[BoardsInterface, MagicMock]:
    services = MagicMock()
    # Autospec so a call that doesn't match the real BoardService signature fails like it does at runtime.
    services.boards = create_autospec(BoardServiceABC, instance=True)
    services.configuration.multiuser = multiuser
    services.users.get.return_value = user
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    return BoardsInterface(services, data), services


def test_get_all_runs_as_admin_in_single_user_mode() -> None:
    boards, services = _make_interface(multiuser=False, user=None)

    boards.get_all()

    services.boards.get_all.assert_called_once_with(
        "queue-user", True, order_by=BoardRecordOrderBy.CreatedAt, direction=SQLiteDirection.Descending
    )
    services.users.get.assert_not_called()


@pytest.mark.parametrize(
    ("user", "expected_is_admin"),
    [
        (MagicMock(is_admin=True, is_active=True), True),
        (MagicMock(is_admin=False, is_active=True), False),
    ],
)
def test_get_all_uses_queue_user_role_in_multiuser_mode(user: MagicMock | None, expected_is_admin: bool) -> None:
    boards, services = _make_interface(multiuser=True, user=user)

    boards.get_all()

    services.users.get.assert_called_once_with("queue-user")
    services.boards.get_all.assert_called_once_with(
        "queue-user", expected_is_admin, order_by=BoardRecordOrderBy.CreatedAt, direction=SQLiteDirection.Descending
    )


@pytest.mark.parametrize(
    "user", [MagicMock(is_admin=True, is_active=False), MagicMock(is_admin=False, is_active=False), None]
)
def test_get_all_rejects_a_deactivated_or_deleted_queue_user_in_multiuser_mode(user: MagicMock | None) -> None:
    """An account that no longer exists keeps no queue-time privileges, exactly as a deactivated one."""
    boards, services = _make_interface(multiuser=True, user=user)

    with pytest.raises(PermissionError):
        boards.get_all()

    services.boards.get_all.assert_not_called()


ACTIVE_USER = MagicMock(is_admin=False, is_active=True)
QUEUE_USER = BoardViewer(user_id="queue-user", is_admin=False)


def test_a_board_a_node_reads_is_checked_and_described_to_the_queue_user() -> None:
    boards, services = _make_interface(multiuser=True, user=ACTIVE_USER)

    boards.get_dto("board")

    services.board_access.require.assert_called_once_with("board", "queue-user", False, BoardAction.Read)
    services.boards.get_dto.assert_called_once_with("board", viewer=QUEUE_USER)


@pytest.mark.parametrize("read", ["get_dto", "get_all_image_names_for_board"])
def test_a_node_cannot_read_a_board_its_user_cannot(read: str) -> None:
    boards, services = _make_interface(multiuser=True, user=ACTIVE_USER)
    services.board_access.require.side_effect = BoardAccessDeniedError("board", BoardAction.Read)

    with pytest.raises(PermissionError, match="not authorized"):
        getattr(boards, read)("board")

    services.boards.get_dto.assert_not_called()
    services.board_images.get_all_board_image_names_for_board.assert_not_called()


def test_a_node_adds_only_its_users_own_images_to_boards_they_can_add_to() -> None:
    boards, services = _make_interface(multiuser=True, user=ACTIVE_USER)

    boards.add_image_to_board("board", "image.png")
    services.board_access.require.assert_called_once_with("board", "queue-user", False, BoardAction.AddMedia)
    services.board_access.require_image.assert_called_once_with("image.png", "queue-user", False, ItemAction.Edit)
    services.board_images.add_image_to_board.assert_called_once_with("board", "image.png")

    services.board_images.add_image_to_board.reset_mock()
    services.board_access.require_image.side_effect = ItemAccessDeniedError("image.png", ItemAction.Edit)
    with pytest.raises(PermissionError, match="not authorized"):
        boards.add_image_to_board("board", "image.png")
    services.board_images.add_image_to_board.assert_not_called()


def test_a_deactivated_queue_user_cannot_create_or_read_boards() -> None:
    boards, services = _make_interface(multiuser=True, user=MagicMock(is_admin=True, is_active=False))

    with pytest.raises(PermissionError):
        boards.create("New")
    with pytest.raises(PermissionError):
        boards.get_dto("board")

    services.boards.create.assert_not_called()
    services.boards.get_dto.assert_not_called()
