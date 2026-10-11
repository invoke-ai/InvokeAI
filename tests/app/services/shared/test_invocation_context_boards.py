from unittest.mock import MagicMock, create_autospec

import pytest

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
        (None, False),
    ],
)
def test_get_all_uses_queue_user_role_in_multiuser_mode(user: MagicMock | None, expected_is_admin: bool) -> None:
    boards, services = _make_interface(multiuser=True, user=user)

    boards.get_all()

    services.users.get.assert_called_once_with("queue-user")
    services.boards.get_all.assert_called_once_with(
        "queue-user", expected_is_admin, order_by=BoardRecordOrderBy.CreatedAt, direction=SQLiteDirection.Descending
    )


@pytest.mark.parametrize("is_admin", [True, False])
def test_get_all_rejects_deactivated_queue_user_in_multiuser_mode(is_admin: bool) -> None:
    boards, services = _make_interface(multiuser=True, user=MagicMock(is_admin=is_admin, is_active=False))

    with pytest.raises(PermissionError):
        boards.get_all()

    services.boards.get_all.assert_not_called()
