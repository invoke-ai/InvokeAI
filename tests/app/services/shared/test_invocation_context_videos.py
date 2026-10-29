from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.board_access.board_access_common import (
    BoardAccessDeniedError,
    BoardAction,
    ItemAccessDeniedError,
    ItemAction,
)
from invokeai.app.services.shared.invocation_context import VideosInterface


def _make_interface(allowed: bool) -> tuple[VideosInterface, MagicMock]:
    """The board access service decides; these tests check how generation asks it and acts on the answer."""
    services = MagicMock()
    services.configuration.multiuser = True
    services.users.get.return_value = MagicMock(is_admin=False)
    if not allowed:
        services.board_access.require.side_effect = BoardAccessDeniedError("board", BoardAction.AddMedia)
        services.board_access.require_image.side_effect = ItemAccessDeniedError("item", ItemAction.Read)
        services.board_access.require_video.side_effect = ItemAccessDeniedError("item", ItemAction.Read)
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    data.queue_item.workflow = None
    data.queue_item.session.graph = None
    data.queue_item.session_id = "session"
    data.queue_item.project_id = "project-1"
    data.invocation.is_intermediate = False
    return VideosInterface(services, data, MagicMock()), services


# The account as `users.get` answers for one that is deactivated, and for one that is gone.
_GONE_ACCOUNTS = [
    pytest.param(MagicMock(is_admin=False, is_active=False), id="deactivated"),
    pytest.param(None, id="deleted"),
]


def test_video_save_rejects_a_board_the_user_cannot_add_to() -> None:
    videos, services = _make_interface(allowed=False)

    with pytest.raises(PermissionError, match="not authorized"):
        videos.save(Path("output.mp4"), width=64, height=64, duration=1.0, board_id="foreign-board")

    services.videos.create.assert_not_called()


def test_video_save_allows_a_board_the_user_can_add_to() -> None:
    videos, services = _make_interface(allowed=True)

    videos.save(Path("output.mp4"), width=64, height=64, duration=1.0, board_id="public-board")

    services.videos.create.assert_called_once()
    services.board_access.require.assert_called_once_with("public-board", "queue-user", False, BoardAction.AddMedia)


def test_video_path_rejects_foreign_private_video() -> None:
    videos, services = _make_interface(allowed=False)
    services.video_records.get_user_id.return_value = "owner"
    services.board_video_records.get_board_for_video.return_value = "foreign-board"

    with pytest.raises(PermissionError, match="not authorized"):
        videos.get_path("private.mp4")

    services.videos.get_path.assert_not_called()


def test_video_dto_allows_foreign_shared_video() -> None:
    videos, services = _make_interface(allowed=True)
    services.video_records.get_user_id.return_value = "owner"
    services.board_video_records.get_board_for_video.return_value = "shared-board"

    videos.get_dto("shared.mp4")

    services.videos.get_dto.assert_called_once_with("shared.mp4")
    services.board_access.require_video.assert_called_once_with("shared.mp4", "queue-user", False, ItemAction.Read)
    services.board_access.require_image.assert_not_called()


@pytest.mark.parametrize("account", _GONE_ACCOUNTS)
def test_video_read_rejects_a_deactivated_or_deleted_queue_user(account: MagicMock | None) -> None:
    """Such an account keeps no queue-time privileges, even for its own videos."""
    videos, services = _make_interface(allowed=False)
    services.users.get.return_value = account
    services.video_records.get_user_id.return_value = "queue-user"

    with pytest.raises(PermissionError, match="not authorized"):
        videos.get_dto("own-video")

    services.videos.get_dto.assert_not_called()


@pytest.mark.parametrize("account", _GONE_ACCOUNTS)
def test_video_save_rejects_a_deactivated_or_deleted_queue_user(account: MagicMock | None) -> None:
    """No output may be saved on behalf of such an account, even uncategorized."""
    videos, services = _make_interface(allowed=False)
    services.users.get.return_value = account

    with pytest.raises(PermissionError, match="not authorized"):
        videos.save(Path("output.mp4"), width=64, height=64, duration=1.0)

    services.videos.create.assert_not_called()


def test_video_save_allows_active_queue_user_without_board() -> None:
    videos, services = _make_interface(allowed=False)
    services.users.get.return_value = MagicMock(is_admin=False, is_active=True)

    videos.save(Path("output.mp4"), width=64, height=64, duration=1.0)

    services.videos.create.assert_called_once()


def test_videos_save_attributes_the_output_to_the_queue_items_project() -> None:
    videos, services = _make_interface(allowed=True)

    videos.save(Path("output.mp4"), width=64, height=64, duration=1.0)

    assert services.videos.create.call_args.kwargs["project_id"] == "project-1"
