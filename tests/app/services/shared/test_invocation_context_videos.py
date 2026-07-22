from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.shared.invocation_context import VideosInterface


def _make_interface(visibility: BoardVisibility, owner_id: str = "owner") -> tuple[VideosInterface, MagicMock]:
    services = MagicMock()
    services.users.get.return_value = MagicMock(is_admin=False)
    services.boards.get_dto.return_value = MagicMock(user_id=owner_id, board_visibility=visibility)
    data = MagicMock()
    data.queue_item.user_id = "queue-user"
    data.queue_item.workflow = None
    data.queue_item.session.graph = None
    data.queue_item.session_id = "session"
    data.invocation.is_intermediate = False
    return VideosInterface(services, data, MagicMock()), services


def test_video_save_rejects_foreign_private_board() -> None:
    videos, services = _make_interface(BoardVisibility.Private)

    with pytest.raises(PermissionError, match="not authorized"):
        videos.save(Path("output.mp4"), width=64, height=64, duration=1.0, board_id="foreign-board")

    services.videos.create.assert_not_called()


@pytest.mark.parametrize("visibility", [BoardVisibility.Public])
def test_video_save_allows_writable_foreign_board(visibility: BoardVisibility) -> None:
    videos, services = _make_interface(visibility)

    videos.save(Path("output.mp4"), width=64, height=64, duration=1.0, board_id="public-board")

    services.videos.create.assert_called_once()


def test_video_save_allows_board_owner() -> None:
    videos, services = _make_interface(BoardVisibility.Private, owner_id="queue-user")

    videos.save(Path("output.mp4"), width=64, height=64, duration=1.0, board_id="owned-board")

    services.videos.create.assert_called_once()


def test_video_path_rejects_foreign_private_video() -> None:
    videos, services = _make_interface(BoardVisibility.Private)
    services.video_records.get_user_id.return_value = "owner"
    services.board_video_records.get_board_for_video.return_value = "foreign-board"

    with pytest.raises(PermissionError, match="not authorized"):
        videos.get_path("private.mp4")

    services.videos.get_path.assert_not_called()


def test_video_dto_allows_foreign_shared_video() -> None:
    videos, services = _make_interface(BoardVisibility.Shared)
    services.video_records.get_user_id.return_value = "owner"
    services.board_video_records.get_board_for_video.return_value = "shared-board"

    videos.get_dto("shared.mp4")

    services.videos.get_dto.assert_called_once_with("shared.mp4")
