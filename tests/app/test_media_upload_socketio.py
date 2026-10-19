from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from invokeai.app.api.sockets import SocketIO
from invokeai.app.services.board_access.board_access_common import BoardAudience
from invokeai.app.services.events.events_common import ImageUploadedEvent, MediaUploadedEventBase, VideoUploadedEvent
from invokeai.app.services.image_records.image_records_common import ImageCategory


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _patch_services(monkeypatch: pytest.MonkeyPatch, multiuser: bool, audience: Optional[BoardAudience]) -> MagicMock:
    """The routing reads the multiuser setting and asks the board access service who sees the board."""
    board_access = MagicMock()
    board_access.audience.return_value = audience
    invoker = SimpleNamespace(
        services=SimpleNamespace(configuration=SimpleNamespace(multiuser=multiuser), board_access=board_access)
    )
    monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))
    return board_access


def _image_event(board_id: Optional[str] = "board-1") -> ImageUploadedEvent:
    return ImageUploadedEvent.build(
        image_name="upload.png", image_category=ImageCategory.USER, user_id="user-1", board_id=board_id
    )


async def _route(
    monkeypatch: pytest.MonkeyPatch,
    event: MediaUploadedEventBase,
    multiuser: bool = True,
    audience: Optional[BoardAudience] = None,
) -> tuple[list, MagicMock]:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()
    board_access = _patch_services(monkeypatch, multiuser, audience)

    await socketio._handle_media_event((event.__event_name__, event))

    return [call.kwargs for call in socketio._sio.emit.await_args_list], board_access


def test_upload_events_are_registered_with_the_socket_handler() -> None:
    """Routing tests below call the handler directly; this pins the wiring they skip."""
    SocketIO(FastAPI())

    from fastapi_events.handlers.local import local_handler

    for event_name in ("image_uploaded", "video_uploaded"):
        assert any(
            getattr(handler, "__func__", handler).__name__ == "_handle_media_event"
            for handler in local_handler._registry.get(event_name, [])
        ), event_name


@pytest.mark.anyio
async def test_single_user_upload_goes_to_the_admin_room(monkeypatch: pytest.MonkeyPatch) -> None:
    emits, _ = await _route(monkeypatch, _image_event(), multiuser=False)

    assert [emit["room"] for emit in emits] == ["admin"]
    assert emits[0]["event"] == "image_uploaded"
    assert emits[0]["data"]["image_name"] == "upload.png"


@pytest.mark.anyio
async def test_an_upload_to_no_board_reaches_only_the_uploader_and_admins(monkeypatch: pytest.MonkeyPatch) -> None:
    emits, board_access = await _route(monkeypatch, _image_event(board_id=None))

    assert [emit["room"] for emit in emits] == [["user:user-1", "admin"]]
    board_access.audience.assert_not_called()


@pytest.mark.anyio
async def test_a_restricted_board_upload_reaches_its_owner_and_members(monkeypatch: pytest.MonkeyPatch) -> None:
    """An admin may upload into another account's board; its owner and members must see it arrive."""
    audience = BoardAudience(everyone=False, user_ids=("owner-2", "member-1"))

    emits, board_access = await _route(monkeypatch, _image_event(), audience=audience)

    assert [emit["room"] for emit in emits] == [["user:user-1", "admin", "user:owner-2", "user:member-1"]]
    board_access.audience.assert_called_once_with("board-1")


@pytest.mark.anyio
async def test_an_upload_to_a_board_everyone_can_open_is_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    emits, _ = await _route(monkeypatch, _image_event(), audience=BoardAudience(everyone=True, user_ids=()))

    assert len(emits) == 1
    assert "room" not in emits[0]


@pytest.mark.anyio
async def test_an_upload_to_a_board_deleted_meanwhile_reaches_only_the_uploader_and_admins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emits, _ = await _route(monkeypatch, _image_event(), audience=None)

    assert [emit["room"] for emit in emits] == [["user:user-1", "admin"]]


@pytest.mark.anyio
async def test_video_uploads_route_like_image_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    event = VideoUploadedEvent.build(
        video_name="upload.mp4", video_category=ImageCategory.USER, user_id="user-1", board_id="board-1"
    )

    emits, _ = await _route(monkeypatch, event, audience=BoardAudience(everyone=False, user_ids=("user-1",)))

    assert [(emit["event"], emit["room"]) for emit in emits] == [("video_uploaded", ["user:user-1", "admin"])]


@pytest.mark.anyio
async def test_an_upload_whose_board_cannot_be_looked_up_reaches_only_the_uploader_and_admins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storage error must not drop the event, nor announce it to anyone who may not see the board."""
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()
    board_access = _patch_services(monkeypatch, multiuser=True, audience=None)
    board_access.audience.side_effect = RuntimeError("database is locked")

    event = _image_event()
    await socketio._handle_media_event((event.__event_name__, event))

    assert [call.kwargs["room"] for call in socketio._sio.emit.await_args_list] == [["user:user-1", "admin"]]
