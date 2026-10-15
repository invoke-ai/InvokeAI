from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from invokeai.app.api.sockets import SocketIO
from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.events.events_common import ImageUploadedEvent, MediaUploadedEventBase, VideoUploadedEvent
from invokeai.app.services.image_records.image_records_common import ImageCategory


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _patch_multiuser(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    invoker = SimpleNamespace(services=SimpleNamespace(configuration=SimpleNamespace(multiuser=enabled)))
    monkeypatch.setattr("invokeai.app.api.dependencies.ApiDependencies", SimpleNamespace(invoker=invoker))


def _image_event(
    board_visibility: BoardVisibility | None,
    board_id: str | None = "board-1",
    board_owner_id: str | None = "user-1",
    shared_user_ids: list[str] | None = None,
) -> ImageUploadedEvent:
    return ImageUploadedEvent.build(
        image_name="upload.png",
        image_category=ImageCategory.USER,
        user_id="user-1",
        board_id=board_id,
        board_owner_id=board_owner_id if board_id else None,
        board_visibility=board_visibility,
        shared_user_ids=shared_user_ids or [],
    )


async def _route(monkeypatch: pytest.MonkeyPatch, event: MediaUploadedEventBase, multiuser: bool) -> list:
    socketio = SocketIO(FastAPI())
    socketio._sio.emit = AsyncMock()
    _patch_multiuser(monkeypatch, multiuser)

    await socketio._handle_media_event((event.__event_name__, event))

    return [call.kwargs for call in socketio._sio.emit.await_args_list]


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
    emits = await _route(monkeypatch, _image_event(BoardVisibility.Private), multiuser=False)

    assert [emit["room"] for emit in emits] == ["admin"]
    assert emits[0]["event"] == "image_uploaded"
    assert emits[0]["data"]["image_name"] == "upload.png"


@pytest.mark.parametrize(
    ("board_visibility", "board_id"),
    [(BoardVisibility.Private, "board-1"), (None, None)],
    ids=["private-board", "no-board"],
)
@pytest.mark.anyio
async def test_private_upload_reaches_only_the_uploader_and_admins(
    monkeypatch: pytest.MonkeyPatch, board_visibility: BoardVisibility | None, board_id: str | None
) -> None:
    emits = await _route(monkeypatch, _image_event(board_visibility, board_id), multiuser=True)

    assert len(emits) == 1
    assert emits[0]["room"] == ["user:user-1", "admin"]


@pytest.mark.anyio
async def test_admin_upload_into_another_users_private_board_reaches_that_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emits = await _route(monkeypatch, _image_event(BoardVisibility.Private, board_owner_id="owner-2"), multiuser=True)

    assert len(emits) == 1
    assert emits[0]["room"] == ["user:user-1", "admin", "user:owner-2"]


@pytest.mark.anyio
async def test_private_board_upload_also_reaches_its_share_recipients(monkeypatch: pytest.MonkeyPatch) -> None:
    """A board shared with named users is readable by them, so their galleries must refresh too."""
    event = _image_event(BoardVisibility.Private, shared_user_ids=["viewer-1", "viewer-2"])

    emits = await _route(monkeypatch, event, multiuser=True)

    assert len(emits) == 1
    assert emits[0]["room"] == ["user:user-1", "admin", "user:viewer-1", "user:viewer-2"]


@pytest.mark.parametrize("board_visibility", [BoardVisibility.Shared, BoardVisibility.Public])
@pytest.mark.anyio
async def test_upload_to_a_board_everyone_can_view_is_broadcast(
    monkeypatch: pytest.MonkeyPatch, board_visibility: BoardVisibility
) -> None:
    emits = await _route(monkeypatch, _image_event(board_visibility), multiuser=True)

    assert len(emits) == 1
    assert "room" not in emits[0]
    assert emits[0]["data"]["board_visibility"] == board_visibility.value


@pytest.mark.anyio
async def test_video_uploads_route_like_image_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    event = VideoUploadedEvent.build(
        video_name="upload.mp4",
        video_category=ImageCategory.USER,
        user_id="user-1",
        board_id="board-1",
        board_owner_id="user-1",
        board_visibility=BoardVisibility.Private,
        shared_user_ids=[],
    )

    emits = await _route(monkeypatch, event, multiuser=True)

    assert [(emit["event"], emit["room"]) for emit in emits] == [("video_uploaded", ["user:user-1", "admin"])]
