"""Router tests for ordinary Gallery item-location lookup."""

import asyncio
from datetime import date
from types import SimpleNamespace
from typing import Any

import anyio.to_thread
import httpx
import pytest
from fastapi import FastAPI, HTTPException, status
from sqlalchemy import update

from invokeai.app.api.auth_dependencies import get_current_user_or_default
from invokeai.app.api.routers import gallery as gallery_router_module
from invokeai.app.api.routers.gallery import get_gallery_item_location
from invokeai.app.services.auth.token_service import TokenData
from invokeai.app.services.gallery.gallery_common import GalleryItemKind, GalleryItemLocation
from invokeai.app.services.image_records.image_records_common import ImageCategory, ImageRecordChanges, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.shared.pagination import SQLiteDirection
from invokeai.app.services.video_records.video_records_common import VideoRecordChanges


def _save_image(invoker: Invoker, name: str, user_id: str, category: ImageCategory = ImageCategory.GENERAL) -> None:
    invoker.services.image_records.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=category,
        width=10,
        height=10,
        has_workflow=False,
        is_intermediate=False,
        user_id=user_id,
    )


def _save_video(invoker: Invoker, name: str, user_id: str) -> None:
    invoker.services.video_records.save(
        video_name=name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=10,
        height=10,
        duration=1.0,
        fps=8.0,
        has_workflow=False,
        is_intermediate=False,
        user_id=user_id,
    )


def _set_created_at(invoker: Invoker, table: str, name_column: str, name: str, created_at: str) -> None:
    media = {"images": images, "videos": videos}[table]
    with invoker.services.database.begin(write=True) as conn:
        conn.execute(update(media).where(media.c[name_column] == name).values(created_at=created_at))


def _locate(
    user_id: str,
    kind: GalleryItemKind,
    name: str,
    **filters: Any,
) -> GalleryItemLocation:
    return get_gallery_item_location(
        current_user=TokenData(user_id=user_id, email=f"{user_id}@test.com", is_admin=False),
        kind=kind,
        name=name,
        origin=filters.get("origin"),
        categories=filters.get("categories"),
        is_intermediate=filters.get("is_intermediate"),
        board_id=filters.get("board_id"),
        order_dir=filters.get("order_dir", SQLiteDirection.Descending),
        starred=filters.get("starred"),
        search_term=filters.get("search_term"),
        created_from=filters.get("created_from"),
        created_to=filters.get("created_to"),
    )


def test_location_forwards_ordinary_filters_and_returns_total_and_index(
    enable_multiuser: Any, mock_invoker: Invoker
) -> None:
    _save_image(mock_invoker, "plain.png", "alice")
    _save_video(mock_invoker, "target.mp4", "alice")
    _set_created_at(mock_invoker, "images", "image_name", "plain.png", "2026-05-02 10:00:00")
    _set_created_at(mock_invoker, "videos", "video_name", "target.mp4", "2026-05-02 11:00:00")
    mock_invoker.services.video_records.update("target.mp4", VideoRecordChanges(starred=True))
    board = mock_invoker.services.board_records.save("Gallery", "alice")
    mock_invoker.services.board_video_records.add_video_to_board(board.board_id, "target.mp4")

    location = _locate(
        "alice",
        GalleryItemKind.VIDEO,
        "target.mp4",
        board_id=board.board_id,
        categories=[ImageCategory.GENERAL],
        is_intermediate=False,
        order_dir=SQLiteDirection.Descending,
        starred=True,
        search_term="2026-05-02",
        created_from=date(2026, 5, 2),
        created_to=date(2026, 5, 2),
    )

    assert location == GalleryItemLocation(kind=GalleryItemKind.VIDEO, name="target.mp4", index=0, total=1)


def test_http_location_validates_query_and_serializes_response(monkeypatch: Any, mock_invoker: Invoker) -> None:
    _save_image(mock_invoker, "http.png", "alice")
    monkeypatch.setattr(gallery_router_module, "ApiDependencies", SimpleNamespace(invoker=mock_invoker))

    async def run_sync_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    # This environment's AnyIO blocking portal cannot wake its event loop. Run this small
    # synchronous handler inline while keeping the ASGI request/validation/serialization path real.
    monkeypatch.setattr(anyio.to_thread, "run_sync", run_sync_inline)

    app = FastAPI()
    app.include_router(gallery_router_module.gallery_router, prefix="/api")
    app.dependency_overrides[get_current_user_or_default] = lambda: TokenData(
        user_id="alice", email="alice@test.com", is_admin=False
    )

    async def make_requests() -> tuple[httpx.Response, httpx.Response]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/api/v1/gallery/items/location",
                params={"kind": "image", "name": "http.png", "order_dir": "ASC"},
            )
            invalid = await client.get("/api/v1/gallery/items/location", params={"name": "http.png"})
        return response, invalid

    response, invalid = asyncio.run(make_requests())

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json() == {"kind": "image", "name": "http.png", "index": 0, "total": 1}
    assert invalid.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_location_order_uses_created_time_without_starred_first(enable_multiuser: Any, mock_invoker: Invoker) -> None:
    _save_image(mock_invoker, "starred.png", "alice")
    _save_image(mock_invoker, "newer.png", "alice")
    _set_created_at(mock_invoker, "images", "image_name", "starred.png", "2026-01-01 10:00:00")
    _set_created_at(mock_invoker, "images", "image_name", "newer.png", "2026-01-02 10:00:00")
    mock_invoker.services.image_records.update("starred.png", ImageRecordChanges(starred=True))

    location = _locate("alice", GalleryItemKind.IMAGE, "newer.png")

    assert location == GalleryItemLocation(kind=GalleryItemKind.IMAGE, name="newer.png", index=0, total=2)


def test_missing_filtered_and_cross_user_targets_share_same_not_found(
    enable_multiuser: Any, mock_invoker: Invoker
) -> None:
    _save_image(mock_invoker, "foreign.png", "bob")

    responses = []
    for user_id, name, filters in [
        ("alice", "missing.png", {}),
        ("alice", "foreign.png", {}),
        ("bob", "foreign.png", {"categories": [ImageCategory.CONTROL]}),
    ]:
        with pytest.raises(HTTPException) as error:
            _locate(user_id, GalleryItemKind.IMAGE, name, **filters)
        responses.append((error.value.status_code, error.value.detail))

    assert responses == [(status.HTTP_404_NOT_FOUND, "Gallery item not found")] * 3


def test_inaccessible_explicit_board_keeps_existing_forbidden_behavior(
    enable_multiuser: Any, mock_invoker: Invoker
) -> None:
    board = mock_invoker.services.board_records.save("Private", "bob")

    with pytest.raises(HTTPException) as error:
        _locate("alice", GalleryItemKind.IMAGE, "anything.png", board_id=board.board_id)

    assert error.value.status_code == status.HTTP_403_FORBIDDEN
    assert error.value.detail == "Not authorized to access this board"
