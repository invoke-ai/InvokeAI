"""Router contracts for polymorphic gallery filtering."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.gallery.gallery_common import GalleryItemNames, GalleryItemNamesResult
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.urls.urls_default import LocalUrlService
from tests.app.routers.conftest import _auth, _create_board


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker: Invoker) -> None:
        self.invoker = invoker


def _prepare_gallery_router_test(monkeypatch: Any, mock_invoker: Invoker) -> tuple[MagicMock, MagicMock, MagicMock]:
    dependencies = MockApiDependencies(mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.gallery.ApiDependencies", dependencies)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", dependencies)

    list_items = MagicMock(return_value=OffsetPaginatedResults(items=[], offset=0, limit=10, total=0))
    list_item_names = MagicMock(return_value=GalleryItemNamesResult(items=[], starred_count=0, total_count=0))
    get_item_names = MagicMock(return_value=GalleryItemNames(item_names=[], starred_count=0, total_count=0))
    monkeypatch.setattr(mock_invoker.services.gallery, "list_items", list_items)
    monkeypatch.setattr(mock_invoker.services.gallery, "list_item_names", list_item_names)
    monkeypatch.setattr(mock_invoker.services.gallery, "get_item_names", get_item_names)
    return list_items, list_item_names, get_item_names


@pytest.mark.parametrize(
    ("path", "service_index"),
    [("/api/v1/gallery/items/", 0), ("/api/v1/gallery/items/names", 1), ("/api/v1/gallery/item_names", 2)],
)
def test_gallery_list_endpoints_parse_and_forward_created_ranges(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str, service_index: int
) -> None:
    service_calls = _prepare_gallery_router_test(monkeypatch, mock_invoker)

    response = client.get(path, params={"created_from": "2026-07-01", "created_to": "2026-07-15"})

    assert response.status_code == 200
    kwargs = service_calls[service_index].call_args.kwargs
    assert kwargs["created_from"] == "2026-07-01"
    assert kwargs["created_to"] == "2026-07-15"


@pytest.mark.parametrize(
    "path", ["/api/v1/gallery/items/", "/api/v1/gallery/items/names", "/api/v1/gallery/item_names"]
)
@pytest.mark.parametrize("param", ["created_from", "created_to"])
def test_gallery_list_endpoints_reject_invalid_created_range_dates(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str, param: str
) -> None:
    _prepare_gallery_router_test(monkeypatch, mock_invoker)

    response = client.get(path, params={param: "2026-02-31"})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("path", "service_index"),
    [("/api/v1/gallery/items/", 0), ("/api/v1/gallery/items/names", 1), ("/api/v1/gallery/item_names", 2)],
)
@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False), (None, None)])
def test_gallery_list_endpoints_forward_starred_filter(
    monkeypatch: Any,
    mock_invoker: Invoker,
    client: TestClient,
    path: str,
    service_index: int,
    value: str | None,
    expected: bool | None,
) -> None:
    service_calls = _prepare_gallery_router_test(monkeypatch, mock_invoker)

    response = client.get(path, params={} if value is None else {"starred": value})

    assert response.status_code == 200
    assert service_calls[service_index].call_args.kwargs["starred"] is expected


@pytest.mark.parametrize(
    "path", ["/api/v1/gallery/items/", "/api/v1/gallery/items/names", "/api/v1/gallery/item_names"]
)
def test_gallery_list_endpoints_reject_non_boolean_starred(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str
) -> None:
    _prepare_gallery_router_test(monkeypatch, mock_invoker)

    response = client.get(path, params={"starred": "maybe"})

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("path", "service_index"),
    [("/api/v1/gallery/items/", 0), ("/api/v1/gallery/items/names", 1), ("/api/v1/gallery/item_names", 2)],
)
@pytest.mark.parametrize(("value", "expected"), [("image", "image"), ("video", "video"), (None, None)])
def test_gallery_list_endpoints_forward_media_kind(
    monkeypatch: Any,
    mock_invoker: Invoker,
    client: TestClient,
    path: str,
    service_index: int,
    value: str | None,
    expected: str | None,
) -> None:
    service_calls = _prepare_gallery_router_test(monkeypatch, mock_invoker)

    response = client.get(path, params={} if value is None else {"media_kind": value})

    assert response.status_code == 200
    assert service_calls[service_index].call_args.kwargs["media_kind"] == expected


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/gallery/items/",
        "/api/v1/gallery/items/names",
        "/api/v1/gallery/item_names",
        "/api/v1/gallery/items/location?kind=image&name=a.png",
        "/api/v1/virtual_boards/by_date/2026-05-18/item_names",
    ],
)
def test_gallery_listings_reject_an_unknown_media_kind(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str
) -> None:
    _prepare_gallery_router_test(monkeypatch, mock_invoker)

    response = client.get(path, params={"media_kind": "audio"})

    assert response.status_code == 422


def _save_general_image(mock_invoker: Invoker, image_name: str, user_id: str) -> None:
    mock_invoker.services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=10,
        height=10,
        has_workflow=False,
        user_id=user_id,
    )


@pytest.mark.parametrize(
    ("path", "names_of"),
    [
        ("/api/v1/gallery/items/", lambda body: {item["name"] for item in body["items"]}),
        ("/api/v1/gallery/items/names", lambda body: {item["name"] for item in body["items"]}),
        ("/api/v1/gallery/item_names", lambda body: set(body["item_names"])),
    ],
)
def test_the_all_boards_scope_lists_readable_items_without_a_board_check(
    monkeypatch: Any,
    mock_invoker: Invoker,
    client: TestClient,
    user1_token: str,
    user2_token: str,
    path: str,
    names_of: Callable[[Any], set[str]],
) -> None:
    def no_board_check(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("'all' is a scope, not a board")

    monkeypatch.setattr("invokeai.app.api.routers.gallery._assert_board_read_access", no_board_check)
    # A page carries each item's URLs.
    monkeypatch.setattr(mock_invoker.services, "urls", LocalUrlService())
    user1, user2 = (mock_invoker.services.users.get_by_email(f"user{n}@test.com") for n in (1, 2))
    assert user1 is not None and user2 is not None
    own_board, private_board = _create_board(client, user1_token, "Mine"), _create_board(client, user2_token, "Theirs")
    for name, owner, board_id in (
        ("own-unboarded.png", user1.user_id, None),
        ("own-boarded.png", user1.user_id, own_board),
        ("other-unboarded.png", user2.user_id, None),
        ("other-private.png", user2.user_id, private_board),
    ):
        _save_general_image(mock_invoker, name, owner)
        if board_id is not None:
            mock_invoker.services.board_image_records.add_image_to_board(board_id, name)

    response = client.get(path, params={"board_id": "all"}, headers=_auth(user1_token))
    own = client.get(
        "/api/v1/gallery/items/location",
        params={"board_id": "all", "kind": "image", "name": "own-boarded.png"},
        headers=_auth(user1_token),
    )
    foreign = client.get(
        "/api/v1/gallery/items/location",
        params={"board_id": "all", "kind": "image", "name": "other-private.png"},
        headers=_auth(user1_token),
    )

    assert response.status_code == 200
    assert names_of(response.json()) == {"own-unboarded.png", "own-boarded.png"}
    assert own.status_code == 200 and own.json()["total"] == 2
    assert foreign.status_code == 404
