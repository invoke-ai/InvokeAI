"""Router contracts for polymorphic gallery filtering."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.gallery.gallery_common import GalleryItemNames, GalleryItemNamesResult
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults


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
