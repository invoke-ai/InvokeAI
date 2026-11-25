"""Router contracts for polymorphic gallery filtering."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers._limits import MAX_IMAGE_BATCH_SIZE, MAX_VIDEO_BATCH_SIZE
from invokeai.app.services.gallery.gallery_common import GalleryItemNames, GalleryItemNamesResult
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.schema.boards import board_members
from invokeai.app.services.shared.media_marks import ColorLabel, MarkFilters, MarkFlag
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


_LISTINGS = [
    ("/api/v1/gallery/items/", 0),
    ("/api/v1/gallery/items/names", 1),
    ("/api/v1/gallery/item_names", 2),
    ("/api/v1/virtual_boards/by_date/2026-05-18/item_names", 1),
]


@pytest.mark.parametrize(("path", "service_index"), _LISTINGS)
def test_gallery_listings_forward_mark_filters(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, path: str, service_index: int
) -> None:
    service_calls = _prepare_gallery_router_test(monkeypatch, mock_invoker)
    monkeypatch.setattr("invokeai.app.api.routers.virtual_boards.ApiDependencies", MockApiDependencies(mock_invoker))

    response = client.get(
        path,
        params=[
            ("flags", "pick"),
            ("flags", "none"),
            ("min_rating", "2"),
            ("max_rating", "4"),
            ("color_labels", "red"),
            ("color_labels", "none"),
        ],
    )

    assert response.status_code == 200
    assert service_calls[service_index].call_args.kwargs["marks"] == MarkFilters(
        flags=[MarkFlag.PICK, MarkFlag.NONE], min_rating=2, max_rating=4, color_labels=[ColorLabel.RED, "none"]
    )


@pytest.mark.parametrize(
    "params",
    [
        {"flags": "starred"},
        {"color_labels": "orange"},
        {"min_rating": "0"},
        {"min_rating": "6"},
        {"max_rating": "-1"},
        {"max_rating": "6"},
        {"min_rating": "4", "max_rating": "3"},
        [("flags", "pick")] * 4,
    ],
)
def test_gallery_listings_reject_invalid_mark_filters(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, params: Any
) -> None:
    # Every listing route takes its mark filters from the one `_mark_filters` dependency.
    _prepare_gallery_router_test(monkeypatch, mock_invoker)

    assert client.get("/api/v1/gallery/items/", params=params).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"image_names": ["a.png"], "rating": 6},
        {"image_names": ["a.png"], "rating": -1},
        {"image_names": ["a.png"], "rating": "3"},
        {"image_names": ["a.png"], "flag": "starred"},
        {"image_names": ["a.png"], "color_label": "orange"},
        {"image_names": ["a.png"], "colour_label": "red"},
        {"image_names": ["a.png"]},
        {"image_names": [], "video_names": [], "rating": 1},
        {"image_names": ["a.png"] * (MAX_IMAGE_BATCH_SIZE + 1), "rating": 1},
        {"video_names": ["a.mp4"] * (MAX_VIDEO_BATCH_SIZE + 1), "rating": 1},
        {"image_names": ["a" * 256], "rating": 1},
    ],
)
def test_set_marks_rejects_invalid_requests(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, body: dict[str, Any]
) -> None:
    _prepare_gallery_router_test(monkeypatch, mock_invoker)
    set_marks = MagicMock()
    monkeypatch.setattr(mock_invoker.services.gallery, "set_marks", set_marks)

    assert client.post("/api/v1/gallery/items/marks", json=body).status_code == 422
    set_marks.assert_not_called()


def _save_general_video(mock_invoker: Invoker, video_name: str, user_id: str) -> None:
    mock_invoker.services.video_records.save(
        video_name=video_name,
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


def _add_member(database: Database, board_id: str, user_id: str, role: str) -> None:
    with database.begin(write=True) as conn:
        conn.execute(insert(board_members).values(board_id=board_id, user_id=user_id, role=role))


@pytest.mark.parametrize(("role", "may_mark"), [("viewer", False), ("contributor", False), ("editor", True)])
def test_marking_someone_elses_items_takes_a_managers_role_on_their_board(
    mock_invoker: Invoker,
    mock_sqlite_database: Database,
    client: TestClient,
    user1_token: str,
    user2_token: str,
    role: str,
    may_mark: bool,
) -> None:
    user1, user2 = (mock_invoker.services.users.get_by_email(f"user{n}@test.com") for n in (1, 2))
    assert user1 is not None and user2 is not None
    board_id = _create_board(client, user1_token, "Shared")
    _add_member(mock_sqlite_database, board_id, user2.user_id, role)
    _save_general_image(mock_invoker, "theirs.png", user1.user_id)
    _save_general_video(mock_invoker, "theirs.mp4", user1.user_id)
    mock_invoker.services.board_image_records.add_image_to_board(board_id, "theirs.png")
    mock_invoker.services.board_video_records.add_video_to_board(board_id, "theirs.mp4")

    response = client.post(
        "/api/v1/gallery/items/marks",
        json={"image_names": ["theirs.png"], "video_names": ["theirs.mp4"], "flag": "reject", "rating": 2},
        headers=_auth(user2_token),
    )

    assert response.status_code == 200
    assert response.json() == {
        "updated_image_names": ["theirs.png"] if may_mark else [],
        "updated_video_names": ["theirs.mp4"] if may_mark else [],
        "failed_image_names": [],
        "failed_video_names": [],
        "affected_boards": [board_id] if may_mark else [],
    }
    image, video = (
        mock_invoker.services.image_records.get("theirs.png"),
        mock_invoker.services.video_records.get("theirs.mp4"),
    )
    expected = (2, True) if may_mark else (0, False)
    assert (image.rating, image.rejected) == (video.rating, video.rejected) == expected


def test_a_partial_batch_marks_what_the_caller_may_once_and_skips_the_rest(
    mock_invoker: Invoker, client: TestClient, user1_token: str, user2_token: str
) -> None:
    user1, user2 = (mock_invoker.services.users.get_by_email(f"user{n}@test.com") for n in (1, 2))
    assert user1 is not None and user2 is not None
    board_id = _create_board(client, user1_token, "Mine")
    _save_general_image(mock_invoker, "boarded.png", user1.user_id)
    mock_invoker.services.board_image_records.add_image_to_board(board_id, "boarded.png")
    _save_general_image(mock_invoker, "loose.png", user1.user_id)
    _save_general_image(mock_invoker, "foreign.png", user2.user_id)
    _save_general_video(mock_invoker, "own.mp4", user1.user_id)
    _save_general_video(mock_invoker, "foreign.mp4", user2.user_id)

    response = client.post(
        "/api/v1/gallery/items/marks",
        json={
            "image_names": ["loose.png", "foreign.png", "missing.png", "boarded.png", "loose.png"],
            "video_names": ["foreign.mp4", "own.mp4"],
            "color_label": "green",
        },
        headers=_auth(user1_token),
    )

    assert response.status_code == 200
    assert response.json() == {
        "updated_image_names": ["loose.png", "boarded.png"],
        "updated_video_names": ["own.mp4"],
        "failed_image_names": [],
        "failed_video_names": [],
        "affected_boards": sorted([board_id, "none"]),
    }
    assert mock_invoker.services.image_records.get("foreign.png").color_label is None
    assert mock_invoker.services.video_records.get("foreign.mp4").color_label is None
    assert mock_invoker.services.video_records.get("own.mp4").color_label == ColorLabel.GREEN


def test_a_failed_save_reports_every_markable_item_as_failed(
    monkeypatch: Any, mock_invoker: Invoker, client: TestClient, user1_token: str, user2_token: str
) -> None:
    user1, user2 = (mock_invoker.services.users.get_by_email(f"user{n}@test.com") for n in (1, 2))
    assert user1 is not None and user2 is not None
    _save_general_image(mock_invoker, "own.png", user1.user_id)
    _save_general_image(mock_invoker, "foreign.png", user2.user_id)
    _save_general_video(mock_invoker, "own.mp4", user1.user_id)
    monkeypatch.setattr(mock_invoker.services.gallery, "set_marks", MagicMock(side_effect=OSError("disk I/O error")))

    response = client.post(
        "/api/v1/gallery/items/marks",
        json={"image_names": ["own.png", "foreign.png"], "video_names": ["own.mp4"], "flag": "pick"},
        headers=_auth(user1_token),
    )

    assert response.status_code == 200
    assert response.json() == {
        "updated_image_names": [],
        "updated_video_names": [],
        "failed_image_names": ["own.png"],
        "failed_video_names": ["own.mp4"],
        "affected_boards": [],
    }
