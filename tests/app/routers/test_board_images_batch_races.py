"""Race handling on the board_images batch routes.

Two races found in review, both between a batch loop's read and its write:

- The scoped DELETE in the batch remove can match zero rows when a concurrent session moves the
  image between the DTO read and the write. The row count is the only signal the scope held, and
  the classification depends on where the image went.
- The per-name destination re-check in the batch add can start refusing mid-batch when the
  destination board is revoked or deleted. That refusal is the request's problem, not the
  name's: treated as a skip it answers 201 with empty lists, which the client reads as success
  and clears the user's selection over.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.invoker import Invoker


class MockApiDependencies(ApiDependencies):
    invoker: Invoker

    def __init__(self, invoker) -> None:
        self.invoker = invoker


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _install(monkeypatch: pytest.MonkeyPatch, mock_invoker: Invoker) -> None:
    mock_deps = MockApiDependencies(mock_invoker)
    # Several of these are None on the conftest's mock services; the routes need whole service
    # doubles, installed via monkeypatch so they are restored between tests.
    for name in ("image_moves", "images", "image_records", "board_images", "board_image_records", "board_records"):
        monkeypatch.setattr(mock_invoker.services, name, MagicMock())
    mock_invoker.services.image_moves.is_maintenance_active.return_value = False
    monkeypatch.setattr("invokeai.app.api.routers.board_images.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers._access.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.routers.image_move_maintenance.ApiDependencies", mock_deps)
    monkeypatch.setattr("invokeai.app.api.auth_dependencies.ApiDependencies", mock_deps)


@pytest.mark.parametrize(
    ("now_on_board", "record_exists", "expect_removed", "expect_failed", "expect_boards"),
    [
        # Moved to another board mid-batch: the ask -- off every board -- is not satisfied, and
        # a retry will re-read and re-authorize against the board it actually sits on now.
        ("board-q", True, [], ["raced.png"], []),
        # Concurrently uncategorized by someone else: the postcondition holds, and reporting it
        # removed is what lets this client's stale view of the old board catch up. Safe in
        # removed_images, unlike a deleted name: the DTO exists, so tag refetches succeed.
        (None, True, ["raced.png"], [], ["none", "board-p"]),
        # Deleted concurrently: a skip, matching the route's existing treatment of a name that
        # vanished before the loop reached it. removed_images would drive a 404 refetch.
        (None, False, [], [], []),
    ],
)
def test_remove_classifies_a_zero_row_scoped_delete_by_where_the_image_went(
    monkeypatch: pytest.MonkeyPatch,
    mock_invoker: Invoker,
    client: TestClient,
    now_on_board: str | None,
    record_exists: bool,
    expect_removed: list[str],
    expect_failed: list[str],
    expect_boards: list[str],
) -> None:
    _install(monkeypatch, mock_invoker)
    dto = MagicMock()
    dto.board_id = "board-p"
    monkeypatch.setattr(mock_invoker.services.images, "get_dto", MagicMock(return_value=dto))
    # The default single-user identity is an admin, so the write-access check passes without
    # touching board storage -- which is fine: the subject here is the write, not the check.
    monkeypatch.setattr(mock_invoker.services.board_images, "remove_image_from_board", MagicMock(return_value=0))
    monkeypatch.setattr(
        mock_invoker.services.board_image_records, "get_board_for_image", MagicMock(return_value=now_on_board)
    )
    if record_exists:
        monkeypatch.setattr(mock_invoker.services.image_records, "get", MagicMock(return_value=MagicMock()))
    else:
        from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException

        monkeypatch.setattr(
            mock_invoker.services.image_records, "get", MagicMock(side_effect=ImageRecordNotFoundException)
        )

    response = client.post("/api/v1/board_images/batch/delete", json={"image_names": ["raced.png"]})

    assert response.status_code == 201
    body = response.json()
    assert body["removed_images"] == expect_removed
    assert body["failed_images"] == expect_failed
    assert set(body["affected_boards"]) == set(expect_boards)


def test_remove_still_reports_a_nonzero_scoped_delete_as_removed(
    monkeypatch: pytest.MonkeyPatch, mock_invoker: Invoker, client: TestClient
) -> None:
    _install(monkeypatch, mock_invoker)
    dto = MagicMock()
    dto.board_id = "board-p"
    monkeypatch.setattr(mock_invoker.services.images, "get_dto", MagicMock(return_value=dto))
    monkeypatch.setattr(mock_invoker.services.board_images, "remove_image_from_board", MagicMock(return_value=1))

    response = client.post("/api/v1/board_images/batch/delete", json={"image_names": ["ok.png"]})

    assert response.status_code == 201
    body = response.json()
    assert body["removed_images"] == ["ok.png"]
    assert body["failed_images"] == []
    assert set(body["affected_boards"]) == {"none", "board-p"}


def test_add_reports_names_refused_by_a_destination_revoked_mid_batch(
    monkeypatch: pytest.MonkeyPatch, mock_invoker: Invoker, client: TestClient
) -> None:
    """A revoked destination fails the remaining names; it must not empty into a silent 201."""
    _install(monkeypatch, mock_invoker)
    monkeypatch.setattr(mock_invoker.services.image_records, "get_user_id", MagicMock(return_value="system"))
    monkeypatch.setattr(mock_invoker.services.board_image_records, "get_board_for_image", MagicMock(return_value=None))
    monkeypatch.setattr(mock_invoker.services.board_images, "add_image_to_board", MagicMock(return_value=None))
    # Pre-loop check passes, first per-name check passes, then the board flips Private (or is
    # deleted): every later re-check refuses. Patched at the route seam because the default
    # single-user identity is an admin, for whom the real helper never refuses.
    calls = {"n": 0}

    def write_access(board_id: str, current_user: object) -> None:
        calls["n"] += 1
        if calls["n"] > 2:
            raise HTTPException(status_code=403, detail="Not authorized to modify this board")

    monkeypatch.setattr("invokeai.app.api.routers.board_images._assert_board_write_access", write_access)

    response = client.post(
        "/api/v1/board_images/batch",
        json={"board_id": "board-x", "image_names": ["first.png", "second.png", "third.png"]},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["added_images"] == ["first.png"]
    # The refused names are failures the client can toast and retry -- not skips that leave a
    # 201 with empty lists for the UI to read as success.
    assert set(body["failed_images"]) == {"second.png", "third.png"}


def test_sqlite_remove_returns_the_row_count() -> None:
    """The storage layer itself: the row count is the only signal the scoped DELETE's scope
    held, and a `None` return silently restores report-a-removal-that-did-not-happen at the
    route (`None == 0` is False). Stubbed at the cursor in the manner of the image-records
    storage tests."""
    from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage

    storage = SqliteBoardImageRecordStorage.__new__(SqliteBoardImageRecordStorage)

    class _Cursor:
        rowcount = 0

        def execute(self, *args: object, **kwargs: object) -> None:
            pass

    class _Db:
        def transaction(self):
            from contextlib import contextmanager

            @contextmanager
            def _cm():
                yield _Cursor()

            return _cm()

    storage._db = _Db()  # pyright: ignore[reportAttributeAccessIssue]

    assert storage.remove_image_from_board("raced.png", "board-p") == 0
    _Cursor.rowcount = 1
    assert storage.remove_image_from_board("ok.png", "board-p") == 1
