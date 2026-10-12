from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.board_records.board_records_common import BoardRecordOrderBy
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import LockTimeoutError
from invokeai.app.services.shared.database.queries.boards import BoardQueries
from invokeai.app.services.shared.pagination import SQLiteDirection
from tests.fixtures.sqlite_database import sqlite_cursor


def test_board_cover_uses_gallery_tie_breakers(mock_invoker: Invoker) -> None:
    created_at = datetime(2026, 7, 25, 12, 0, 0)
    mock_invoker.services.image_records.get_most_recent_image_for_board = MagicMock(
        return_value=SimpleNamespace(image_name="same.png", starred=True, created_at=created_at)
    )
    mock_invoker.services.video_records.get_most_recent_video_for_board = MagicMock(
        return_value=SimpleNamespace(video_name="same.mp4", starred=True, created_at=created_at)
    )

    assert mock_invoker.services.boards._resolve_cover("board") == (None, "same.mp4")


def test_board_listing_fetches_media_summaries_once(mock_invoker: Invoker) -> None:
    board_ids = [mock_invoker.services.board_records.save(f"Board {index}", "user").board_id for index in range(100)]
    summaries = {
        board_id: SimpleNamespace(
            cover_image_name=None,
            cover_video_name=f"{board_id}.mp4",
            image_count=1,
            video_count=2,
            asset_count=3,
            asset_video_count=1,
        )
        for board_id in board_ids
    }
    mock_invoker.services.gallery.get_board_media_summaries = MagicMock(return_value=summaries)  # type: ignore[attr-defined]
    mock_invoker.services.image_records.get_most_recent_image_for_board = MagicMock()
    mock_invoker.services.video_records.get_most_recent_video_for_board = MagicMock()
    mock_invoker.services.board_image_records.get_counts_for_board = MagicMock()
    mock_invoker.services.board_video_records.get_counts_for_board = MagicMock()

    result = mock_invoker.services.boards.get_many(
        user_id="user",
        is_admin=False,
        order_by=BoardRecordOrderBy.Name,
        direction=SQLiteDirection.Ascending,
        offset=0,
        limit=100,
    )

    assert len(result.items) == 100
    mock_invoker.services.gallery.get_board_media_summaries.assert_called_once()  # type: ignore[attr-defined]
    assert set(mock_invoker.services.gallery.get_board_media_summaries.call_args.args[0]) == set(board_ids)  # type: ignore[attr-defined]
    mock_invoker.services.image_records.get_most_recent_image_for_board.assert_not_called()
    mock_invoker.services.video_records.get_most_recent_video_for_board.assert_not_called()
    mock_invoker.services.board_image_records.get_counts_for_board.assert_not_called()
    mock_invoker.services.board_video_records.get_counts_for_board.assert_not_called()


def test_admin_board_listing_batches_owner_lookup(mock_invoker: Invoker) -> None:
    """Owner display names are fetched for the whole page in one call.

    An admin listing used to issue one `users.get` per board — 50 boards meant 50 extra
    queries for what is usually a handful of distinct owners.
    """
    owners = ["alice", "bob", "alice", "carol"] * 5
    board_ids = [
        mock_invoker.services.board_records.save(f"Board {index}", owner).board_id for index, owner in enumerate(owners)
    ]
    summaries = {
        board_id: SimpleNamespace(
            cover_image_name=None,
            cover_video_name=None,
            image_count=0,
            video_count=0,
            asset_count=0,
            asset_video_count=0,
        )
        for board_id in board_ids
    }
    mock_invoker.services.gallery.get_board_media_summaries = MagicMock(return_value=summaries)  # type: ignore[attr-defined]
    mock_invoker.services.users.get = MagicMock()  # type: ignore[method-assign]
    mock_invoker.services.users.get_many = MagicMock(  # type: ignore[method-assign]
        return_value={
            name: SimpleNamespace(display_name=name.title(), email=f"{name}@example.com")
            for name in ("alice", "bob", "carol")
        }
    )

    result = mock_invoker.services.boards.get_all(
        user_id="admin",
        is_admin=True,
        order_by=BoardRecordOrderBy.Name,
        direction=SQLiteDirection.Ascending,
    )

    assert len(result) == len(board_ids)
    assert {dto.owner_username for dto in result} == {"Alice", "Bob", "Carol"}
    mock_invoker.services.users.get.assert_not_called()  # type: ignore[attr-defined]
    mock_invoker.services.users.get_many.assert_called_once()  # type: ignore[attr-defined]


def test_non_admin_board_listing_skips_owner_lookup(mock_invoker: Invoker) -> None:
    """Non-admin listings don't show owner names, so they must not query for them at all."""
    board_id = mock_invoker.services.board_records.save("Board", "user").board_id
    mock_invoker.services.gallery.get_board_media_summaries = MagicMock(  # type: ignore[attr-defined]
        return_value={
            board_id: SimpleNamespace(
                cover_image_name=None,
                cover_video_name=None,
                image_count=0,
                video_count=0,
                asset_count=0,
                asset_video_count=0,
            )
        }
    )
    mock_invoker.services.users.get = MagicMock()  # type: ignore[method-assign]
    mock_invoker.services.users.get_many = MagicMock()  # type: ignore[method-assign]

    result = mock_invoker.services.boards.get_all(
        user_id="user",
        is_admin=False,
        order_by=BoardRecordOrderBy.Name,
        direction=SQLiteDirection.Ascending,
    )

    assert [dto.owner_username for dto in result] == [None]
    mock_invoker.services.users.get.assert_not_called()  # type: ignore[attr-defined]
    mock_invoker.services.users.get_many.assert_not_called()  # type: ignore[attr-defined]


def test_a_board_dto_counts_each_kind_of_media(mock_invoker: Invoker, mock_sqlite_database: Database) -> None:
    services = mock_invoker.services
    board_id = services.board_records.save("Board", "system").board_id
    with sqlite_cursor(mock_sqlite_database) as cursor:
        cursor.executemany(
            "INSERT INTO images (image_name, image_origin, image_category, width, height) VALUES (?, 'internal', ?, 1, 1)",
            [("general.png", "general"), ("control.png", "control"), ("mask.png", "mask")],
        )
        cursor.executemany(
            "INSERT INTO videos (video_name, video_origin, video_category, width, height) VALUES (?, 'internal', ?, 1, 1)",
            [("first.mp4", "general"), ("second.mp4", "general"), ("asset.mp4", "user")],
        )
    for name in ("general.png", "control.png", "mask.png"):
        services.board_image_records.add_image_to_board(board_id, name)
    for name in ("first.mp4", "second.mp4", "asset.mp4"):
        services.board_video_records.add_video_to_board(board_id, name)

    dto = services.boards.get_dto(board_id)

    assert (dto.image_count, dto.asset_count, dto.video_count, dto.asset_video_count) == (1, 2, 3, 1)


def test_board_records_get_does_not_disguise_a_storage_error_as_not_found(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A database error out of the SELECT must stay a database error, here a lock held too long elsewhere.

    Translating it made BoardRecordNotFoundException mean "no such board, OR the database is
    unreadable", and the board-image batch routes cannot tell those apart: they decide write
    access off this read once per name and treat not-found as a name to skip. A disk error would
    then drop names out of the response silently — reported neither as moved nor as failed —
    and the client would keep showing them as moved until the next refresh. Mirrors
    test_image_records_get_does_not_disguise_a_storage_error_as_not_found.
    """

    def locked(self: BoardQueries, board_id: str) -> None:
        raise LockTimeoutError("database is locked")

    monkeypatch.setattr(BoardQueries, "get", locked)

    with pytest.raises(LockTimeoutError):
        BoardRecordStorage(database).get("board-1")
