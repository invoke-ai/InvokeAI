"""Tests for VideoService (videos_default.py).

Covers the board-cascade delete contract (JPPhoto PR #9163 follow-up). The old
implementation silently swallowed per-file delete errors and then deleted every
record anyway, which orphaned the file on disk while reporting success.
"""

from unittest.mock import MagicMock

import pytest

from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.video_records.video_records_common import VideoRecord
from invokeai.app.services.videos.videos_default import VideoService
from invokeai.app.util.misc import get_iso_timestamp


def _make_record(video_name: str = "abc.mp4", video_subfolder: str = "") -> VideoRecord:
    now = get_iso_timestamp()
    return VideoRecord(
        video_name=video_name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=ImageCategory.GENERAL,
        width=64,
        height=64,
        duration=1.0,
        fps=24.0,
        created_at=now,
        updated_at=now,
        is_intermediate=False,
        starred=False,
        has_workflow=False,
        video_subfolder=video_subfolder,
    )


@pytest.fixture
def video_service() -> VideoService:
    svc = VideoService()
    invoker = MagicMock()
    svc.start(invoker)
    return svc


class TestDeleteVideosOnBoardContract:
    """Per JPPhoto's PR review: a file-delete failure must NOT result in the DB record being
    deleted. Otherwise the API reports success while the file lingers on disk with no record
    pointing at it, and the user has no way to discover or clean up the leak.
    """

    def test_record_preserved_when_file_delete_fails(self, video_service: VideoService):
        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        invoker.services.board_video_records.get_all_board_video_names_for_board.return_value = [
            "good.mp4",
            "bad.mp4",
        ]
        invoker.services.video_records.get.side_effect = [
            _make_record(video_name="good.mp4", video_subfolder="general"),
            _make_record(video_name="bad.mp4", video_subfolder="general"),
        ]
        invoker.services.video_files.delete.side_effect = [None, Exception("disk error")]

        video_service.delete_videos_on_board("board-1")

        # Only the video whose file we successfully removed should have its record deleted.
        invoker.services.video_records.delete_many.assert_called_once_with(["good.mp4"])

    def test_file_cleanup_failure_does_not_raise(self, video_service: VideoService):
        """A single file-delete failure must not surface as a 500 to the user — the rest of
        the board deletion has to keep going so other videos and the board itself can still
        be cleaned up."""
        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        invoker.services.board_video_records.get_all_board_video_names_for_board.return_value = ["v.mp4"]
        invoker.services.video_records.get.return_value = _make_record(video_name="v.mp4")
        invoker.services.video_files.delete.side_effect = Exception("permission denied")

        # Should not raise
        video_service.delete_videos_on_board("board-1")

        # And the failing video's record must be preserved.
        invoker.services.video_records.delete_many.assert_called_once_with([])

    def test_all_records_deleted_on_full_success(self, video_service: VideoService):
        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        invoker.services.board_video_records.get_all_board_video_names_for_board.return_value = [
            "a.mp4",
            "b.mp4",
        ]
        invoker.services.video_records.get.side_effect = [
            _make_record(video_name="a.mp4"),
            _make_record(video_name="b.mp4"),
        ]
        invoker.services.video_files.delete.return_value = None

        video_service.delete_videos_on_board("board-1")

        invoker.services.video_records.delete_many.assert_called_once_with(["a.mp4", "b.mp4"])
