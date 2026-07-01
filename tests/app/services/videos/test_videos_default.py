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

        deleted = video_service.delete_videos_on_board("board-1")

        # Only the video whose file we successfully removed should have its record deleted.
        invoker.services.video_records.delete_many.assert_called_once_with(["good.mp4"])
        # The method must also surface the truthful set to the caller so the API response can
        # avoid claiming the preserved record was deleted (JPPhoto PR #9163 May-22 follow-up).
        assert deleted == ["good.mp4"]

    def test_file_cleanup_failure_does_not_raise(self, video_service: VideoService):
        """A single file-delete failure must not surface as a 500 to the user — the rest of
        the board deletion has to keep going so other videos and the board itself can still
        be cleaned up."""
        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        invoker.services.board_video_records.get_all_board_video_names_for_board.return_value = ["v.mp4"]
        invoker.services.video_records.get.return_value = _make_record(video_name="v.mp4")
        invoker.services.video_files.delete.side_effect = Exception("permission denied")

        # Should not raise
        deleted = video_service.delete_videos_on_board("board-1")

        # And the failing video's record must be preserved.
        invoker.services.video_records.delete_many.assert_called_once_with([])
        assert deleted == []

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

        deleted = video_service.delete_videos_on_board("board-1")

        invoker.services.video_records.delete_many.assert_called_once_with(["a.mp4", "b.mp4"])
        assert deleted == ["a.mp4", "b.mp4"]


class TestCreateRollback:
    """Per JPPhoto's PR review (May 22 follow-up): if the video file save fails after the DB
    record has been written, the create path must roll back the record (and any board
    attachment) so the gallery never contains a DB-only ghost whose file endpoints 404.
    """

    def _wire_minimal_create_dependencies(self, invoker: MagicMock, video_name: str = "v.mp4") -> None:
        invoker.services.names.create_video_name.return_value = video_name
        invoker.services.configuration.image_subfolder_strategy = "flat"
        invoker.services.logger = MagicMock()

    def test_record_and_board_relation_rolled_back_when_file_save_fails(self, video_service: VideoService, tmp_path):
        from invokeai.app.services.video_files.video_files_common import VideoFileSaveException

        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        self._wire_minimal_create_dependencies(invoker, video_name="ghost.mp4")

        # The DB record save succeeds; the board attach succeeds; the file save explodes.
        invoker.services.video_records.save.return_value = None
        invoker.services.board_video_records.add_video_to_board.return_value = None
        invoker.services.video_files.save.side_effect = VideoFileSaveException("disk full")

        with pytest.raises(VideoFileSaveException):
            video_service.create(
                source_path=tmp_path / "src.mp4",
                width=64,
                height=64,
                duration=1.0,
                fps=24.0,
                video_origin=ResourceOrigin.EXTERNAL,
                video_category=ImageCategory.GENERAL,
                board_id="some-board",
            )

        # Both the DB record AND the board association must be unwound — otherwise the
        # gallery would show a ghost video whose file endpoints 404.
        invoker.services.board_video_records.remove_video_from_board.assert_called_once_with(video_name="ghost.mp4")
        invoker.services.video_records.delete.assert_called_once_with("ghost.mp4")

    def test_record_rolled_back_when_no_board_and_file_save_fails(self, video_service: VideoService, tmp_path):
        from invokeai.app.services.video_files.video_files_common import VideoFileSaveException

        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        self._wire_minimal_create_dependencies(invoker, video_name="solo.mp4")
        invoker.services.video_records.save.return_value = None
        invoker.services.video_files.save.side_effect = VideoFileSaveException("disk full")

        with pytest.raises(VideoFileSaveException):
            video_service.create(
                source_path=tmp_path / "src.mp4",
                width=64,
                height=64,
                duration=1.0,
                fps=24.0,
                video_origin=ResourceOrigin.EXTERNAL,
                video_category=ImageCategory.GENERAL,
                board_id=None,
            )

        # No board attachment was attempted, so no detach call should be made — but the
        # record must still be rolled back.
        invoker.services.board_video_records.remove_video_from_board.assert_not_called()
        invoker.services.video_records.delete.assert_called_once_with("solo.mp4")
