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
        invoker.services.video_files.stage_delete.side_effect = [object(), Exception("disk error")]

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
        invoker.services.video_files.stage_delete.side_effect = Exception("permission denied")

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
        invoker.services.video_files.stage_delete.side_effect = [object(), object()]

        deleted = video_service.delete_videos_on_board("board-1")

        invoker.services.video_records.delete_many.assert_called_once_with(["a.mp4", "b.mp4"])
        assert deleted == ["a.mp4", "b.mp4"]

    def test_staging_cleanup_failure_is_deferred_after_records_are_deleted(self, video_service: VideoService):
        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        invoker.services.board_video_records.get_all_board_video_names_for_board.return_value = ["v.mp4"]
        invoker.services.video_records.get.return_value = _make_record(video_name="v.mp4")
        invoker.services.video_files.stage_delete.return_value = object()
        invoker.services.video_files.commit_delete.side_effect = OSError("staging directory busy")

        deleted = video_service.delete_videos_on_board("board-1")

        assert deleted == ["v.mp4"]
        invoker.services.video_records.delete_many.assert_called_once_with(["v.mp4"])
        invoker.services.logger.error.assert_called()


class TestDeleteAtomicity:
    def test_single_delete_rolls_files_back_when_record_delete_fails(self, video_service: VideoService):
        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        invoker.services.video_records.get.return_value = _make_record()
        invoker.services.video_records.delete.side_effect = RuntimeError("database unavailable")

        with pytest.raises(RuntimeError, match="database unavailable"):
            video_service.delete("abc.mp4")

        invoker.services.video_files.stage_delete.assert_called_once_with("abc.mp4", video_subfolder="")
        invoker.services.video_files.rollback_delete.assert_called_once()
        invoker.services.video_files.commit_delete.assert_not_called()


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

    def test_files_rolled_back_when_failure_occurs_after_file_save(self, video_service: VideoService, tmp_path):
        """The disk layer cleans up after its own save failures, but a failure *after* a
        successful file save (e.g. building the DTO) must also unwind the files — otherwise
        they'd sit on disk with no record pointing at them (JPPhoto PR #9163 July-10
        follow-up)."""
        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        self._wire_minimal_create_dependencies(invoker, video_name="late.mp4")
        invoker.services.video_records.save.return_value = None
        invoker.services.video_files.save.return_value = None
        # get_dto reads the record back — make that step explode.
        invoker.services.video_records.get.side_effect = RuntimeError("db went away")

        with pytest.raises(RuntimeError):
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

        invoker.services.video_records.delete.assert_called_once_with("late.mp4")
        invoker.services.video_files.delete.assert_called_once()
        assert invoker.services.video_files.delete.call_args.args[0] == "late.mp4"


class TestCreateBoardAttachFallback:
    """Board attachment during create is best-effort, mirroring ImageService.create: a board
    deleted between the caller's access check and the insert must not destroy a just-generated
    video. The fallback must be explicit, not silent — the returned DTO reports the actual
    (missing) board association and a warning is logged (JPPhoto PR #9163 July-10 follow-up).
    """

    def test_create_succeeds_with_explicit_fallback_when_board_attach_fails(
        self, video_service: VideoService, tmp_path
    ):
        invoker = video_service._VideoService__invoker  # type: ignore[attr-defined]
        invoker.services.names.create_video_name.return_value = "orphan.mp4"
        invoker.services.configuration.image_subfolder_strategy = "flat"
        invoker.services.logger = MagicMock()

        invoker.services.video_records.save.return_value = None
        invoker.services.board_video_records.add_video_to_board.side_effect = Exception("board was deleted")
        invoker.services.video_files.save.return_value = None
        # get_dto reads the record and the *actual* board association back.
        invoker.services.video_records.get.return_value = _make_record(video_name="orphan.mp4")
        invoker.services.board_video_records.get_board_for_video.return_value = None
        invoker.services.urls.get_video_url.return_value = "http://localhost/videos/orphan.mp4"

        video_dto = video_service.create(
            source_path=tmp_path / "src.mp4",
            width=64,
            height=64,
            duration=1.0,
            fps=24.0,
            video_origin=ResourceOrigin.EXTERNAL,
            video_category=ImageCategory.GENERAL,
            board_id="deleted-board",
        )

        # The video survives, but the DTO must not claim the requested board attachment
        # succeeded, and the fallback must be logged.
        assert video_dto.board_id is None
        invoker.services.logger.warning.assert_called_once()
        assert "deleted-board" in invoker.services.logger.warning.call_args.args[0]
        # Nothing was attached, so nothing should be unwound.
        invoker.services.board_video_records.remove_video_from_board.assert_not_called()
