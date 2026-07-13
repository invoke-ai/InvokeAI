"""Tests for DiskVideoFileStorage (video_files_disk.py).

Covers the save-failure cleanup contract (JPPhoto PR #9163 follow-up): ``save()`` moves the
source MP4 into permanent storage *before* writing the thumbnail and sidecar, so a failure in
either of those later steps used to leave the moved MP4 (and any partial artifacts) on disk
with no DB record through which they could be managed — the caller rolls the record back on
``VideoFileSaveException`` but nothing removed the files.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.services.video_files.video_files_common import VideoFileSaveException
from invokeai.app.services.video_files.video_files_disk import DiskVideoFileStorage

VIDEO_NAME = "abc123.mp4"


@pytest.fixture
def storage(tmp_path: Path) -> DiskVideoFileStorage:
    return DiskVideoFileStorage(tmp_path / "videos")


def _make_source(tmp_path: Path) -> Path:
    # Not a decodable MP4 — thumbnail extraction fails gracefully (best-effort), which lets
    # these tests drive the sidecar path without a real video file.
    source = tmp_path / "source.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42 not a real mp4")
    return source


def _all_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def test_save_writes_video_and_sidecar(storage: DiskVideoFileStorage, tmp_path: Path):
    source = _make_source(tmp_path)

    storage.save(source_path=source, video_name=VIDEO_NAME, metadata='{"seed": 1}')

    assert storage.get_path(VIDEO_NAME).exists()
    assert not source.exists()
    assert storage.get_workflow(VIDEO_NAME) is None  # sidecar readable, workflow not set


def test_save_failure_after_move_removes_all_destination_files(
    storage: DiskVideoFileStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = _make_source(tmp_path)

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    # Force the sidecar write (the last step of save) to fail after the MP4 has been moved.
    monkeypatch.setattr("invokeai.app.services.video_files.video_files_disk.json.dump", broken_dump)

    with pytest.raises(VideoFileSaveException):
        storage.save(source_path=source, video_name=VIDEO_NAME, metadata='{"seed": 1}')

    assert _all_files(tmp_path / "videos") == []


def test_save_failure_in_thumbnail_write_removes_moved_video(
    storage: DiskVideoFileStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = _make_source(tmp_path)

    # Frame extraction itself is best-effort, but a failure while *writing* the extracted
    # thumbnail propagates. Simulate that: extraction succeeds, the write blows up.
    monkeypatch.setattr(
        "invokeai.app.services.video_files.video_files_disk.extract_video_frame",
        lambda *args, **kwargs: MagicMock(),
    )
    broken_thumbnail = MagicMock()
    broken_thumbnail.save.side_effect = OSError("read-only filesystem")
    monkeypatch.setattr(
        "invokeai.app.services.video_files.video_files_disk.make_thumbnail",
        lambda *args, **kwargs: broken_thumbnail,
    )

    with pytest.raises(VideoFileSaveException):
        storage.save(source_path=source, video_name=VIDEO_NAME)

    assert _all_files(tmp_path / "videos") == []


def test_save_failure_cleanup_covers_subfolders(
    storage: DiskVideoFileStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = _make_source(tmp_path)

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("invokeai.app.services.video_files.video_files_disk.json.dump", broken_dump)

    with pytest.raises(VideoFileSaveException):
        storage.save(
            source_path=source,
            video_name=VIDEO_NAME,
            video_subfolder="2026/07",
            metadata='{"seed": 1}',
        )

    assert _all_files(tmp_path / "videos") == []


def test_staged_delete_can_be_rolled_back(storage: DiskVideoFileStorage, tmp_path: Path):
    source = _make_source(tmp_path)
    storage.save(source_path=source, video_name=VIDEO_NAME, metadata='{"seed": 1}')
    video_path = storage.get_path(VIDEO_NAME)

    token = storage.stage_delete(VIDEO_NAME)
    assert not video_path.exists()

    storage.rollback_delete(token)
    assert video_path.exists()
    assert storage.get_workflow(VIDEO_NAME) is None


def test_staged_delete_can_be_committed(storage: DiskVideoFileStorage, tmp_path: Path):
    source = _make_source(tmp_path)
    storage.save(source_path=source, video_name=VIDEO_NAME, metadata='{"seed": 1}')

    token = storage.stage_delete(VIDEO_NAME)
    storage.commit_delete(token)

    assert _all_files(tmp_path / "videos") == []


def test_start_restores_staged_delete_when_record_still_exists(storage: DiskVideoFileStorage, tmp_path: Path):
    source = _make_source(tmp_path)
    storage.save(source_path=source, video_name=VIDEO_NAME, metadata='{"seed": 1}')
    storage.stage_delete(VIDEO_NAME)
    assert not storage.get_path(VIDEO_NAME).exists()
    invoker = MagicMock()
    invoker.services.video_records.get.return_value = MagicMock()

    DiskVideoFileStorage(tmp_path / "videos").start(invoker)

    assert storage.get_path(VIDEO_NAME).exists()
    assert not list((tmp_path / "videos").glob(".delete_*"))


def test_start_purges_staged_delete_when_record_is_gone(storage: DiskVideoFileStorage, tmp_path: Path):
    from invokeai.app.services.video_records.video_records_common import VideoRecordNotFoundException

    source = _make_source(tmp_path)
    storage.save(source_path=source, video_name=VIDEO_NAME, metadata='{"seed": 1}')
    storage.stage_delete(VIDEO_NAME)
    invoker = MagicMock()
    invoker.services.video_records.get.side_effect = VideoRecordNotFoundException

    DiskVideoFileStorage(tmp_path / "videos").start(invoker)

    assert _all_files(tmp_path / "videos") == []
    assert not list((tmp_path / "videos").glob(".delete_*"))
