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
from PIL import Image

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


def test_save_uses_a_passed_first_frame_instead_of_running_the_ladder(
    storage: DiskVideoFileStorage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The upload path decodes a frame to prove decodability and hands it over; running the
    seek ladder again would cost up to four more decode-worker spawns inside the request."""
    source = _make_source(tmp_path)
    calls: list[Path] = []
    monkeypatch.setattr(
        "invokeai.app.services.video_files.video_files_disk.extract_representative_video_frame",
        lambda path, *args, **kwargs: calls.append(path),
    )

    storage.save(source_path=source, video_name=VIDEO_NAME, first_frame=Image.new("RGB", (48, 32), (200, 90, 40)))

    assert calls == []
    with Image.open(storage.get_path(VIDEO_NAME.replace(".mp4", ".webp"), thumbnail=True)) as thumbnail:
        assert thumbnail.size == (48, 32)


def test_save_with_move_source_false_leaves_the_source_intact(storage: DiskVideoFileStorage, tmp_path: Path):
    """Copying a video the server already owns must not consume it.

    ``save()`` moves by default because every generation and upload hands it a temp file. But
    ``POST /videos/copy`` passes the *source's own* managed path, so a move there deletes the
    original project's video and leaves its record pointing at nothing.
    """
    source = _make_source(tmp_path)
    original_bytes = source.read_bytes()

    storage.save(source_path=source, video_name=VIDEO_NAME, move_source=False)

    assert source.exists()
    assert source.read_bytes() == original_bytes
    assert storage.get_path(VIDEO_NAME).read_bytes() == original_bytes


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
        "invokeai.app.services.video_files.video_files_disk.extract_representative_video_frame",
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


@pytest.mark.parametrize(
    "video_name,video_subfolder",
    [
        ("../outside.mp4", ""),
        (VIDEO_NAME, "../outside"),
    ],
)
def test_invalid_staged_delete_does_not_create_staging_directory(
    storage: DiskVideoFileStorage, tmp_path: Path, video_name: str, video_subfolder: str
) -> None:
    video_path = storage.get_path(VIDEO_NAME)
    video_path.write_bytes(b"video")

    with pytest.raises(ValueError):
        storage.stage_delete(video_name, video_subfolder)

    assert video_path.exists()
    assert not list((tmp_path / "videos").glob(".delete_*"))


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


def test_start_keeps_staged_delete_when_the_record_cannot_be_read(storage: DiskVideoFileStorage, tmp_path: Path):
    """A database it merely could not read must not count as proof the delete committed.

    The recovery decides between purging the staged files and restoring them by asking whether
    the record is still there, so it needs that read to distinguish "gone" from "could not
    look". `SqliteVideoRecordStorage.get` used to translate every sqlite3.Error into
    VideoRecordNotFoundException, which made an unreadable database delete the user's video
    files outright. Now the staged copy survives for a later attempt.
    """
    import sqlite3

    source = _make_source(tmp_path)
    storage.save(source_path=source, video_name=VIDEO_NAME, metadata='{"seed": 1}')
    storage.stage_delete(VIDEO_NAME)
    invoker = MagicMock()
    invoker.services.video_records.get.side_effect = sqlite3.OperationalError("database is locked")

    DiskVideoFileStorage(tmp_path / "videos").start(invoker)

    assert list((tmp_path / "videos").glob(".delete_*")), "the staged files were destroyed"


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


# --- Embedded metadata -------------------------------------------------------------------------
#
# Metadata, workflow and graph live inside the MP4 as keyed metadata; the JSON sidecar is only
# the fallback when the remux fails, and the read path for videos stored before embedding.


@pytest.fixture
def real_source(tmp_path: Path) -> Path:
    import numpy as np

    from invokeai.app.util.video_encoding import make_mp4_writer

    source = tmp_path / "real.mp4"
    writer = make_mp4_writer(source, fps=8.0)
    try:
        for _ in range(2):
            writer.append_data(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.close()
    return source


def _sidecars(storage: DiskVideoFileStorage) -> list[Path]:
    return [p for p in _all_files(storage.get_path(VIDEO_NAME).parent) if p.suffix == ".json"]


def test_save_embeds_metadata_workflow_and_graph_in_the_mp4_and_writes_no_sidecar(
    storage: DiskVideoFileStorage, real_source: Path
):
    from invokeai.app.util.mp4_metadata import read_mp4_tags

    storage.save(
        source_path=real_source,
        video_name=VIDEO_NAME,
        metadata='{"seed": 1}',
        workflow='{"name": "wf"}',
        graph='{"nodes": {}, "edges": []}',
    )

    embedded = read_mp4_tags(storage.get_path(VIDEO_NAME))
    assert {key: value for key, value in embedded.items() if key.startswith("invokeai_")} == {
        "invokeai_metadata": '{"seed": 1}',
        "invokeai_workflow": '{"name": "wf"}',
        "invokeai_graph": '{"nodes": {}, "edges": []}',
    }
    assert storage.get_workflow(VIDEO_NAME) == '{"name": "wf"}'
    assert storage.get_graph(VIDEO_NAME) == '{"nodes": {}, "edges": []}'
    assert _sidecars(storage) == []
    # The thumbnail was taken from the rewritten file, which still decodes.
    assert storage.get_path(VIDEO_NAME, thumbnail=True).exists()


def test_save_without_any_metadata_leaves_the_file_byte_identical(storage: DiskVideoFileStorage, real_source: Path):
    original = real_source.read_bytes()

    storage.save(source_path=real_source, video_name=VIDEO_NAME)

    assert storage.get_path(VIDEO_NAME).read_bytes() == original
    assert _sidecars(storage) == []


def test_save_falls_back_to_a_sidecar_when_embedding_fails(
    storage: DiskVideoFileStorage, real_source: Path, monkeypatch: pytest.MonkeyPatch
):
    from invokeai.app.util.mp4_metadata import Mp4MetadataError

    original = real_source.read_bytes()

    def broken_write(*args, **kwargs):
        raise Mp4MetadataError("ffmpeg exploded")

    monkeypatch.setattr("invokeai.app.services.video_files.video_files_disk.write_mp4_tags", broken_write)

    storage.save(source_path=real_source, video_name=VIDEO_NAME, workflow='{"name": "wf"}')

    assert storage.get_path(VIDEO_NAME).read_bytes() == original
    assert len(_sidecars(storage)) == 1
    assert storage.get_workflow(VIDEO_NAME) == '{"name": "wf"}'
    assert storage.get_graph(VIDEO_NAME) is None
    # No stray remux temp file survives the failure.
    assert [p.name for p in _all_files(storage.get_path(VIDEO_NAME).parent) if p.name.startswith(".embed_")] == []


def test_legacy_sidecar_is_still_read_for_an_untagged_video(storage: DiskVideoFileStorage, real_source: Path):
    storage.save(source_path=real_source, video_name=VIDEO_NAME)
    sidecar_dir = storage.get_path(VIDEO_NAME).parent / "sidecars"
    sidecar_dir.mkdir(exist_ok=True)
    (sidecar_dir / "abc123.json").write_text(
        '{"invokeai_metadata": null, "invokeai_workflow": "{\\"name\\": \\"old\\"}", "invokeai_graph": null}'
    )

    assert storage.get_workflow(VIDEO_NAME) == '{"name": "old"}'
    assert storage.get_graph(VIDEO_NAME) is None


def test_embedded_tags_win_over_a_stale_sidecar(storage: DiskVideoFileStorage, real_source: Path):
    storage.save(source_path=real_source, video_name=VIDEO_NAME, workflow='{"name": "embedded"}')
    sidecar_dir = storage.get_path(VIDEO_NAME).parent / "sidecars"
    sidecar_dir.mkdir(exist_ok=True)
    (sidecar_dir / "abc123.json").write_text('{"invokeai_workflow": "{\\"name\\": \\"stale\\"}"}')

    assert storage.get_workflow(VIDEO_NAME) == '{"name": "embedded"}'


def test_save_of_an_already_tagged_file_skips_the_rewrite(
    storage: DiskVideoFileStorage, real_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A re-uploaded InvokeAI download carries exactly the tags being saved; a 1 GB rewrite for
    an identical result is wasted I/O."""
    from invokeai.app.util.mp4_metadata import write_mp4_tags

    tags = {"invokeai_metadata": '{"seed": 9}', "invokeai_graph": '{"nodes": {}, "edges": []}'}
    tagged = tmp_path / "tagged.mp4"
    write_mp4_tags(real_source, tagged, tags)
    original = tagged.read_bytes()
    monkeypatch.setattr(
        "invokeai.app.services.video_files.video_files_disk.write_mp4_tags",
        lambda *args, **kwargs: pytest.fail("remux ran for identical tags"),
    )

    storage.save(
        source_path=tagged, video_name=VIDEO_NAME, metadata=tags["invokeai_metadata"], graph=tags["invokeai_graph"]
    )

    assert storage.get_path(VIDEO_NAME).read_bytes() == original
    assert _sidecars(storage) == []


def test_crlf_metadata_is_embedded_not_rejected(storage: DiskVideoFileStorage, real_source: Path):
    """CRLF-formatted JSON from an API client must neither fail the save nor lose the record."""
    from invokeai.app.util.mp4_metadata import read_mp4_tags

    metadata = '{\r\n  "seed": 1\r\n}'

    storage.save(source_path=real_source, video_name=VIDEO_NAME, metadata=metadata)

    assert read_mp4_tags(storage.get_path(VIDEO_NAME), keys=["invokeai_metadata"]) == {"invokeai_metadata": metadata}
    assert _sidecars(storage) == []


def test_a_tag_the_writer_refuses_falls_back_to_a_sidecar_without_leaking(
    storage: DiskVideoFileStorage, real_source: Path
):
    original = real_source.read_bytes()

    storage.save(source_path=real_source, video_name=VIDEO_NAME, metadata="ends in a backslash \\")

    assert storage.get_path(VIDEO_NAME).read_bytes() == original
    assert len(_sidecars(storage)) == 1
    assert [p.name for p in _all_files(storage.get_path(VIDEO_NAME).parent) if p.name.startswith(".embed_")] == []


def test_start_sweeps_remux_temp_files_left_by_a_crash(tmp_path: Path):
    storage = DiskVideoFileStorage(tmp_path / "videos")
    stale = tmp_path / "videos" / ".embed_abc.mp4"
    stale.write_bytes(b"partial")
    invoker = MagicMock()

    storage.start(invoker)

    assert not stale.exists()


def test_a_file_carrying_a_tag_the_caller_dropped_is_rewritten_without_it(
    storage: DiskVideoFileStorage, real_source: Path, tmp_path: Path
):
    """An upload whose embedded workflow failed validation arrives with metadata and graph only;
    the stored file must not keep serving the rejected workflow through get_workflow."""
    from invokeai.app.util.mp4_metadata import read_mp4_tags, write_mp4_tags

    tagged = tmp_path / "tagged.mp4"
    write_mp4_tags(
        real_source,
        tagged,
        {"invokeai_metadata": '{"seed": 1}', "invokeai_workflow": '{"not": "a workflow"}', "invokeai_graph": "{}"},
    )

    storage.save(source_path=tagged, video_name=VIDEO_NAME, metadata='{"seed": 1}', graph="{}")

    embedded = read_mp4_tags(storage.get_path(VIDEO_NAME))
    assert "invokeai_workflow" not in embedded
    assert embedded["invokeai_metadata"] == '{"seed": 1}'
    assert storage.get_workflow(VIDEO_NAME) is None
    assert storage.get_graph(VIDEO_NAME) == "{}"


def test_a_file_whose_only_tag_was_rejected_is_stored_without_it(
    storage: DiskVideoFileStorage, real_source: Path, tmp_path: Path
):
    """No record to save, but the upload carried an invalid workflow: the stored file must not keep it,
    or get_workflow (and a later copy) would launder the rejected value back into a record."""
    from invokeai.app.util.mp4_metadata import read_mp4_tags, write_mp4_tags

    tagged = tmp_path / "tagged.mp4"
    write_mp4_tags(real_source, tagged, {"invokeai_workflow": '{"not": "a workflow"}'})

    storage.save(source_path=tagged, video_name=VIDEO_NAME)

    assert "invokeai_workflow" not in read_mp4_tags(storage.get_path(VIDEO_NAME))
    assert storage.get_workflow(VIDEO_NAME) is None
    assert _sidecars(storage) == []
