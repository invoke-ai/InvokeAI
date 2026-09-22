import errno
import hashlib
import os
import platform
import shutil
import stat
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from invokeai.app.services.image_files.image_files_common import ImageFileDeleteException, ImageFileSaveException
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage, _should_use_png_rle
from invokeai.app.util.thumbnails import get_thumbnail_name


def _restart(storage: DiskImageFileStorage, record_exists: bool) -> DiskImageFileStorage:
    """Simulates a restart over the same output folder, running journal recovery."""
    invoker = MagicMock()
    invoker.services.image_records.exists.return_value = record_exists
    restarted = DiskImageFileStorage(storage.image_root)
    restarted.start(invoker)
    return restarted


posix_only = pytest.mark.skipif(os.name == "nt", reason="Windows cannot open a directory for fsync")


def _failing_directory_fsync(error_number: int):
    """An ``os.fsync`` that fails for directory descriptors only; file fsyncs still work."""
    real_fsync = os.fsync

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(error_number, os.strerror(error_number))
        real_fsync(fd)

    return fsync


@pytest.fixture
def image_names() -> list[str]:
    # Determine the platform and return a path that matches its format
    if platform.system() == "Windows":
        return [
            # Relative paths
            "folder\\evil.txt",
            "folder\\..\\evil.txt",
            # Absolute paths
            "\\folder\\evil.txt",
            "C:\\folder\\..\\evil.txt",
        ]
    else:
        return [
            # Relative paths
            "folder/evil.txt",
            "folder/../evil.txt",
            # Absolute paths
            "/folder/evil.txt",
            "/folder/../evil.txt",
        ]


@pytest.fixture
def disk_storage(tmp_path: Path) -> DiskImageFileStorage:
    storage = DiskImageFileStorage(tmp_path)
    # Mock the invoker for save() which needs compress_level
    mock_invoker = MagicMock()
    mock_invoker.services.configuration.pil_compress_level = 6
    # Deletion asks the record store whether an image is still referenced; say yes unless a test
    # says otherwise, so nothing here depends on a bare MagicMock happening to be truthy.
    mock_invoker.services.image_records.exists.return_value = True
    storage._DiskImageFileStorage__invoker = mock_invoker  # type: ignore
    return storage


def test_directory_traversal_protection(tmp_path: Path, image_names: list[str]):
    """Test that the image file storage prevents directory traversal attacks.

    There are two safeguards in the `DiskImageFileStorage.get_path` method:
    1. Check if the image name contains any directory traversal characters
    2. Check if the resulting path is relative to the base folder

    This test checks the first safeguard. I'd like to check the second but I cannot figure out a test case that would
    pass the first check but fail the second check.
    """
    image_files_disk = DiskImageFileStorage(tmp_path)
    for name in image_names:
        with pytest.raises(ValueError, match="Invalid image name, potential directory traversal detected"):
            image_files_disk.get_path(name)


def test_image_paths_relative_to_storage_dir(tmp_path: Path):
    image_files_disk = DiskImageFileStorage(tmp_path)
    path = image_files_disk.get_path("foo.png")
    assert path.is_relative_to(tmp_path)


@pytest.mark.parametrize(
    ("compress_level", "expected_compress_type"),
    [(0, None), (1, zlib.Z_RLE), (7, None)],
)
def test_save_uses_rle_only_for_compression_level_one(
    tmp_path: Path, compress_level: int, expected_compress_type: int | None
):
    storage = DiskImageFileStorage(tmp_path)
    mock_invoker = MagicMock()
    mock_invoker.services.configuration.pil_compress_level = compress_level
    storage._DiskImageFileStorage__invoker = mock_invoker  # type: ignore

    with (
        patch("invokeai.app.services.image_files.image_files_disk._should_use_png_rle", return_value=True),
        patch.object(Image.Image, "save", autospec=True) as save_mock,
    ):
        storage.save(image=Image.new("RGBA", (32, 32)), image_name="test.png")

    png_calls = [call for call in save_mock.call_args_list if len(call.args) > 2 and call.args[2] == "PNG"]
    assert len(png_calls) == 1
    assert png_calls[0].kwargs["compress_level"] == compress_level
    if expected_compress_type is None:
        assert "compress_type" not in png_calls[0].kwargs
    else:
        assert png_calls[0].kwargs["compress_type"] == expected_compress_type


def test_png_rle_probe_rejects_structured_images():
    entropy = Image.frombytes("RGB", (512, 512), hashlib.shake_256(b"png-rle-test").digest(512 * 512 * 3))
    gradient = Image.linear_gradient("L").resize((512, 512)).convert("RGB")

    assert _should_use_png_rle(entropy)
    assert not _should_use_png_rle(gradient)

    entropy.close()
    gradient.close()


def _make_round_trip_image(mode: str) -> Image.Image:
    image = Image.new(mode, (4, 4))
    if mode == "P":
        palette = [component for index in range(256) for component in (index, 255 - index, index // 2, index)]
        image.putpalette(palette, rawmode="RGBA")
        image.putdata(range(16))
    else:
        values = {
            "1": [0, 1],
            "L": [0, 255],
            "LA": [(17, 0), (201, 255)],
            "RGB": [(1, 2, 3), (251, 252, 253)],
            "RGBA": [(1, 2, 3, 0), (251, 252, 253, 255)],
            "I;16": [0, 65535],
        }
        image.putdata(values[mode] * 8)
    return image


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGB", "RGBA", "I;16"])
def test_level_one_png_round_trip_from_disk(tmp_path: Path, mode: str):
    storage = DiskImageFileStorage(tmp_path)
    mock_invoker = MagicMock()
    mock_invoker.services.configuration.pil_compress_level = 1
    storage._DiskImageFileStorage__invoker = mock_invoker  # type: ignore

    image = _make_round_trip_image(mode)
    expected_bytes = image.tobytes()
    expected_rgba = image.convert("RGBA").tobytes() if mode == "P" else None
    metadata = f'{{"mode":"{mode}"}}'
    image_name = f"round-trip-{mode.replace(';', '-')}.png"

    with patch("invokeai.app.services.image_files.image_files_disk._should_use_png_rle", return_value=True):
        storage.save(image=image, image_name=image_name, metadata=metadata)
    image_path = storage.get_path(image_name)
    storage.evict_cache_paths([image_path])

    with Image.open(image_path) as loaded:
        loaded.load()
        assert loaded.format == "PNG"
        assert loaded.mode == mode
        assert loaded.tobytes() == expected_bytes
        assert loaded.info["invokeai_metadata"] == metadata
        if mode in {"LA", "RGBA"}:
            assert loaded.getchannel("A").tobytes() == image.getchannel("A").tobytes()
        if mode == "P":
            assert loaded.info["transparency"] == bytes(range(256))
            assert loaded.convert("RGBA").tobytes() == expected_rgba

    image.close()


def test_large_16_bit_png_save_creates_thumbnail(tmp_path: Path):
    storage = DiskImageFileStorage(tmp_path)
    mock_invoker = MagicMock()
    mock_invoker.services.configuration.pil_compress_level = 6
    storage._DiskImageFileStorage__invoker = mock_invoker  # type: ignore
    image_name = "large-16-bit.png"
    image = Image.new("I;16", (1024, 1024), 32768)

    try:
        storage.save(image=image, image_name=image_name)

        image_path = storage.get_path(image_name)
        thumbnail_path = storage.get_path(image_name, thumbnail=True)
        assert image_path.exists()
        assert thumbnail_path.exists()
        with Image.open(thumbnail_path) as thumbnail:
            thumbnail.load()
            assert thumbnail.format == "WEBP"
            assert thumbnail.mode == "RGB"
    finally:
        image.close()


def test_palette_transparency_survives_thumbnail_save(tmp_path: Path):
    storage = DiskImageFileStorage(tmp_path)
    mock_invoker = MagicMock()
    mock_invoker.services.configuration.pil_compress_level = 6
    storage._DiskImageFileStorage__invoker = mock_invoker  # type: ignore
    image = Image.new("P", (32, 32), 0)
    image.putpalette([255, 0, 0] * 256)
    image.info["transparency"] = 0

    try:
        storage.save(image=image, image_name="transparent-palette.png")

        with Image.open(storage.get_path("transparent-palette.png", thumbnail=True)) as thumbnail:
            thumbnail.load()
            assert thumbnail.mode == "RGBA"
            assert thumbnail.getpixel((0, 0))[3] == 0
    finally:
        image.close()


def test_save_removes_partial_files_when_thumbnail_save_fails(tmp_path: Path):
    storage = DiskImageFileStorage(tmp_path)
    mock_invoker = MagicMock()
    mock_invoker.services.configuration.pil_compress_level = 6
    storage._DiskImageFileStorage__invoker = mock_invoker  # type: ignore
    image_name = "thumbnail-failure.png"
    image = Image.new("RGB", (32, 32), "red")
    broken_thumbnail = MagicMock()
    broken_thumbnail.save.side_effect = OSError("thumbnail filesystem failure")

    try:
        with patch(
            "invokeai.app.services.image_files.image_files_disk.make_thumbnail",
            return_value=broken_thumbnail,
        ):
            with pytest.raises(ImageFileSaveException):
                storage.save(image=image, image_name=image_name)

        assert not storage.get_path(image_name).exists()
        assert not storage.get_path(image_name, thumbnail=True).exists()
    finally:
        image.close()


# ── Subfolder validation tests (Point 1) ──


class TestValidateSubfolder:
    """Tests for _validate_subfolder() and get_path() with image_subfolder."""

    def test_valid_single_segment(self, tmp_path: Path):
        storage = DiskImageFileStorage(tmp_path)
        path = storage.get_path("img.png", image_subfolder="general")
        assert path.is_relative_to(tmp_path)
        assert "general" in path.parts

    def test_valid_nested_subfolder(self, tmp_path: Path):
        storage = DiskImageFileStorage(tmp_path)
        path = storage.get_path("img.png", image_subfolder="2026/03/17")
        assert path.is_relative_to(tmp_path)
        assert path.name == "img.png"

    @pytest.mark.parametrize(
        "subfolder,error_match",
        [
            ("../x", "Parent directory references not allowed"),
            ("x/../y", "Parent directory references not allowed"),
            ("/abs", "Absolute paths not allowed"),
            ("a//b", "Empty path segments not allowed"),
            ("a\\b", "Backslashes not allowed"),
        ],
        ids=["parent_traversal", "mid_traversal", "absolute", "double_slash", "backslash"],
    )
    def test_invalid_subfolders(self, tmp_path: Path, subfolder: str, error_match: str):
        storage = DiskImageFileStorage(tmp_path)
        with pytest.raises(ValueError, match=error_match):
            storage.get_path("img.png", image_subfolder=subfolder)

    def test_empty_subfolder_gives_root(self, tmp_path: Path):
        storage = DiskImageFileStorage(tmp_path)
        path = storage.get_path("img.png", image_subfolder="")
        assert path == (tmp_path / "img.png").resolve()

    def test_thumbnail_mirrors_subfolder(self, tmp_path: Path):
        storage = DiskImageFileStorage(tmp_path)
        subfolder = "2026/03/17"
        img_path = storage.get_path("img.png", thumbnail=False, image_subfolder=subfolder)
        thumb_path = storage.get_path("img.png", thumbnail=True, image_subfolder=subfolder)

        # Both should contain the subfolder segments
        assert subfolder.replace("/", "\\") in str(img_path) or subfolder in str(img_path)
        assert subfolder.replace("/", "\\") in str(thumb_path) or subfolder in str(thumb_path)

        # Thumbnail should be under thumbnails folder
        thumbnails_folder = (tmp_path / "thumbnails").resolve()
        assert thumb_path.is_relative_to(thumbnails_folder)


class TestSaveDeleteRoundTrip:
    """Save/delete round-trip with subfolders, including thumbnail mirroring."""

    def test_save_and_delete_with_subfolder(self, disk_storage: DiskImageFileStorage, tmp_path: Path):
        subfolder = "2026/04/05"
        image_name = "test_image.png"
        image = Image.new("RGB", (64, 64), color="red")

        disk_storage.save(image=image, image_name=image_name, image_subfolder=subfolder)

        # Image file exists
        image_path = disk_storage.get_path(image_name, image_subfolder=subfolder)
        assert image_path.exists()

        # Thumbnail file exists in mirrored subfolder
        thumbnail_name = get_thumbnail_name(image_name)
        thumb_path = disk_storage.get_path(image_name, thumbnail=True, image_subfolder=subfolder)
        assert thumb_path.name == thumbnail_name
        assert not thumb_path.name.startswith("thumbnail_thumbnail_")
        assert thumb_path.exists()

        # Round-trip read
        loaded = disk_storage.get(image_name, image_subfolder=subfolder)
        assert loaded.size == (64, 64)

        # Delete removes both
        disk_storage.delete(image_name, image_subfolder=subfolder)
        assert not image_path.exists()
        assert not thumb_path.exists()

    def test_save_flat_and_subfolder_coexist(self, disk_storage: DiskImageFileStorage, tmp_path: Path):
        image = Image.new("RGB", (32, 32), color="blue")

        disk_storage.save(image=image, image_name="flat.png", image_subfolder="")
        disk_storage.save(image=image, image_name="nested.png", image_subfolder="general")

        flat_path = disk_storage.get_path("flat.png", image_subfolder="")
        nested_path = disk_storage.get_path("nested.png", image_subfolder="general")

        assert flat_path.exists()
        assert nested_path.exists()
        assert flat_path.parent != nested_path.parent

    def test_staged_delete_can_be_rolled_back(self, disk_storage: DiskImageFileStorage):
        image_name = "rollback.png"
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=image_name)
        image_path = disk_storage.get_path(image_name)
        thumbnail_path = disk_storage.get_path(image_name, thumbnail=True)

        token = disk_storage.stage_delete(image_name)

        assert not image_path.exists()
        assert not thumbnail_path.exists()

        disk_storage.rollback_delete(token)

        assert image_path.exists()
        assert thumbnail_path.exists()

    def test_staged_delete_can_be_committed(self, disk_storage: DiskImageFileStorage, tmp_path: Path):
        image_name = "commit.png"
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=image_name)

        token = disk_storage.stage_delete(image_name)
        disk_storage.commit_delete(token)

        assert not list(tmp_path.glob(".delete_*"))

    def test_invalid_staged_delete_does_not_create_staging_directory(
        self, disk_storage: DiskImageFileStorage, tmp_path: Path
    ):
        with pytest.raises(ValueError, match="Invalid image name"):
            disk_storage.stage_delete("../invalid.png")

        assert not list(tmp_path.glob(".delete_*"))

    def test_startup_restores_staged_files_when_record_exists(self, disk_storage: DiskImageFileStorage):
        image_name = "recover.png"
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=image_name)
        image_path = disk_storage.get_path(image_name)
        thumbnail_path = disk_storage.get_path(image_name, thumbnail=True)
        disk_storage.stage_delete(image_name)

        _restart(disk_storage, record_exists=True)

        assert image_path.exists()
        assert thumbnail_path.exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_startup_purges_staged_files_when_record_was_deleted(self, disk_storage: DiskImageFileStorage):
        image_name = "purge.png"
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=image_name)
        image_path = disk_storage.get_path(image_name)
        disk_storage.stage_delete(image_name)

        _restart(disk_storage, record_exists=False)

        assert not image_path.exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_startup_leaves_the_journal_when_the_record_store_is_unreadable(self, disk_storage: DiskImageFileStorage):
        """A database fault must not decide an image's fate; the journal is retried next startup."""
        image_name = "unreadable.png"
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=image_name)
        disk_storage.stage_delete(image_name)

        invoker = MagicMock()
        invoker.services.image_records.exists.side_effect = RuntimeError("database is locked")
        restarted = DiskImageFileStorage(disk_storage.image_root)
        restarted.start(invoker)

        assert list(disk_storage.image_root.glob(".delete_*"))

    def test_startup_purges_restored_files_whose_record_vanished_during_recovery(
        self, disk_storage: DiskImageFileStorage
    ):
        """Recovery re-checks the record after restoring, exactly as rollback_delete() does.

        Another Invoke sharing the output folder can delete the record while the files sit staged;
        its purge finds nothing, and restoring them afterwards would strand them with no record and
        no journal left to find them by (JPPhoto, PR #9361 round 4).
        """
        image_name = "raced.png"
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=image_name)
        disk_storage.stage_delete(image_name)

        invoker = MagicMock()
        # Present when recovery first looks, gone by the time the files are back.
        invoker.services.image_records.exists.side_effect = [True, False]
        restarted = DiskImageFileStorage(disk_storage.image_root)
        restarted.start(invoker)

        assert not disk_storage.get_path(image_name).exists()
        assert not disk_storage.get_path(image_name, thumbnail=True).exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_startup_keeps_the_journal_when_the_recheck_cannot_read_the_record_store(
        self, disk_storage: DiskImageFileStorage
    ):
        """A fault on the post-restore re-check keeps the files and the journal for the next start."""
        image_name = "kept.png"
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=image_name)
        disk_storage.stage_delete(image_name)

        invoker = MagicMock()
        invoker.services.image_records.exists.side_effect = [True, RuntimeError("database is locked")]
        restarted = DiskImageFileStorage(disk_storage.image_root)
        restarted.start(invoker)

        assert disk_storage.get_path(image_name).exists()
        assert disk_storage.get_path(image_name, thumbnail=True).exists()
        assert list(disk_storage.image_root.glob(".delete_*"))

    def test_startup_does_not_recheck_a_pending_journal_that_restored_nothing(self, disk_storage: DiskImageFileStorage):
        """Only a restore can strand files; a journal that moved nothing costs one lookup."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="pending.png")
        disk_storage.begin_delete([("pending.png", "")])

        invoker = MagicMock()
        invoker.services.image_records.exists.side_effect = [True, AssertionError("unexpected re-check")]
        restarted = DiskImageFileStorage(disk_storage.image_root)
        restarted.start(invoker)

        assert disk_storage.get_path("pending.png").exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))


class TestPendingDeleteJournal:
    """begin_delete() writes the journal that makes records-first deletion recoverable.

    Nothing is moved, so a failure can only ever leave files nothing references — and the journal
    is what lets the next startup find and purge exactly those.
    """

    def test_begin_delete_leaves_the_files_in_place(self, disk_storage: DiskImageFileStorage):
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="live.png")

        disk_storage.begin_delete([("live.png", "")])

        assert disk_storage.get_path("live.png").exists()
        assert disk_storage.get_path("live.png", thumbnail=True).exists()
        assert len(list(disk_storage.image_root.glob(".delete_*"))) == 1

    def test_begin_delete_rejects_an_unusable_name_before_writing_a_journal(
        self, disk_storage: DiskImageFileStorage, tmp_path: Path
    ):
        """The caller deletes records straight after this returns, so a bad name must fail here."""
        with pytest.raises(ValueError, match="Invalid image name"):
            disk_storage.begin_delete([("ok.png", ""), ("../evil.png", "")])

        assert not list(tmp_path.glob(".delete_*"))

    def test_commit_purges_the_files_and_drops_the_journal(self, disk_storage: DiskImageFileStorage):
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="gone.png")
        token = disk_storage.begin_delete([("gone.png", "")])

        disk_storage.commit_delete(token)

        assert not disk_storage.get_path("gone.png").exists()
        assert not disk_storage.get_path("gone.png", thumbnail=True).exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_commit_purges_only_the_named_images(self, disk_storage: DiskImageFileStorage):
        """The journal lists every candidate; only the records that were really deleted are purged."""
        for name in ("deleted.png", "promoted.png"):
            disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=name)
        token = disk_storage.begin_delete([("deleted.png", ""), ("promoted.png", "")])

        disk_storage.commit_delete(token, image_names=["deleted.png"])

        assert not disk_storage.get_path("deleted.png").exists()
        assert disk_storage.get_path("promoted.png").exists()
        assert disk_storage.get_path("promoted.png", thumbnail=True).exists()

    def test_commit_keeps_the_journal_when_a_file_cannot_be_purged(self, disk_storage: DiskImageFileStorage):
        """One unremovable file must not abort the other purges, and must not discard the journal:
        the entry has to survive so the next startup can retry it."""
        for name in ("bad.png", "good.png"):
            disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=name)
        token = disk_storage.begin_delete([("bad.png", ""), ("good.png", "")])
        bad_path = disk_storage.get_path("bad.png")
        real_unlink = Path.unlink

        def unlink(self: Path, missing_ok: bool = False):
            if self == bad_path:
                raise OSError("device busy")
            return real_unlink(self, missing_ok=missing_ok)

        with patch.object(Path, "unlink", unlink), pytest.raises(ImageFileDeleteException):
            disk_storage.commit_delete(token)

        assert bad_path.exists()
        # The failure did not stop the rest of the purge...
        assert not disk_storage.get_path("good.png").exists()
        # ...and the journal is still there for startup recovery to finish.
        assert list(disk_storage.image_root.glob(".delete_*"))

        _restart(disk_storage, record_exists=False)

        assert not bad_path.exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_abandon_keeps_the_files_and_drops_the_journal(self, disk_storage: DiskImageFileStorage):
        """The record delete failed, so the image is still live and must be left completely alone."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="kept.png")
        token = disk_storage.begin_delete([("kept.png", "")])

        disk_storage.abandon_delete(token)

        assert disk_storage.get_path("kept.png").exists()
        assert disk_storage.get_path("kept.png", thumbnail=True).exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_startup_purges_journalled_files_whose_record_is_gone(self, disk_storage: DiskImageFileStorage):
        """The crash window records-first opens: records committed as deleted, purge never ran."""
        for name in ("orphan.png", "survivor.png"):
            disk_storage.save(image=Image.new("RGB", (32, 32)), image_name=name)
        disk_storage.begin_delete([("orphan.png", ""), ("survivor.png", "")])

        invoker = MagicMock()
        invoker.services.image_records.exists.side_effect = lambda name: name == "survivor.png"
        restarted = DiskImageFileStorage(disk_storage.image_root)
        restarted.start(invoker)

        assert not disk_storage.get_path("orphan.png").exists()
        assert not disk_storage.get_path("orphan.png", thumbnail=True).exists()
        # The record survived, so this one was never deleted and keeps its files.
        assert disk_storage.get_path("survivor.png").exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_rollback_purges_instead_of_restoring_when_the_record_is_gone(self, disk_storage: DiskImageFileStorage):
        """A staged delete that fails must not resurrect files another request has already
        unreferenced. Restoring them would strand them with no record and no journal to find
        them by (JPPhoto, PR #9361)."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="raced.png")
        token = disk_storage.stage_delete("raced.png")
        # Meanwhile another request deleted the record.
        disk_storage._DiskImageFileStorage__invoker.services.image_records.exists.return_value = False

        disk_storage.rollback_delete(token)

        assert not disk_storage.get_path("raced.png").exists()
        assert not disk_storage.get_path("raced.png", thumbnail=True).exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_rollback_restores_when_the_record_store_cannot_be_read(self, disk_storage: DiskImageFileStorage):
        """An unreadable database must never cost a live image its files."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="kept.png")
        token = disk_storage.stage_delete("kept.png")
        records = disk_storage._DiskImageFileStorage__invoker.services.image_records
        records.exists.side_effect = RuntimeError("database is locked")

        disk_storage.rollback_delete(token)

        assert disk_storage.get_path("kept.png").exists()
        assert disk_storage.get_path("kept.png", thumbnail=True).exists()


class TestJournalDurability:
    """The journal only makes a deletion recoverable if it outlives a power loss.

    SQLite fsyncs the record deletion, so a journal that is merely written — and not fsynced, both
    its manifest and its own directory entry in the output folder — can be lost while the record
    stays deleted, which is exactly the orphan the journal exists to prevent.
    """

    def _record_fsyncs(self, monkeypatch) -> list[Path]:
        fsynced: list[Path] = []
        monkeypatch.setattr(
            DiskImageFileStorage,
            "_DiskImageFileStorage__fsync_directory",
            staticmethod(lambda directory, **_: fsynced.append(Path(directory))),
        )
        return fsynced

    def test_begin_delete_fsyncs_the_journal_and_the_output_folder(
        self, disk_storage: DiskImageFileStorage, monkeypatch
    ):
        fsynced = self._record_fsyncs(monkeypatch)

        token = disk_storage.begin_delete([("img.png", "")])

        assert Path(token.directory) in fsynced
        assert disk_storage.image_root in [path.resolve() for path in fsynced]

    def test_stage_delete_fsyncs_before_it_moves_the_files(self, disk_storage: DiskImageFileStorage, monkeypatch):
        """The manifest has to be durable first, or a crash leaves staged files naming nothing."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="staged.png")
        moved: list[str] = []
        fsynced: list[Path] = []

        def record_fsync(directory):
            fsynced.append(Path(directory))

        real_replace = Path.replace

        def record_replace(self: Path, target):
            moved.append(str(target))
            assert fsynced, "the manifest was not made durable before the files were moved"
            return real_replace(self, target)

        monkeypatch.setattr(DiskImageFileStorage, "_DiskImageFileStorage__fsync_directory", staticmethod(record_fsync))
        with patch.object(Path, "replace", record_replace):
            token = disk_storage.stage_delete("staged.png")

        assert moved
        assert Path(token.directory) in fsynced
        assert disk_storage.image_root in [path.resolve() for path in fsynced]

    @posix_only
    def test_begin_delete_fails_when_the_journal_cannot_be_made_durable(self, disk_storage: DiskImageFileStorage):
        """A journal whose durability is unknown must not license a record deletion: the caller
        deletes records straight after this returns, so the failure has to surface here."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="live.png")

        with patch.object(os, "fsync", _failing_directory_fsync(errno.EIO)), pytest.raises(ImageFileDeleteException):
            disk_storage.begin_delete([("live.png", "")])

        assert not list(disk_storage.image_root.glob(".delete_*"))
        assert disk_storage.get_path("live.png").exists()
        assert disk_storage.get_path("live.png", thumbnail=True).exists()

    @posix_only
    def test_stage_delete_fails_before_moving_anything_when_the_journal_cannot_be_made_durable(
        self, disk_storage: DiskImageFileStorage
    ):
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="live.png")

        with patch.object(os, "fsync", _failing_directory_fsync(errno.EIO)), pytest.raises(ImageFileDeleteException):
            disk_storage.stage_delete("live.png")

        assert not list(disk_storage.image_root.glob(".delete_*"))
        assert disk_storage.get_path("live.png").exists()
        assert disk_storage.get_path("live.png", thumbnail=True).exists()

    @posix_only
    @pytest.mark.parametrize("error_number", [errno.EINVAL, errno.ENOTSUP])
    def test_a_filesystem_that_cannot_sync_directories_is_not_a_failure(
        self, disk_storage: DiskImageFileStorage, error_number: int
    ):
        """Some filesystems refuse directory fsync outright; there is nothing more to be had from
        them, and refusing every delete on such a volume would be worse than the risk."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="gone.png")

        with patch.object(os, "fsync", _failing_directory_fsync(error_number)):
            token = disk_storage.begin_delete([("gone.png", "")])
            disk_storage.commit_delete(token)

        assert not disk_storage.get_path("gone.png").exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))


class TestPurgeDurability:
    """The journal may only be dropped once the purge it describes is on disk.

    Unlinks live in the parent directories' entries and nothing else syncs those. A filesystem that
    persists the journal's removal ahead of the unlinks brings the files back after a power loss
    with no journal left to find them by — the permanent orphan the journal exists to prevent.
    """

    SUBFOLDER = "2026/09/04"

    def _parents(self, disk_storage: DiskImageFileStorage, image_name: str) -> set[Path]:
        return {
            disk_storage.get_path(image_name, image_subfolder=self.SUBFOLDER).parent.resolve(),
            disk_storage.get_path(image_name, thumbnail=True, image_subfolder=self.SUBFOLDER).parent.resolve(),
        }

    def _assert_parents_synced_before_journal_removal(self, disk_storage, monkeypatch, image_name: str):
        """Returns the list the patched fsync records into; rmtree of the journal asserts on it."""
        fsynced: list[Path] = []
        monkeypatch.setattr(
            DiskImageFileStorage,
            "_DiskImageFileStorage__fsync_directory",
            staticmethod(lambda directory, **_: fsynced.append(Path(directory).resolve())),
        )
        real_rmtree = shutil.rmtree
        parents = self._parents(disk_storage, image_name)

        def rmtree(path, *args, **kwargs):
            if Path(path).name.startswith(".delete_"):
                assert parents <= set(fsynced), "the journal was dropped before the purge was made durable"
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", rmtree)
        return fsynced

    def test_pending_commit_syncs_the_purged_directories_before_dropping_the_journal(
        self, disk_storage: DiskImageFileStorage, monkeypatch
    ):
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="gone.png", image_subfolder=self.SUBFOLDER)
        token = disk_storage.begin_delete([("gone.png", self.SUBFOLDER)])
        fsynced = self._assert_parents_synced_before_journal_removal(disk_storage, monkeypatch, "gone.png")

        disk_storage.commit_delete(token)

        assert self._parents(disk_storage, "gone.png") <= set(fsynced)
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_staged_commit_syncs_the_purged_directories_before_dropping_the_journal(
        self, disk_storage: DiskImageFileStorage, monkeypatch
    ):
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="gone.png", image_subfolder=self.SUBFOLDER)
        token = disk_storage.stage_delete("gone.png", image_subfolder=self.SUBFOLDER)
        fsynced = self._assert_parents_synced_before_journal_removal(disk_storage, monkeypatch, "gone.png")

        disk_storage.commit_delete(token)

        assert self._parents(disk_storage, "gone.png") <= set(fsynced)
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_rollback_syncs_the_restored_directories_before_dropping_the_journal(
        self, disk_storage: DiskImageFileStorage, monkeypatch
    ):
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="back.png", image_subfolder=self.SUBFOLDER)
        token = disk_storage.stage_delete("back.png", image_subfolder=self.SUBFOLDER)
        fsynced = self._assert_parents_synced_before_journal_removal(disk_storage, monkeypatch, "back.png")

        disk_storage.rollback_delete(token)

        assert self._parents(disk_storage, "back.png") <= set(fsynced)
        assert disk_storage.get_path("back.png", image_subfolder=self.SUBFOLDER).exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_startup_syncs_before_dropping_a_recovered_journal(self, disk_storage: DiskImageFileStorage, monkeypatch):
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="orphan.png", image_subfolder=self.SUBFOLDER)
        disk_storage.begin_delete([("orphan.png", self.SUBFOLDER)])
        fsynced = self._assert_parents_synced_before_journal_removal(disk_storage, monkeypatch, "orphan.png")

        _restart(disk_storage, record_exists=False)

        assert self._parents(disk_storage, "orphan.png") <= set(fsynced)
        assert not disk_storage.get_path("orphan.png", image_subfolder=self.SUBFOLDER).exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    @posix_only
    def test_commit_keeps_the_journal_when_the_purge_cannot_be_made_durable(self, disk_storage: DiskImageFileStorage):
        """The records are already gone, so the journal is the only thing that can redo a purge the
        disk lost. It stays, and the next startup finishes the job."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="gone.png", image_subfolder=self.SUBFOLDER)
        token = disk_storage.begin_delete([("gone.png", self.SUBFOLDER)])

        with patch.object(os, "fsync", _failing_directory_fsync(errno.EIO)), pytest.raises(ImageFileDeleteException):
            disk_storage.commit_delete(token)

        assert list(disk_storage.image_root.glob(".delete_*"))

        _restart(disk_storage, record_exists=False)

        assert not disk_storage.get_path("gone.png", image_subfolder=self.SUBFOLDER).exists()
        assert not disk_storage.get_path("gone.png", thumbnail=True, image_subfolder=self.SUBFOLDER).exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    @posix_only
    def test_staged_commit_keeps_the_journal_when_the_purge_cannot_be_made_durable(
        self, disk_storage: DiskImageFileStorage
    ):
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="gone.png")
        token = disk_storage.stage_delete("gone.png")

        with patch.object(os, "fsync", _failing_directory_fsync(errno.EIO)), pytest.raises(ImageFileDeleteException):
            disk_storage.commit_delete(token)

        assert list(disk_storage.image_root.glob(".delete_*"))

        _restart(disk_storage, record_exists=False)

        assert not disk_storage.get_path("gone.png").exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    @posix_only
    def test_startup_keeps_the_journal_when_the_purge_cannot_be_made_durable(self, disk_storage: DiskImageFileStorage):
        """Injected fsync failure at the recovery step itself: the journal must survive that
        startup and be finished by the next one."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="orphan.png")
        disk_storage.begin_delete([("orphan.png", "")])

        with patch.object(os, "fsync", _failing_directory_fsync(errno.EIO)):
            _restart(disk_storage, record_exists=False)

        assert list(disk_storage.image_root.glob(".delete_*"))

        _restart(disk_storage, record_exists=False)

        assert not disk_storage.get_path("orphan.png").exists()
        assert not list(disk_storage.image_root.glob(".delete_*"))

    def test_a_subfolder_that_no_longer_exists_does_not_block_the_commit(self, disk_storage: DiskImageFileStorage):
        """A journal can name a subfolder whose directory is gone entirely (moved or already purged);
        there are no entries there to make durable, so the commit completes."""
        token = disk_storage.begin_delete([("ghost.png", "never/made")])

        disk_storage.commit_delete(token)

        assert not list(disk_storage.image_root.glob(".delete_*"))


class TestRecoveryToleratesStrayEntries:
    """Recovery runs during start(); anything it cannot make sense of must not stop the app."""

    def test_a_stray_file_does_not_stop_startup(self, disk_storage: DiskImageFileStorage):
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="live.png")
        (disk_storage.image_root / ".delete_stray").write_text("not a directory")

        _restart(disk_storage, record_exists=True)

        assert disk_storage.get_path("live.png").exists()

    def test_a_journal_with_no_manifest_is_left_alone(self, disk_storage: DiskImageFileStorage):
        """Its contents cannot be attributed to an image, so removing them would destroy data."""
        disk_storage.save(image=Image.new("RGB", (32, 32)), image_name="live.png")
        orphan_journal = disk_storage.image_root / ".delete_nomanifest"
        orphan_journal.mkdir()
        (orphan_journal / "0").write_bytes(b"staged image bytes")

        _restart(disk_storage, record_exists=True)

        assert (orphan_journal / "0").read_bytes() == b"staged image bytes"
        assert disk_storage.get_path("live.png").exists()

    def test_an_empty_journal_directory_is_removed(self, disk_storage: DiskImageFileStorage):
        (disk_storage.image_root / ".delete_empty").mkdir()

        _restart(disk_storage, record_exists=True)

        assert not list(disk_storage.image_root.glob(".delete_*"))
