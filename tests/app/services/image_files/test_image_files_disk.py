import hashlib
import platform
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage, _should_use_png_rle
from invokeai.app.util.thumbnails import get_thumbnail_name


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
