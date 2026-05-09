import platform
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
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
        thumb_path = disk_storage.get_path(thumbnail_name, thumbnail=True, image_subfolder=subfolder)
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
