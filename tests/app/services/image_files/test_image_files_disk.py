import platform
from pathlib import Path

import pytest

from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage


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
