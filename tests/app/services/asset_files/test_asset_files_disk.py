import platform
from pathlib import Path

import pytest

from invokeai.app.services.asset_files.asset_files_common import AssetFileNotFoundException
from invokeai.app.services.asset_files.asset_files_disk import DiskAssetFileStorage


@pytest.fixture
def disk_storage(tmp_path: Path) -> DiskAssetFileStorage:
    return DiskAssetFileStorage(tmp_path)


# Anything that is not a bare filename must be rejected. Backslash cases are Windows-only: on POSIX a
# backslash is an ordinary filename character, not a separator, so those names are harmless there.
TRAVERSAL_NAMES = [
    "",
    ".",
    "..",
    "../secret",
    "a/b",
    "/etc/passwd",
    "foo/../bar.ply",
]

if platform.system() == "Windows":
    TRAVERSAL_NAMES += [
        "a\\b",
        "..\\secret",
        "C:\\evil\\x.ply",
    ]


@pytest.mark.parametrize("asset_name", TRAVERSAL_NAMES)
def test_resolve_path_rejects_non_bare_names(disk_storage: DiskAssetFileStorage, asset_name: str):
    """_resolve_path is the only defense against path traversal on GET /api/v1/assets/i/{asset_name} —
    a regression here is a direct file-disclosure risk, so each rejection case is pinned down."""
    with pytest.raises(ValueError, match="Invalid asset name"):
        disk_storage._resolve_path(asset_name)


def test_resolve_path_accepts_bare_filename(disk_storage: DiskAssetFileStorage, tmp_path: Path):
    path = disk_storage._resolve_path("splat.ply")
    assert path == (tmp_path / "splat.ply").resolve()


def test_save_and_get_path_round_trip(disk_storage: DiskAssetFileStorage, tmp_path: Path):
    disk_storage.save("splat.ply", b"ply-bytes")
    path = disk_storage.get_path("splat.ply")
    assert path.read_bytes() == b"ply-bytes"
    assert path.parent == tmp_path.resolve()


def test_get_path_missing_asset_raises(disk_storage: DiskAssetFileStorage):
    with pytest.raises(AssetFileNotFoundException):
        disk_storage.get_path("missing.ply")


def test_delete_removes_file(disk_storage: DiskAssetFileStorage):
    disk_storage.save("splat.ply", b"ply-bytes")
    disk_storage.delete("splat.ply")
    with pytest.raises(AssetFileNotFoundException):
        disk_storage.get_path("splat.ply")
