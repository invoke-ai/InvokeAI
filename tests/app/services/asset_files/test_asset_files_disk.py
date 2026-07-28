import platform
from pathlib import Path
from unittest.mock import MagicMock

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


def test_start_sweeps_orphaned_splats(tmp_path: Path):
    """Splats are session-transient with no DB records, so server startup wipes anything left over
    from a previous run — this is the storage's only GC."""
    (tmp_path / "orphan-1.ply").write_bytes(b"a")
    (tmp_path / "orphan-2.splat").write_bytes(b"b")
    (tmp_path / "orphan-3.spz").write_bytes(b"c")
    subdir = tmp_path / "unrelated-dir"
    subdir.mkdir()

    storage = DiskAssetFileStorage(tmp_path)
    storage.start(MagicMock())

    assert list(tmp_path.glob("*.ply")) == []
    assert list(tmp_path.glob("*.splat")) == []
    assert list(tmp_path.glob("*.spz")) == []
    assert subdir.exists()  # only files are swept


def test_sweep_only_deletes_splat_formats(tmp_path: Path):
    """The sweep must not delete non-splat files: a future asset type stored via this service has to
    opt into the transient lifecycle explicitly, not inherit it."""
    (tmp_path / "splat.ply").write_bytes(b"a")
    (tmp_path / "future-mesh.glb").write_bytes(b"b")

    storage = DiskAssetFileStorage(tmp_path)
    storage.start(MagicMock())

    assert not (tmp_path / "splat.ply").exists()
    assert (tmp_path / "future-mesh.glb").exists()


def test_construction_does_not_sweep(tmp_path: Path):
    """Only start() (server boot) sweeps — bare construction must leave existing files alone."""
    (tmp_path / "existing.ply").write_bytes(b"a")

    DiskAssetFileStorage(tmp_path)

    assert (tmp_path / "existing.ply").exists()
