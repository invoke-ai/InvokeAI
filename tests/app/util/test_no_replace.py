import errno
from pathlib import Path
from typing import BinaryIO

import pytest

from invokeai.app.util import no_replace
from invokeai.app.util.no_replace import publish_noreplace, rename_noreplace


def _refuse_link(src: Path, dst: Path) -> None:
    """Refuse hard links the way exFAT does on Linux."""
    raise PermissionError(errno.EPERM, "Operation not permitted", str(dst))


@pytest.fixture
def without_hard_links(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(no_replace.os, "link", _refuse_link)


def _refuse_no_replace_rename(src: Path, dst: Path) -> None:
    """Refuse an atomic no-replace rename the way macOS exFAT, NFS and many FUSE mounts do."""
    raise OSError(errno.ENOTSUP, "Operation not supported", str(dst))


def _limit_mount(monkeypatch: pytest.MonkeyPatch, mount: str) -> None:
    if mount in ("no-hard-links", "copy-only"):
        monkeypatch.setattr(no_replace.os, "link", _refuse_link)
    if mount == "copy-only":
        monkeypatch.setattr(no_replace, "rename_noreplace", _refuse_no_replace_rename)


def test_publish_gives_the_file_its_final_name_and_keeps_the_source_name(tmp_path: Path) -> None:
    src = tmp_path / "work.tmp"
    src.write_bytes(b"complete payload")
    dst = tmp_path / "published"

    publish_noreplace(src, dst)

    assert dst.read_bytes() == b"complete payload"
    assert src.exists()


def test_publish_without_hard_links_moves_the_file(tmp_path: Path, without_hard_links: None) -> None:
    src = tmp_path / "work.tmp"
    src.write_bytes(b"complete payload")
    dst = tmp_path / "published"

    publish_noreplace(src, dst)

    assert dst.read_bytes() == b"complete payload"
    assert not src.exists()


def test_publish_without_link_or_no_replace_rename_copies_and_keeps_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _limit_mount(monkeypatch, "copy-only")
    src = tmp_path / "work.tmp"
    src.write_bytes(b"complete payload")
    dst = tmp_path / "published"

    publish_noreplace(src, dst)

    assert dst.read_bytes() == b"complete payload"
    assert src.read_bytes() == b"complete payload"


@pytest.mark.parametrize("mount", ["hard-links", "no-hard-links", "copy-only"])
def test_publish_refuses_an_existing_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mount: str) -> None:
    _limit_mount(monkeypatch, mount)
    src = tmp_path / "work.tmp"
    src.write_bytes(b"new payload")
    dst = tmp_path / "published"
    dst.write_bytes(b"other owner")

    with pytest.raises(FileExistsError):
        publish_noreplace(src, dst)

    assert dst.read_bytes() == b"other owner"
    assert src.read_bytes() == b"new payload"


def test_a_failed_copy_leaves_no_destination_and_reports_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _limit_mount(monkeypatch, "copy-only")
    src = tmp_path / "work.tmp"
    src.write_bytes(b"complete payload")
    dst = tmp_path / "published"

    def fail_midway(source: BinaryIO, target: BinaryIO) -> None:
        target.write(source.read(4))
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(no_replace.shutil, "copyfileobj", fail_midway)

    with pytest.raises(OSError) as exc_info:
        publish_noreplace(src, dst)

    assert exc_info.value.errno == errno.ENOSPC
    assert not dst.exists()
    assert src.read_bytes() == b"complete payload"
    rename_error = exc_info.value.__cause__
    assert isinstance(rename_error, OSError) and rename_error.errno == errno.ENOTSUP
    assert isinstance(rename_error.__cause__, PermissionError)


def test_rename_refuses_an_existing_destination(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.write_bytes(b"source")
    dst = tmp_path / "dst"
    dst.write_bytes(b"destination")

    with pytest.raises(FileExistsError):
        rename_noreplace(src, dst)

    assert src.read_bytes() == b"source"
    assert dst.read_bytes() == b"destination"


def test_rename_moves_a_directory_with_its_contents(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "weights.safetensors").write_bytes(b"weights")
    dst = tmp_path / "dst"

    rename_noreplace(src, dst)

    assert not src.exists()
    assert (dst / "weights.safetensors").read_bytes() == b"weights"
