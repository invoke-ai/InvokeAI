"""Give a path its final name without ever replacing a path that already holds that name."""

import contextlib
import ctypes
import os
import shutil
import stat
import sys
from pathlib import Path


def rename_noreplace(src: Path, dst: Path) -> None:
    """Atomically rename a file or directory; raises FileExistsError when dst exists, OSError when unsupported."""
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError("atomic no-replace rename is unavailable")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(src), -100, os.fsencode(dst), 1)  # AT_FDCWD, RENAME_NOREPLACE
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(dst))
        return
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError("atomic no-replace rename is unavailable")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(src), os.fsencode(dst), 0x00000004)  # RENAME_EXCL
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(dst))
        return
    if os.name == "nt":
        # Windows rename fails when the destination already exists.
        os.rename(src, dst)
        return
    raise OSError("atomic no-replace rename is unavailable on this platform")


FileSignature = tuple[int, int, int, int]


def file_signature(result: os.stat_result) -> FileSignature:
    """Device, inode, size and modification time: what tells this module's published file apart from a replacement.

    The inode alone is weak evidence on FAT-family drivers (macOS msdosfs and exFAT among them), which derive file IDs
    from where a file sits, so a replacement can carry the ID of the file it replaced.
    """
    return result.st_dev, result.st_ino, result.st_size, result.st_mtime_ns


def publish_noreplace(src: Path, dst: Path) -> FileSignature:
    """Give the regular file src the name dst; raises FileExistsError, leaving dst untouched, when dst exists.

    Where hard links work, dst becomes a second name of src, src keeps its name and the caller removes it. Without hard
    links (exFAT, FAT32, many network shares) src is renamed to dst with an atomic no-replace rename and no longer
    exists. Where neither works (macOS exFAT, NFS, many FUSE mounts), dst is created exclusively with src's permission
    bits and src's bytes copied into it; src keeps its name. Only in that last case can another reader briefly see a
    partially written dst. Callers tolerate that: a claim counts by existence and a partial payload is preserved as
    malformed; thumbnails, archives, restores and database copies are re-validated or transient. A failed copy removes
    dst only while dst is still the file it created.

    Returns the file_signature of the file published as dst, so a caller can compare file_signature(dst.lstat()) later
    to tell whether dst still names it: taken from src before a link or rename, which change neither size nor
    modification time, and from the copy after a copy. On Windows FAT and exFAT a rename to a longer name can change the
    file ID, so there the signature can differ even though dst still names the published file; the thumbnail rollback,
    which compares it, never runs on Windows, whose directory fsync is a no-op.
    """
    # Stat src, which the caller owns, rather than dst, which another writer could replace once it is published.
    source = file_signature(os.lstat(src))
    try:
        os.link(src, dst)
        return source
    except FileExistsError:
        raise
    except OSError as link_error:
        try:
            rename_noreplace(src, dst)
            return source
        except FileExistsError:
            raise
        except OSError as rename_error:
            rename_error.__cause__ = link_error
            fallback_error = rename_error
    try:
        return _copy_noreplace(src, dst)
    except FileExistsError:
        raise
    except OSError as copy_error:
        raise copy_error from fallback_error


def _copy_noreplace(src: Path, dst: Path) -> FileSignature:
    """Create dst exclusively and copy src into it; returns the copy's file_signature."""
    with open(src, "rb") as source:
        # The umask can only narrow src's permission bits, so the copy is never more accessible than src.
        mode = stat.S_IMODE(os.fstat(source.fileno()).st_mode)
        fd = os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), mode)
        try:
            # The writer leaves fd open: its close flushes buffered bytes, also when the copy failed, and fd then
            # still identifies the copy as finally written.
            with os.fdopen(fd, "wb", closefd=False) as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(fd)
            copied = file_signature(os.fstat(fd))
        except BaseException:
            _remove_failed_copy(fd, dst)
            raise
        os.close(fd)
    return copied


def _remove_failed_copy(fd: int, dst: Path) -> None:
    """Unlink dst only while it still names the failed copy behind fd, and close fd; keeps dst on any doubt."""
    # POSIX keeps fd open through the check and the unlink: no replacement can reuse the copy's inode, and times a
    # filesystem updates on close cannot make the copy look foreign. Windows cannot unlink an open file and closes
    # first. Left open, for writers outside InvokeAI (which never replaces dst): a replacement between the check and
    # the unlink, which nothing prevents without hard links or no-replace rename; and, where a replacement can carry
    # the copy's ID (FAT drivers, Windows after the close), a same-size one within one mtime step (2 s on FAT32).
    open_fd: int | None = fd
    try:
        with contextlib.suppress(OSError):
            copy = file_signature(os.fstat(fd))
            if os.name == "nt":
                open_fd = None
                os.close(fd)
            if file_signature(os.lstat(dst)) == copy:
                if os.name == "nt":
                    # Windows refuses to unlink a read-only file, which the copy of a read-only src is.
                    os.chmod(dst, stat.S_IWRITE)
                os.unlink(dst)
    finally:
        if open_fd is not None:
            with contextlib.suppress(OSError):
                os.close(open_fd)
