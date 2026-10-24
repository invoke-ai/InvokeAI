"""Give a path its final name without ever replacing a path that already holds that name."""

import contextlib
import ctypes
import os
import shutil
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


def publish_noreplace(src: Path, dst: Path) -> None:
    """Give the regular file src the name dst; raises FileExistsError, leaving dst untouched, when dst exists.

    Where hard links work, dst becomes a second name of src, src keeps its name and the caller removes it. Without hard
    links (exFAT, FAT32, many network shares) src is renamed to dst with an atomic no-replace rename and no longer
    exists. Where neither works (macOS exFAT, NFS, many FUSE mounts), dst is created exclusively and src's bytes copied
    into it; src keeps its name. Only in that last case can another reader briefly see a partially written dst. Callers
    tolerate that: a claim counts by existence and a partial payload is preserved as malformed; thumbnails, archives,
    restores and database copies are re-validated or transient.
    """
    try:
        os.link(src, dst)
        return
    except FileExistsError:
        raise
    except OSError as link_error:
        try:
            rename_noreplace(src, dst)
            return
        except FileExistsError:
            raise
        except OSError as rename_error:
            rename_error.__cause__ = link_error
            fallback_error = rename_error
    try:
        with open(src, "rb") as source:
            fd = os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o666)
            try:
                with os.fdopen(fd, "wb") as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(dst)
                raise
    except FileExistsError:
        raise
    except OSError as copy_error:
        raise copy_error from fallback_error
