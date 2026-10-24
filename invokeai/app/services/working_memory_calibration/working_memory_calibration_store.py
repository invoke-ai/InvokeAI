"""The calibration store: one JSON file per device class under `<db_dir>/working_memory/`.

Machine data, not account data, so it lives beside the database rather than in it: a device class's measurements stay
valid whichever database the root points at. Every change re-reads the file and is applied to what is on disk under a
process lock and an OS file lock, so two processes sharing a root never lose each other's writes. Files are replaced
atomically. A file this version cannot read -- another schema version, invalid content, a field it does not know, or a device
class that does not match its name -- is reported and never written, moved or deleted.
"""

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import IO, Final, Optional, TypeVar

from pydantic import ValidationError

from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    STORE_SCHEMA_VERSION,
    CalibrationDeviceClass,
    CalibrationStoreError,
    CalibrationStoreFile,
    CalibrationStoreUnusableError,
)

LOCK_TIMEOUT_S: float = 30.0
"""How long a change waits for another process's lock on the same file before it gives up."""
LOCK_RETRY_S: float = 0.05
REPLACE_ATTEMPTS: int = 6
"""How often a write tries to replace the file. On Windows the replace fails with PermissionError while a process that
does not take the store's lock (an antivirus scanner, a search indexer, an editor) has the file open; such a handle is
usually gone within a fraction of a second."""
REPLACE_RETRY_S: float = 0.05
"""The first wait between those tries; it doubles each time (about 1.5 s in all)."""

_FILE_NAME: Final = re.compile(r"^[0-9a-f]{16}\.json$")

T = TypeVar("T")


class StoreRead:
    """A device class's file as read: its content, or why it cannot be used."""

    def __init__(self, device_class_id: str, file: Optional[CalibrationStoreFile], problem: Optional[str]) -> None:
        self.device_class_id = device_class_id
        self.file = file
        self.problem = problem


def _parse(raw: bytes, device_class_id: str) -> tuple[Optional[CalibrationStoreFile], Optional[str]]:
    try:
        data = json.loads(raw)
    except ValueError:
        return None, "The file is not valid JSON."
    if not isinstance(data, dict):
        return None, "The file does not hold a calibration store."
    version = data.get("schema_version")
    if version != STORE_SCHEMA_VERSION:
        return None, f"The file has schema version {version!r}; this version of Invoke reads {STORE_SCHEMA_VERSION}."
    try:
        file = CalibrationStoreFile.model_validate(data)
    except ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        return None, f"The file failed validation ({error.error_count()} errors; first at {location}: {first['msg']})."
    if file.device_class.id != device_class_id:
        return None, f"The file holds device class {file.device_class.id}, not the {device_class_id} its name says."
    return file, None


@contextmanager
def _os_lock(handle: IO[bytes]) -> Iterator[None]:
    if os.name == "nt":
        import msvcrt

        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise CalibrationStoreError(f"Timed out waiting for the lock on {handle.name}") from None
                time.sleep(LOCK_RETRY_S)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        deadline = time.monotonic() + LOCK_TIMEOUT_S
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise CalibrationStoreError(f"Timed out waiting for the lock on {handle.name}") from None
                time.sleep(LOCK_RETRY_S)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _replace(staging: str, path: Path) -> None:
    """`os.replace`, retried while another process's handle on the target refuses it (Windows only)."""
    delay = REPLACE_RETRY_S
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(staging, path)
            return
        except PermissionError:
            if attempt + 1 == REPLACE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay *= 2


class CalibrationStore:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return self._dir

    def path_for(self, device_class_id: str) -> Path:
        return self._dir / f"{device_class_id}.json"

    def device_class_ids(self) -> list[str]:
        """The device classes that have a file, readable or not."""
        if not self._dir.is_dir():
            return []
        try:
            return sorted(path.stem for path in self._dir.iterdir() if _FILE_NAME.match(path.name) and path.is_file())
        except OSError as error:
            raise CalibrationStoreError(f"Cannot list {self._dir}: {error}") from error

    def write_problem(self, device_class_id: str) -> Optional[str]:
        """Why a change of the device class's file could not be written now, or None.

        Checks what a write needs without changing anything that lasts: a file can be created in the directory (or,
        while there is none, in the nearest folder that exists, where the write would create it); the lock file can be
        opened for writing; and, on Windows, the file is not read-only, which refuses the replace there. The file it
        creates is a temporary one the system deletes, also should the process stop.
        """
        folder = self._dir
        while not folder.exists() and folder.parent != folder:
            folder = folder.parent
        try:
            with tempfile.TemporaryFile(dir=folder, prefix=".invokeai-write-check."):
                pass
        except OSError as error:
            return f"Cannot write in {folder}: {error}"
        lock_path = self._dir / f"{device_class_id}.json.lock"
        if lock_path.exists() and not os.access(lock_path, os.W_OK):
            return f"{lock_path} is read-only"
        path = self.path_for(device_class_id)
        if os.name == "nt" and path.exists() and not os.access(path, os.W_OK):
            return f"{path} is read-only"
        return None

    @contextmanager
    def _locked(self, device_class_id: str, *, writing: bool) -> Iterator[None]:
        """Hold this process's lock and the OS lock on the file's `.lock` sibling.

        Readers lock too: on Windows a file another process holds open cannot be replaced. A reader that cannot create
        the lock file (a read-only root) reads without it, since nobody can write there either.
        """
        with self._lock:
            lock_path = self._dir / f"{device_class_id}.json.lock"
            handle: Optional[IO[bytes]] = None
            try:
                if writing:
                    self._dir.mkdir(parents=True, exist_ok=True)
                handle = open(lock_path, "a+b")
            except OSError as error:
                if writing:
                    raise CalibrationStoreError(f"Cannot open {lock_path}: {error}") from error
            if handle is None:
                yield
                return
            with handle, _os_lock(handle):
                yield

    @contextmanager
    def creation_lock(self, run_id: str) -> Iterator[None]:
        """Hold the OS lock of one run's creation, which every process serving this root takes on the same file. A
        holder that dies releases it with its process. Waits `LOCK_TIMEOUT_S` for another holder, then raises
        `CalibrationStoreError`.

        The run ids share 256 lock files, `creations/<xx>.lock` by the first two hex digits of the id's SHA-256: the
        folder never holds more, so no file is ever removed (removing one could let a process that opened it before the
        removal and one that creates it anew each hold a lock of their own). Unrelated runs on one file wait for each
        other, which costs little: a creation is a short administrator's action. No store file has such a name.
        """
        stripe = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:2]
        lock_path = self._dir / "creations" / f"{stripe}.lock"
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(lock_path, "a+b")
        except OSError as error:
            raise CalibrationStoreError(f"Cannot open {lock_path}: {error}") from error
        with handle, _os_lock(handle):
            yield

    def _read_unlocked(self, device_class_id: str) -> Optional[StoreRead]:
        path = self.path_for(device_class_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise CalibrationStoreError(f"Cannot read {path}: {error}") from error
        file, problem = _parse(raw, device_class_id)
        return StoreRead(device_class_id, file, problem)

    def read(self, device_class_id: str) -> Optional[StoreRead]:
        """The device class's file, or None when it has none."""
        if not self.path_for(device_class_id).exists():
            return None
        with self._locked(device_class_id, writing=False):
            return self._read_unlocked(device_class_id)

    def read_raw(self, device_class_id: str) -> Optional[bytes]:
        """The file's bytes as they are, readable or not; None when it has none."""
        path = self.path_for(device_class_id)
        if not path.exists():
            return None
        with self._locked(device_class_id, writing=False):
            try:
                return path.read_bytes()
            except FileNotFoundError:
                return None
            except OSError as error:
                raise CalibrationStoreError(f"Cannot read {path}: {error}") from error

    def change(
        self,
        device_class_id: str,
        apply: Callable[[CalibrationStoreFile], tuple[bool, T]],
        *,
        create: Optional[CalibrationDeviceClass] = None,
    ) -> Optional[T]:
        """Apply `apply` to the file as it is on disk now, and write the file when `apply` says it changed it.

        `apply` returns whether it changed the file, and the result to return. Without a file, `create` names the
        device class to start one for; without `create` nothing is written and the result is None. Raises
        `CalibrationStoreUnusableError` for a file that cannot be read, which stays as it is.
        """
        with self._locked(device_class_id, writing=True):
            current = self._read_unlocked(device_class_id)
            if current is None:
                if create is None:
                    return None
                assert create.id == device_class_id
                file = CalibrationStoreFile(schema_version=STORE_SCHEMA_VERSION, device_class=create)
            elif current.file is None:
                raise CalibrationStoreUnusableError(
                    f"{self.path_for(device_class_id)} is kept as it is and not written: {current.problem}"
                )
            else:
                file = current.file
            changed, result = apply(file)
            if changed:
                self._write(self.path_for(device_class_id), file)
            return result

    @staticmethod
    def _write(path: Path, file: CalibrationStoreFile) -> None:
        payload = file.model_dump_json(indent=1).encode("utf-8")
        try:
            descriptor, staging = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
        except OSError as error:
            raise CalibrationStoreError(f"Cannot write {path}: {error}") from error
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _replace(staging, path)
        except BaseException as error:
            with suppress(OSError):
                os.unlink(staging)
            if isinstance(error, OSError):
                raise CalibrationStoreError(f"Cannot write {path}: {error}") from error
            raise
