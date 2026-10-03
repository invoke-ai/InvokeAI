"""Authenticated, manifest-based LAN transfer of local model directories.

The primary serves files from a short-lived, random-token HTTP endpoint. The
remote InvokeAI process stages and verifies every file before asking its normal
model installer to register the directory. No separate worker plugin is needed.
"""

from __future__ import annotations

import hashlib
import mimetypes
import secrets
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

from invokeai.app.invocations.remote_worker.model_transfer import ModelTransferError, _route_local_ip

_MAX_FILES = 8192
_MAX_BYTES = 2 * 1024**4
_CHUNK = 1024 * 1024
_JOB_LOCK = threading.Lock()
_JOBS: dict[int, dict[str, Any]] = {}
_CANCEL_EVENTS: dict[int, threading.Event] = {}
_ACTIVE_HASHES: dict[str, int] = {}
_NEXT_JOB_ID = 0


def _sha256_file(path: Path, should_cancel: Callable[[], bool] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if should_cancel is not None and should_cancel():
                raise ModelTransferError("Model directory preparation cancelled")
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(text: str) -> PurePosixPath:
    if not text or len(text) > 1024 or "\\" in text or ":" in text or "\x00" in text:
        raise ValueError("Invalid model file path")
    if any(ord(char) < 32 for char in text):
        raise ValueError("Control character in model file path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} or part.endswith((" ", ".")) for part in text.split("/"))
        or path.as_posix() != text
    ):
        raise ValueError("Unsafe model file path")
    return path


@dataclass(frozen=True)
class DirectoryFile:
    path: str
    size: int
    sha256: str


def inventory_directory(
    root: Path,
    should_cancel: Callable[[], bool] | None = None,
) -> list[DirectoryFile]:
    """Inventory only regular files, refusing symlinks and case-fold collisions."""
    if root.is_symlink() or not root.is_dir():
        raise ModelTransferError("Source must be a real model directory")
    children: list[Path] = []
    for child in root.rglob("*"):
        if should_cancel is not None and should_cancel():
            raise ModelTransferError("Model directory preparation cancelled")
        children.append(child)
    children.sort()

    files: list[DirectoryFile] = []
    total = 0
    names: set[str] = set()
    for child in children:
        if should_cancel is not None and should_cancel():
            raise ModelTransferError("Model directory preparation cancelled")
        if child.is_symlink():
            raise ModelTransferError(f"Model contains a symlink: {child}")
        if child.is_dir():
            continue
        if not child.is_file():
            raise ModelTransferError(f"Model contains a non-file entry: {child}")
        relative = child.relative_to(root).as_posix()
        try:
            _safe_path(relative)
        except ValueError as exc:
            raise ModelTransferError(f"Unsafe model path: {relative}") from exc
        folded = relative.casefold()
        if folded in names:
            raise ModelTransferError(f"Case-insensitive duplicate model file: {relative}")
        names.add(folded)
        size = child.stat().st_size
        total += size
        if len(files) >= _MAX_FILES or total > _MAX_BYTES:
            raise ModelTransferError("Model directory exceeds transfer limits")
        files.append(DirectoryFile(relative, size, _sha256_file(child, should_cancel)))
    if not files:
        raise ModelTransferError("Model directory has no files")
    return files


# Match the filename/configuration signals accepted by InvokeAI's model probe.
# This is an early, deliberately conservative diagnostic, not a replacement for
# the model installer: a file passing this check may still be invalid or incomplete.
_MODEL_WEIGHT_SUFFIXES = frozenset({".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"})
_MODEL_CONFIG_NAMES = frozenset({"config.json", "model_index.json", "modular_model_index.json"})


def _validate_transferable_model_files(files: list[DirectoryFile]) -> None:
    """Reject tokenizer-only/empty-weight directories before serving them over LAN."""
    if any(
        file.size > 0
        and (
            PurePosixPath(file.path).suffix.lower() in _MODEL_WEIGHT_SUFFIXES
            or PurePosixPath(file.path).name.lower() in _MODEL_CONFIG_NAMES
        )
        for file in files
    ):
        return
    raise ModelTransferError(
        "Local model directory appears incomplete: no non-empty model weights or supported model "
        "configuration files were found, including in subdirectories. Repair or reinstall the model "
        "on the primary InvokeAI instance before retrying remote transfer."
    )


class TemporaryDirectoryModelServer:
    """Expose only inventoried files at unguessable URLs for a transfer's lifetime."""

    def __init__(
        self,
        path: Path,
        remote_url: str,
        advertise_host: str = "",
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.path = path
        self.remote_url = remote_url
        self.advertise_host = advertise_host.strip()
        self.should_cancel = should_cancel
        self.files: list[DirectoryFile] = []
        self.url = ""
        self._token = secrets.token_urlsafe(32)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "TemporaryDirectoryModelServer":
        self.files = inventory_directory(self.path, self.should_cancel)
        _validate_transferable_model_files(self.files)
        file_paths = {
            f"/{self._token}/{urllib.parse.quote(item.path, safe='/')}": self.path / item.path for item in self.files
        }

        class Handler(BaseHTTPRequestHandler):
            server_version = "InvokeAIRemoteWorkerDirectoryTransfer/1.0"

            def log_message(self, _format: str, *args: Any) -> None:
                return

            def _headers(self) -> tuple[Path, int, int] | None:
                file_path = file_paths.get(urllib.parse.urlsplit(self.path).path)
                if file_path is None or not file_path.is_file() or file_path.is_symlink():
                    self.send_error(404)
                    return None
                size = file_path.stat().st_size
                if size == 0:
                    if self.headers.get("Range"):
                        self.send_error(416)
                        return None
                    start, end, code = 0, -1, 200
                else:
                    start, end, code = 0, size - 1, 200
                    header = self.headers.get("Range", "")
                    if header:
                        if not header.startswith("bytes=") or "," in header:
                            self.send_error(416)
                            return None
                        begin, sep, finish = header[6:].partition("-")
                        if not sep or not begin.isdigit() or (finish and not finish.isdigit()):
                            self.send_error(416)
                            return None
                        start = int(begin)
                        end = int(finish) if finish else size - 1
                        if start >= size or end < start:
                            self.send_error(416)
                            return None
                        end = min(end, size - 1)
                        code = 206
                self.send_response(code)
                self.send_header("Content-Type", mimetypes.guess_type(file_path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                if code == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                return file_path, start, end

            def do_HEAD(self) -> None:  # noqa: N802
                self._headers()

            def do_GET(self) -> None:  # noqa: N802
                selected = self._headers()
                if selected is None:
                    return
                file_path, start, end = selected
                with file_path.open("rb") as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = source.read(min(_CHUNK, remaining))
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        remaining -= len(chunk)

        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True, name="invokeai-directory-transfer")
        self._thread.start()
        host = self.advertise_host or _route_local_ip(self.remote_url)
        if "://" in host:
            host = urllib.parse.urlsplit(host).hostname or host
        host = host.strip("[]")
        self.url = f"http://{host}:{server.server_port}/{self._token}"
        return self

    def manifest(self, *, name: str, model_hash: str) -> dict[str, Any]:
        return {
            "base_url": self.url,
            "name": name,
            "model_hash": model_hash,
            "files": [{"path": file.path, "size": file.size, "sha256": file.sha256} for file in self.files],
        }

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None


def _validated_manifest(manifest: dict[str, Any]) -> tuple[str, str, str, list[DirectoryFile], int]:
    base_url = manifest.get("base_url")
    name = manifest.get("name")
    model_hash = manifest.get("model_hash")
    raw_files = manifest.get("files")
    if not isinstance(base_url, str) or len(base_url) > 2048:
        raise ValueError("Invalid model transfer URL")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Invalid model transfer host")
    if parsed.query or parsed.fragment or not parsed.path.strip("/"):
        raise ValueError("Invalid model transfer URL path")
    if not isinstance(name, str) or not name.strip() or len(name) > 256:
        raise ValueError("Invalid model name")
    if not isinstance(model_hash, str) or not model_hash or len(model_hash) > 256:
        raise ValueError("Invalid model hash")
    if not isinstance(raw_files, list) or not 0 < len(raw_files) <= _MAX_FILES:
        raise ValueError("Invalid model file count")
    files = []
    names: set[str] = set()
    total = 0
    for entry in raw_files:
        if not isinstance(entry, dict):
            raise ValueError("Invalid model file")
        path = _safe_path(entry.get("path")) if isinstance(entry.get("path"), str) else None
        size, digest = entry.get("size"), entry.get("sha256")
        if path is None or type(size) is not int or size < 0 or size > _MAX_BYTES:
            raise ValueError("Invalid model file size or path")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid model file hash")
        if path.as_posix().casefold() in names:
            raise ValueError("Duplicate model path")
        names.add(path.as_posix().casefold())
        total += size
        if total > _MAX_BYTES:
            raise ValueError("Model exceeds transfer size limit")
        files.append(DirectoryFile(path.as_posix(), size, digest))
    return base_url.rstrip("/"), name.strip(), model_hash, files, total


def _set_job(job_id: int, **updates: Any) -> None:
    with _JOB_LOCK:
        _JOBS[job_id].update(updates)


def get_directory_install_job(job_id: int) -> dict[str, Any] | None:
    with _JOB_LOCK:
        current = _JOBS.get(job_id)
        return dict(current) if current is not None else None


class DirectoryTransferCancelled(Exception):
    """The owner no longer needs this temporary model download."""


def _check_cancel(job_id: int) -> None:
    if _CANCEL_EVENTS[job_id].is_set():
        raise DirectoryTransferCancelled()


def cancel_directory_install_job(job_id: int) -> dict[str, Any] | None:
    """Request cooperative cancellation; the worker owns scratch cleanup."""
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        if job["status"] in {"completed", "cancelled", "error"}:
            return dict(job)
        cancel_event = _CANCEL_EVENTS.get(job_id)
        if cancel_event is None:
            # Cancellation cleanup may have released the event just before the
            # terminal status becomes visible. Never turn that race into a 500.
            return dict(job)
        cancel_event.set()
        return dict(job)


def _download_file(base_url: str, file: DirectoryFile, dest: Path, job_id: int, completed_bytes: int) -> None:
    url = f"{base_url}/{urllib.parse.quote(file.path, safe='/')}"
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        _check_cancel(job_id)
        offset = part.stat().st_size if part.exists() else 0
        if offset > file.size:
            part.unlink()
            offset = 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    part.unlink(missing_ok=True)
                    offset = 0
                if offset and response.headers.get("Content-Range", "").split("-", 1)[0] != f"bytes {offset}":
                    raise ValueError(f"Bad resumed response for {file.path}")
                with part.open("ab" if offset else "wb") as target:
                    while chunk := response.read(_CHUNK):
                        _check_cancel(job_id)
                        target.write(chunk)
                        _set_job(job_id, bytes=completed_bytes + target.tell())
                        if target.tell() > file.size:
                            raise ValueError(f"Oversized model file: {file.path}")
            _check_cancel(job_id)
            if part.stat().st_size != file.size or _sha256_file(part) != file.sha256:
                part.unlink(missing_ok=True)
                raise ValueError(f"Hash/size verification failed: {file.path}")
            part.replace(dest)
            return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Download failed for {file.path}: {exc}") from exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Download failed for {file.path}")


def _run_install(job_id: int, manifest: tuple[str, str, str, list[DirectoryFile], int], services: Any) -> None:
    base_url, name, model_hash, files, _total = manifest
    was_cancelled = False
    try:
        _check_cancel(job_id)
        _set_job(job_id, status="downloading")
        models_path = Path(services.configuration.models_path)
        with tempfile.TemporaryDirectory(prefix="tmpinstall_irw_", dir=models_path) as scratch:
            model_path = Path(scratch) / "model"
            model_path.mkdir()
            completed_bytes = 0
            for file in files:
                _check_cancel(job_id)
                _download_file(
                    base_url, file, model_path.joinpath(*PurePosixPath(file.path).parts), job_id, completed_bytes
                )
                completed_bytes += file.size
                _set_job(job_id, bytes=completed_bytes)
            _check_cancel(job_id)
            _set_job(job_id, status="installing")
            from invokeai.app.services.model_records.model_records_base import ModelRecordChanges

            installer = services.model_manager.install
            install_job = installer.heuristic_import(
                str(model_path), config=ModelRecordChanges(name=name), inplace=False
            )
            # Do not interrupt the native installer mid-move: it must finish before
            # the temporary directory can be safely removed.
            result = installer.wait_for_job(install_job)
            if str(getattr(result.status, "value", result.status)) != "completed":
                raise RuntimeError(result.error or f"Model installer ended with status {result.status}")
            config = result.config_out
            installed_hash = str(getattr(config, "hash", "") or "")
            if installed_hash != model_hash:
                raise RuntimeError(
                    f"Installed model hash differs from primary: expected {model_hash}, got {installed_hash or 'none'}"
                )
            _set_job(job_id, status="completed", model_key=str(getattr(config, "key", "")))
    except DirectoryTransferCancelled:
        was_cancelled = True
    except Exception as exc:
        services.logger.error(f"Remote directory transfer job {job_id}: {exc}")
        _set_job(job_id, status="error", error=str(exc))
    finally:
        with _JOB_LOCK:
            if was_cancelled:
                # Publish the terminal state before releasing the cancel event/hash
                # lease so another request can never observe a live status with no
                # cancellation handle.
                _JOBS[job_id].update(status="cancelled", error=None)
            if _ACTIVE_HASHES.get(model_hash) == job_id:
                _ACTIVE_HASHES.pop(model_hash, None)
            _CANCEL_EVENTS.pop(job_id, None)


def start_directory_install(manifest: dict[str, Any], services: Any) -> dict[str, Any]:
    """Schedule the complete transfer; never block an API request on a large download."""
    global _NEXT_JOB_ID
    validated = _validated_manifest(manifest)
    model_hash = validated[2]
    with _JOB_LOCK:
        active_id = _ACTIVE_HASHES.get(model_hash)
        if active_id is not None:
            if _CANCEL_EVENTS[active_id].is_set():
                raise ValueError("Previous transfer cancellation is still being cleaned up; retry shortly")
            return dict(_JOBS[active_id])
        if len(_ACTIVE_HASHES) >= 8:
            raise ValueError("Too many simultaneous directory transfers")
        _NEXT_JOB_ID += 1
        job_id = _NEXT_JOB_ID
        _JOBS[job_id] = {"id": job_id, "status": "waiting", "bytes": 0, "total_bytes": validated[4], "error": None}
        _CANCEL_EVENTS[job_id] = threading.Event()
        _ACTIVE_HASHES[model_hash] = job_id
        if len(_JOBS) > 256:
            for old_id in list(_JOBS):
                if old_id not in _ACTIVE_HASHES.values() and old_id != job_id:
                    _JOBS.pop(old_id)
                    if len(_JOBS) <= 128:
                        break
    threading.Thread(
        target=_run_install, args=(job_id, validated, services), daemon=True, name=f"irw-directory-{job_id}"
    ).start()
    return get_directory_install_job(job_id) or {"id": job_id, "status": "waiting"}
