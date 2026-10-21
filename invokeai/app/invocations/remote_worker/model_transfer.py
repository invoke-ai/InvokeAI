from __future__ import annotations

import hashlib
import mimetypes
import secrets
import socket
import threading
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class ModelTransferError(RuntimeError):
    """Raised when a local model cannot be exposed safely to a remote worker."""


@dataclass(frozen=True)
class LocalModelFile:
    path: Path
    key: str
    name: str
    hash: str
    base: str
    type: str


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return "" if raw is None else str(raw)


def model_layout_signature(path: Path) -> tuple[str, str]:
    """Return a cheap, cross-platform signature of a model's on-disk layout.

    InvokeAI's model hash intentionally hashes model weight contents but not their
    relative paths or tokenizer/config files. For directories, this signature hashes
    the sorted relative path of every regular file so two installs with identical
    weights but incompatible directory layouts do not compare as equivalent.
    """
    if path.is_file():
        return "file", ""
    if not path.is_dir():
        raise ModelTransferError(f"Model path is not a file or directory: {path}")

    digest = hashlib.sha256()
    for child in sorted((entry for entry in path.rglob("*") if entry.is_file()), key=lambda entry: entry.as_posix()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
    return "directory", digest.hexdigest()


def resolve_local_model_file(services: Any, identifier: dict[str, Any]) -> LocalModelFile:
    """Resolve a graph model identifier to a file or directory on this InvokeAI install."""
    key = str(identifier.get("key") or "").strip()
    if not key:
        raise ModelTransferError("Model identifier has no local model key")

    try:
        config = services.model_manager.store.get_model(key)
    except Exception as exc:
        raise ModelTransferError(f"Could not read local model record {key}: {exc}") from exc

    config_path = getattr(config, "path", None)
    if not config_path:
        raise ModelTransferError(
            f"Local model '{getattr(config, 'name', identifier.get('name', key))}' has no filesystem path"
        )

    model_path = Path(str(config_path))
    if not model_path.is_absolute():
        model_path = Path(services.configuration.models_path) / model_path
    model_path = model_path.resolve()

    if not model_path.exists():
        raise ModelTransferError(f"Local model file does not exist: {model_path}")
    if not model_path.is_file() and not model_path.is_dir():
        raise ModelTransferError(f"Local model path is not a file or directory: {model_path}")

    model_hash = str(identifier.get("hash") or getattr(config, "hash", "") or "").strip()
    if not model_hash:
        raise ModelTransferError(
            f"Local model '{getattr(config, 'name', model_path.name)}' has no recorded model hash; "
            "cannot verify a remote transfer safely"
        )

    return LocalModelFile(
        path=model_path,
        key=key,
        name=str(getattr(config, "name", None) or identifier.get("name") or model_path.stem),
        hash=model_hash,
        base=_enum_value(getattr(config, "base", None) or identifier.get("base")),
        type=_enum_value(getattr(config, "type", None) or identifier.get("type")),
    )


def enrich_model_identifier_hashes(graph: dict[str, Any], services: Any) -> int:
    """Fill missing graph model hashes from this InvokeAI install's model records."""
    changed = 0

    def visit(value: Any) -> None:
        nonlocal changed
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return

        if all(field in value for field in ("key", "name", "base", "type")):
            key = str(value.get("key") or "")
            if key and not value.get("hash"):
                try:
                    config = services.model_manager.store.get_model(key)
                    model_hash = str(getattr(config, "hash", "") or "").strip()
                except Exception:
                    model_hash = ""
                if model_hash:
                    value["hash"] = model_hash
                    changed += 1

        for item in value.values():
            visit(item)

    visit(graph.get("nodes"))
    return changed


def _route_local_ip(remote_url: str) -> str:
    parsed = urllib.parse.urlsplit(remote_url)
    host = parsed.hostname
    if not host:
        raise ModelTransferError(f"Cannot determine remote host from URL: {remote_url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        addresses = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
    except OSError as exc:
        raise ModelTransferError(f"Could not resolve remote host '{host}': {exc}") from exc
    if not addresses:
        raise ModelTransferError(f"Could not resolve an IPv4 route to remote host '{host}'")

    target = addresses[0][4]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(target)
        local_ip = str(sock.getsockname()[0])
    except OSError as exc:
        raise ModelTransferError(f"Could not determine local LAN address for {remote_url}: {exc}") from exc
    finally:
        sock.close()

    if not local_ip or local_ip.startswith("127.") or local_ip == "0.0.0.0":
        raise ModelTransferError(
            f"Automatic LAN address detection returned '{local_ip}'. Set Model Transfer Host manually on the node."
        )
    return local_ip


class TemporaryModelServer:
    """Serve exactly one model file on a random-token URL for the life of this context."""

    def __init__(self, model: LocalModelFile, remote_url: str, advertise_host: str = "") -> None:
        self.model = model
        self.remote_url = remote_url
        self.advertise_host = advertise_host.strip()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        token = secrets.token_urlsafe(32)
        self._request_path = f"/{token}/{urllib.parse.quote(model.path.name, safe='')}"
        self.url = ""

    def __enter__(self) -> "TemporaryModelServer":
        file_path = self.model.path
        request_path = self._request_path
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

        class Handler(BaseHTTPRequestHandler):
            server_version = "InvokeAIRemoteWorkerModelTransfer/0.10"

            def log_message(self, _format: str, *args: Any) -> None:
                return

            def _match(self) -> bool:
                return urllib.parse.urlsplit(self.path).path == request_path

            def _range(self, size: int) -> tuple[int, int] | None:
                header = self.headers.get("Range", "").strip()
                if not header:
                    return None
                if not header.startswith("bytes=") or "," in header:
                    return None
                value = header[6:]
                start_text, _, end_text = value.partition("-")
                try:
                    if start_text:
                        start = int(start_text)
                        end = int(end_text) if end_text else size - 1
                    else:
                        suffix = int(end_text)
                        start = max(0, size - suffix)
                        end = size - 1
                except ValueError:
                    return None
                if start < 0 or start >= size or end < start:
                    return None
                return start, min(end, size - 1)

            def _send_headers(self, *, body: bool) -> tuple[int, int] | None:
                if not self._match():
                    self.send_error(404)
                    return None
                size = file_path.stat().st_size
                byte_range = self._range(size)
                if self.headers.get("Range") and byte_range is None:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return None
                if byte_range is None:
                    start, end = 0, size - 1
                    self.send_response(200)
                else:
                    start, end = byte_range
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(max(0, end - start + 1)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Disposition", f'attachment; filename="{file_path.name}"')
                self.end_headers()
                return start, end

            def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                self._send_headers(body=False)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                selected = self._send_headers(body=True)
                if selected is None:
                    return
                start, end = selected
                remaining = end - start + 1
                with file_path.open("rb") as stream:
                    stream.seek(start)
                    while remaining > 0:
                        chunk = stream.read(min(1024 * 1024, remaining))
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
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="invokeai-remote-worker-model-transfer",
            daemon=True,
        )
        self._thread.start()

        host = self.advertise_host or _route_local_ip(self.remote_url)
        if "://" in host:
            host = urllib.parse.urlsplit(host).hostname or host
        host = host.strip("[]")
        self.url = f"http://{host}:{server.server_port}{self._request_path}"
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
