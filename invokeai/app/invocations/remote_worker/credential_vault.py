"""Cross-platform, per-user remote-worker credential vault.

AES-256-GCM protects credentials at rest. The randomly generated key is stored
separately from the encrypted vault under InvokeAI's configured runtime root.
Protect the entire runtime directory and preserve both files when moving an
installation: anyone with both files can decrypt the credentials.

Older Windows ``credentials.dpapi`` files are not read or migrated. Re-enter
saved worker logins after upgrading from the DPAPI-only implementation.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_LOCK = threading.RLock()
_AAD = b"invokeai-remote-workers-credentials-v1"
_NONCE_BYTES = 12
_KEY_BYTES = 32
_SETTINGS_KEY = "__settings__"


def normalize_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.netloc
    ):
        raise ValueError(
            "Enter a worker URL beginning with http:// or https:// (without login details or query strings)"
        )
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid port in worker URL") from exc
    return url


def _vault_path() -> Path:
    # Import lazily: remote_client is discovered during core invocation import.
    from invokeai.app.services.config.config_default import get_config

    return get_config().root_path / "remote_workers" / "credentials.enc"


def _key_path() -> Path:
    return _vault_path().with_name("credentials.key")


def _ensure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        path.chmod(0o700)


def _read_key(*, create: bool) -> bytes:
    key_path = _key_path()
    if create:
        _ensure_directory(key_path.parent)
        # Exclusive creation: never rotate a key just because a file is damaged.
        # In particular, an existing ciphertext must never be paired with a new key.
        try:
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            try:
                with os.fdopen(fd, "wb") as key_file:
                    key_file.write(AESGCM.generate_key(bit_length=256))
                    key_file.flush()
                    os.fsync(key_file.fileno())
            except BaseException:
                key_path.unlink(missing_ok=True)
                raise
    try:
        key = key_path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Remote Workers encryption key is missing; restore credentials.key with credentials.enc"
        ) from exc
    if len(key) != _KEY_BYTES:
        raise ValueError("Remote Workers encryption key is invalid; restore the original credentials.key")
    return key


def _read() -> dict[str, dict[str, dict[str, object]]]:
    path = _vault_path()
    if not path.exists():
        return {}
    # A missing/corrupt key is an error, not an empty vault or an excuse to
    # replace the vault. This also prevents accidental credential loss.
    key = _read_key(create=False)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or envelope.get("version") != 1:
        raise ValueError("Remote Workers credentials file has an unsupported format")
    try:
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
        if len(nonce) != _NONCE_BYTES:
            raise ValueError("Invalid AES-GCM nonce")
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, _AAD)
    except (InvalidTag, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Remote Workers credentials could not be authenticated; check the key and vault files"
        ) from exc
    data = json.loads(plaintext)
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("users"), dict):
        raise ValueError("Remote Workers credentials file has an unsupported format")
    return data["users"]


def _write(users: dict[str, dict[str, dict[str, object]]]) -> None:
    path = _vault_path()
    _ensure_directory(path.parent)
    key = _read_key(create=True)
    plaintext = json.dumps({"version": 1, "users": users}, separators=(",", ":")).encode("utf-8")
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _AAD)
    envelope = json.dumps(
        {
            "version": 1,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(envelope)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def get_saved_credentials(user_id: str, url: str) -> dict[str, object] | None:
    with _LOCK:
        entry = _read().get(user_id, {}).get(normalize_url(url))
        return dict(entry) if isinstance(entry, dict) else None


def get_saved_settings(user_id: str) -> dict[str, object] | None:
    """Return this InvokeAI user's encrypted remote-worker settings, if saved."""
    with _LOCK:
        entry = _read().get(user_id, {}).get(_SETTINGS_KEY)
        return dict(entry) if isinstance(entry, dict) else None


def save_settings(user_id: str, settings: dict[str, object]) -> None:
    """Persist this InvokeAI user's remote-worker settings in the encrypted vault."""
    if not user_id:
        raise ValueError("A user is required")
    with _LOCK:
        users = _read()
        users.setdefault(user_id, {})[_SETTINGS_KEY] = dict(settings)
        _write(users)


def save_credentials(user_id: str, url: str, email: str, password: str, remember_me: bool = True) -> None:
    if not user_id or not email.strip() or not password:
        raise ValueError("A user, email, and password are required")
    normalized = normalize_url(url)
    with _LOCK:
        users = _read()
        users.setdefault(user_id, {})[normalized] = {
            "email": email.strip(),
            "password": password,
            "remember_me": remember_me,
        }
        _write(users)


def delete_credentials(user_id: str, url: str) -> None:
    normalized = normalize_url(url)
    with _LOCK:
        users = _read()
        if normalized in users.get(user_id, {}):
            del users[user_id][normalized]
            if not users[user_id]:
                del users[user_id]
            _write(users)
