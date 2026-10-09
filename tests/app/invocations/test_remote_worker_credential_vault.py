"""Cross-platform remote-worker AES-256-GCM vault regression tests."""

import base64
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from invokeai.app.invocations.remote_worker import credential_vault as vault


@pytest.fixture
def vault_path(tmp_path, monkeypatch):
    path = tmp_path / "invokeai" / "remote_workers" / "credentials.enc"
    monkeypatch.setattr(vault, "_vault_path", lambda: path)
    return path


def test_first_use_does_not_create_key_or_migrate_dpapi(vault_path):
    vault_path.parent.mkdir(parents=True)
    legacy = vault_path.with_name("credentials.dpapi")
    legacy.write_bytes(b"old Windows-only encrypted credentials")

    assert vault.get_saved_credentials("alice", "http://worker:9090") is None
    assert not vault_path.exists()
    assert not vault_path.with_name("credentials.key").exists()

    vault.save_credentials("alice", "http://worker:9090", "a@example.com", "new-password")
    assert legacy.read_bytes() == b"old Windows-only encrypted credentials"
    assert vault.get_saved_credentials("alice", "http://worker:9090")["password"] == "new-password"


def test_cross_platform_roundtrip_and_per_user_isolation(vault_path):
    vault.save_credentials("alice", "http://worker:9090/", "a@example.com", "password-A", False)
    vault.save_credentials("bob", "http://worker:9090", "b@example.com", "password-B")
    vault.save_credentials("alice", "https://second", "a@example.com", "password-C")

    assert vault.get_saved_credentials("alice", "http://worker:9090") == {
        "email": "a@example.com",
        "password": "password-A",
        "remember_me": False,
    }
    assert vault.get_saved_credentials("bob", "http://worker:9090")["password"] == "password-B"
    assert vault.get_saved_credentials("alice", "https://second")["password"] == "password-C"
    assert vault.get_saved_credentials("bob", "https://second") is None

    vault.delete_credentials("alice", "http://worker:9090")
    assert vault.get_saved_credentials("alice", "http://worker:9090") is None
    assert vault.get_saved_credentials("bob", "http://worker:9090")["password"] == "password-B"


def test_key_and_ciphertext_are_separate_and_nonce_rotates(vault_path):
    vault.save_credentials("alice", "https://worker", "a@example.com", "secret-plaintext")
    key_path = vault_path.with_name("credentials.key")
    first_key = key_path.read_bytes()
    first = json.loads(vault_path.read_text())
    assert len(first_key) == 32
    assert b"secret-plaintext" not in vault_path.read_bytes()
    assert len(base64.b64decode(first["nonce"], validate=True)) == 12
    assert len(base64.b64decode(first["ciphertext"], validate=True)) > 16

    vault.save_credentials("alice", "https://worker", "a@example.com", "secret-plaintext")
    second = json.loads(vault_path.read_text())
    assert key_path.read_bytes() == first_key
    assert second["nonce"] != first["nonce"]
    assert second["ciphertext"] != first["ciphertext"]

    if os.name == "posix":
        assert stat.S_IMODE(vault_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(vault_path.stat().st_mode) == 0o600


def test_tampering_is_detected_and_not_overwritten(vault_path):
    vault.save_credentials("alice", "http://worker", "a@example.com", "secret")
    payload = json.loads(vault_path.read_text())
    ciphertext = bytearray(base64.b64decode(payload["ciphertext"]))
    ciphertext[-1] ^= 1
    payload["ciphertext"] = base64.b64encode(ciphertext).decode("ascii")
    vault_path.write_text(json.dumps(payload))
    corrupted = vault_path.read_bytes()

    with pytest.raises(ValueError, match="could not be authenticated"):
        vault.get_saved_credentials("alice", "http://worker")
    with pytest.raises(ValueError, match="could not be authenticated"):
        vault.save_credentials("alice", "http://worker", "a@example.com", "replacement")
    assert vault_path.read_bytes() == corrupted


def test_missing_or_incorrect_key_never_regenerates_it(vault_path):
    vault.save_credentials("alice", "http://worker", "a@example.com", "secret")
    key_path = vault_path.with_name("credentials.key")
    original_vault = vault_path.read_bytes()
    original_key = key_path.read_bytes()
    key_path.unlink()

    with pytest.raises(RuntimeError, match="encryption key is missing"):
        vault.get_saved_credentials("alice", "http://worker")
    with pytest.raises(RuntimeError, match="encryption key is missing"):
        vault.save_credentials("alice", "http://worker", "a@example.com", "new")
    assert not key_path.exists()
    assert vault_path.read_bytes() == original_vault

    key_path.write_bytes(os.urandom(32))
    with pytest.raises(ValueError, match="could not be authenticated"):
        vault.get_saved_credentials("alice", "http://worker")
    assert vault_path.read_bytes() == original_vault
    key_path.write_bytes(original_key)
    assert vault.get_saved_credentials("alice", "http://worker")["password"] == "secret"


def test_concurrent_writes_preserve_all_users(vault_path):
    def save_user(index):
        vault.save_credentials(f"user-{index}", "http://worker", f"{index}@example.com", f"pass-{index}")

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(save_user, range(12)))

    for index in range(12):
        assert vault.get_saved_credentials(f"user-{index}", "http://worker")["password"] == f"pass-{index}"


def test_url_and_required_field_validation(vault_path):
    with pytest.raises(ValueError, match="without login details"):
        vault.save_credentials("alice", "https://user:password@worker", "a@example.com", "secret")
    with pytest.raises(ValueError, match="required"):
        vault.save_credentials("alice", "http://worker", "a@example.com", "")
    assert not vault_path.exists()


def test_settings_share_encrypted_vault_without_disturbing_credentials(vault_path):
    vault.save_credentials("alice", "http://worker:9090", "a@example.com", "secret")
    settings = {
        "enabled": True,
        "dispatchMode": "remote_only",
        "workerUrls": "http://worker:9090",
        "workerNames": {"http://worker:9090": "RTX5080"},
        "disabledWorkerUrls": [],
        "autoTransferMissingModels": True,
        "keepRemoteCopies": False,
        "modelTransferHost": "",
    }

    vault.save_settings("alice", settings)

    assert vault.get_saved_settings("alice") == settings
    assert vault.get_saved_credentials("alice", "http://worker:9090")["password"] == "secret"
    assert b"RTX5080" not in vault_path.read_bytes()
    assert b"http://worker:9090" not in vault_path.read_bytes()

    vault.delete_credentials("alice", "http://worker:9090")
    assert vault.get_saved_credentials("alice", "http://worker:9090") is None
    assert vault.get_saved_settings("alice") == settings
