import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from invokeai.app.services.model_images.model_images_common import (
    ModelImageFileDeleteException,
    ModelImageFileNotFoundException,
    ModelImageFileSaveException,
)
from invokeai.app.services.model_images.model_images_default import ModelImageFileStorageDisk
from invokeai.app.util import path_safety

# Model keys reach this service from the `/v2/models/i/{key}/image` path parameter and from an install
# request's `ModelRecordChanges.key`. Note which of these are actually reachable over HTTP: uvicorn
# percent-decodes before Starlette routes, so `%2F` never matches the `{key}` segment and the forward-slash
# shapes 404 at routing. The backslash and drive-letter shapes DO reach the handler, which is why the guard
# must reject them on posix rather than deferring to the platform's `Path`.
TRAVERSAL_KEYS = [
    "..",
    ".",
    "",
    "../escaped",
    "sub/nested",
    "/etc/passwd",
    "..\\escaped",
    "C:\\Windows\\win",
    "..:stream",
    "nul\x00byte",
]


# The subset of TRAVERSAL_KEYS that leaves the folder on posix. The rest are ordinary filenames there, and an older
# version may have stored a cover under one, so the store only refuses them on Windows.
POSIX_TRAVERSAL_KEYS = ["..", ".", "", "../escaped", "sub/nested", "/etc/passwd", "nul\x00byte"]


@pytest.fixture
def windows_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply Windows' containment rules whatever the host, so a posix CI run still covers the Windows payloads."""
    monkeypatch.setattr(path_safety, "_ON_WINDOWS", True)


@pytest.fixture
def posix_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_safety, "_ON_WINDOWS", False)


@pytest.fixture
def storage(tmp_path: Path) -> ModelImageFileStorageDisk:
    storage = ModelImageFileStorageDisk(tmp_path / "model_images")
    storage.start(MagicMock())
    return storage


@pytest.mark.parametrize("key", TRAVERSAL_KEYS)
def test_get_path_rejects_traversal_keys(storage: ModelImageFileStorageDisk, windows_rules: None, key: str):
    with pytest.raises(ModelImageFileNotFoundException):
        storage.get_path(key)


@pytest.mark.parametrize("key", POSIX_TRAVERSAL_KEYS)
def test_get_path_rejects_posix_traversal_keys(storage: ModelImageFileStorageDisk, posix_rules: None, key: str):
    with pytest.raises(ModelImageFileNotFoundException):
        storage.get_path(key)


@pytest.mark.parametrize("key", ["legacy:key", "legacy?key", "a\\b", "NUL", "trailing.", "..:stream"])
def test_get_path_keeps_legacy_posix_keys_inside_the_folder(
    storage: ModelImageFileStorageDisk, posix_rules: None, key: str
):
    """On posix these are ordinary filenames that stay in the folder, and older versions stored covers under them."""
    assert storage.get_path(key).parent == storage._model_images_folder


def test_get_does_not_read_outside_the_images_folder(storage: ModelImageFileStorageDisk, tmp_path: Path):
    """A traversal key that names a real file outside the folder must not be opened - without a guard, `get()`
    finds it, `_validate_path` is happy, and it is served."""
    Image.new("RGB", (8, 8)).save(tmp_path / "outside.webp", format="webp")

    with pytest.raises(ModelImageFileNotFoundException):
        storage.get("../outside")


@pytest.mark.parametrize("key", TRAVERSAL_KEYS)
def test_save_rejects_traversal_keys(storage: ModelImageFileStorageDisk, windows_rules: None, key: str):
    with pytest.raises(ModelImageFileSaveException):
        storage.save(Image.new("RGB", (8, 8)), key)


def test_save_does_not_write_outside_the_images_folder(storage: ModelImageFileStorageDisk, tmp_path: Path):
    """`save()` used to build its own path instead of going through `get_path()`, so it needs its own coverage."""
    with pytest.raises(ModelImageFileSaveException):
        storage.save(Image.new("RGB", (8, 8)), "../planted")
    assert not (tmp_path / "planted.webp").exists()


def test_delete_does_not_unlink_outside_the_images_folder(storage: ModelImageFileStorageDisk, tmp_path: Path):
    victim = tmp_path / "victim.webp"
    victim.write_bytes(b"not really a webp")

    with pytest.raises(ModelImageFileDeleteException):
        storage.delete("../victim")
    assert victim.exists()


def test_get_url_returns_none_for_a_traversal_key(storage: ModelImageFileStorageDisk, tmp_path: Path):
    """`get_url()` runs for every model in the list response, so a bad key must not fail the whole listing - and
    it must not advertise a file outside the folder either. The file has to exist for this to be a real test:
    `get_url()` returns `None` for a missing file regardless."""
    Image.new("RGB", (8, 8)).save(tmp_path / "outside.webp", format="webp")

    assert storage.get_url("../outside") is None


def test_ordinary_keys_still_round_trip(storage: ModelImageFileStorageDisk):
    """The guard must not reject the key shapes the server actually generates: uuids and external-model slugs."""
    for key in ("ecd3b3a5-6c4f-4a5f-9a0e-4b1c2d3e4f50", "openai-dall-e-3"):
        storage.save(Image.new("RGB", (8, 8)), key)
        assert storage.get_path(key).parent == storage._model_images_folder
        assert storage.get(key).size == (8, 8)
        storage.delete(key)
        with pytest.raises(ModelImageFileNotFoundException):
            storage.get(key)


@pytest.mark.skipif(os.name == "nt", reason="Legacy colon keys could only have been stored on POSIX filesystems")
def test_existing_legacy_posix_key_still_serves_its_cover(storage: ModelImageFileStorageDisk):
    """Keys with Windows-reserved punctuation were valid on POSIX before this guard and may already have cover files."""
    key = "legacy:key"
    image_path = storage._model_images_folder / f"{key}.webp"
    Image.new("RGB", (8, 8)).save(image_path, format="webp")
    storage._invoker.services.urls.get_model_image_url.return_value = f"/api/v2/models/i/{key}/image"

    url = storage.get_url(key)

    assert url is not None
    assert url.startswith(f"/api/v2/models/i/{key}/image?")


@pytest.mark.skipif(os.name == "nt", reason="Legacy colon keys could only have been stored on POSIX filesystems")
def test_existing_legacy_posix_key_cover_can_be_deleted(storage: ModelImageFileStorageDisk):
    """An existing cover for a previously valid POSIX key must remain manageable after upgrade."""
    key = "legacy:key"
    image_path = storage._model_images_folder / f"{key}.webp"
    Image.new("RGB", (8, 8)).save(image_path, format="webp")

    storage.delete(key)

    assert not image_path.exists()
