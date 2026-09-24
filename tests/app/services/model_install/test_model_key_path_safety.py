"""An install request may name the key it wants, and that key is joined onto the filesystem.

`ModelRecordChanges.key` is the entry point (`POST /v2/models/install` takes it in the body); the joins it feeds
are the model's own directory under `models_path` and its cover image under `model_images`.
"""

from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from invokeai.app.services.model_records.model_records_base import ModelRecordChanges
from invokeai.backend.model_manager.util.lora_metadata_extractor import _process_preview_image

TRAVERSAL_KEYS = [
    "../../pwned",
    "..",
    "sub/nested",
    "/etc/pwned",
    "..\\pwned",
    "C:\\pwned",
    "..:stream",
    "nul\x00byte",
]


@pytest.mark.parametrize("key", TRAVERSAL_KEYS)
def test_model_record_changes_rejects_traversal_keys(key: str):
    with pytest.raises(ValidationError):
        ModelRecordChanges(key=key)


@pytest.mark.parametrize("key", ["ecd3b3a5-6c4f-4a5f-9a0e-4b1c2d3e4f50", "openai-dall-e-3", "external_test"])
def test_model_record_changes_accepts_generated_keys(key: str):
    """uuids and `slugify()`ed external-model keys are what the server actually generates."""
    assert ModelRecordChanges(key=key).key == key


def test_model_record_changes_without_a_key_is_unaffected():
    assert ModelRecordChanges(name="some model").key is None


def test_lora_preview_image_is_not_written_outside_the_images_folder(tmp_path: Path):
    """`_process_preview_image` builds its path by hand rather than going through `ModelImageFileStorageDisk`, so
    it needs its own guard: with none, a key of `../../pwned` lands the thumbnail two levels up."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_images = tmp_path / "root" / "model_images"
    model_images.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(model_dir / "mylora.png")

    assert _process_preview_image("mylora", model_dir, "../../pwned", model_images) is False
    assert not (tmp_path / "pwned.webp").exists()
    assert list(model_images.iterdir()) == []


def test_lora_preview_image_is_still_written_for_an_ordinary_key(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_images = tmp_path / "model_images"
    model_images.mkdir()
    Image.new("RGB", (32, 32)).save(model_dir / "mylora.png")

    assert _process_preview_image("mylora", model_dir, "abc-123", model_images) is True
    assert (model_images / "abc-123.webp").exists()


def test_install_path_refuses_an_unvalidated_traversal_key(mm2_installer, embedding_file: Path):
    """`install_path()` moves the model into `models_path / key`. A caller can build `ModelRecordChanges` without
    validation, so the join must be guarded at the sink too - before anything is created or moved."""
    models_path = mm2_installer.app_config.models_path
    escaped = models_path.parent / "escaped"
    config = ModelRecordChanges.model_construct(key="../escaped")

    with pytest.raises(ValueError, match="plain filename"):
        mm2_installer.install_path(embedding_file, config=config)

    assert not escaped.exists()
    assert embedding_file.exists()
    assert mm2_installer.record_store.search_by_attr() == []
