"""An install request may name the key it wants, and that key is joined onto the filesystem.

`POST /v2/models/install` takes it in the body; the joins it feeds are the model's own directory under
`models_path` and its cover image under `model_images`. The check lives on the install route rather than on
`ModelRecordChanges` itself, because that model is also the *update* body and is re-parsed from install markers
written by older versions - see the comment on `ModelRecordChanges.key`.
"""

from pathlib import Path

import pytest
from PIL import Image

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
def test_install_route_rejects_traversal_keys(key: str):
    """`install_model` is where a client-chosen key is adopted, so that is where it is checked."""
    from unittest.mock import MagicMock, patch

    from starlette.exceptions import HTTPException

    from invokeai.app.api.routers import model_manager

    with patch.object(model_manager, "ApiDependencies") as deps:
        deps.invoker.services.logger = MagicMock()

        with pytest.raises(HTTPException) as exc_info:
            model_manager.install_model(MagicMock(), source="/models/x.safetensors", config=ModelRecordChanges(key=key))

        assert exc_info.value.status_code == 422
        deps.invoker.services.model_manager.install.heuristic_import.assert_not_called()


@pytest.mark.parametrize("key", ["ecd3b3a5-6c4f-4a5f-9a0e-4b1c2d3e4f50", "openai-dall-e-3", "external_test"])
def test_install_route_accepts_generated_keys(key: str):
    """uuids and `slugify()`ed external-model keys are what the server generates, and the route must pass them
    through to the installer rather than merely not raising."""
    from unittest.mock import MagicMock, patch

    from invokeai.app.api.routers import model_manager

    with patch.object(model_manager, "ApiDependencies") as deps:
        deps.invoker.services.logger = MagicMock()
        installer = deps.invoker.services.model_manager.install

        model_manager.install_model(MagicMock(), source="/models/x.safetensors", config=ModelRecordChanges(key=key))

        assert installer.heuristic_import.call_count == 1
        assert installer.heuristic_import.call_args.kwargs["config"].key == key


@pytest.mark.parametrize("key", TRAVERSAL_KEYS + ["legacy:key", "legacy?key", "legacy*key"])
def test_model_record_changes_still_parses_a_legacy_key(key: str):
    """`ModelRecordChanges` must NOT validate the key. It is the update body too, and the edit form posts the whole
    record back unchanged (there is no key field in the form - `ModelEdit.tsx` seeds its defaults from the config,
    so the key rides along as an echo), so validating here would answer 422 on every attempt to edit a model whose
    key predates key validation. It is also re-parsed from install markers written by older versions, where a
    rejection strands the partial download. Changing a key is refused by the route, not by this model."""
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


def test_install_refuses_an_unvalidated_traversal_key(mm2_installer, embedding_file: Path):
    """A caller can build `ModelRecordChanges` without going through the route, so `_probe()` - the one place a
    caller-supplied key becomes a record's key, for in-place registration as well as for a move-in install -
    refuses it before anything is created or moved."""
    from invokeai.app.services.model_install.model_install_common import InvalidModelConfigException

    models_path = mm2_installer.app_config.models_path
    escaped = models_path.parent / "escaped"
    config = ModelRecordChanges.model_construct(key="../escaped")

    with pytest.raises(InvalidModelConfigException, match="plain filename"):
        mm2_installer.install_path(embedding_file, config=config)

    assert not escaped.exists()
    assert embedding_file.exists()
    assert mm2_installer.record_store.search_by_attr() == []


def test_install_path_guards_the_join_even_if_the_key_got_past_probe(mm2_installer, embedding_file: Path):
    """`install_path()` moves the model into `models_path / key`, so the join is guarded at the sink as well -
    independently of `_probe()`. Bypass the probe check to prove the sink one is load-bearing on its own."""
    from unittest.mock import patch

    models_path = mm2_installer.app_config.models_path
    escaped = models_path.parent / "escaped"
    probed = mm2_installer._probe(embedding_file, ModelRecordChanges())
    probed.key = "../escaped"

    with patch.object(mm2_installer, "_probe", return_value=probed):
        with pytest.raises(ValueError, match="plain filename"):
            mm2_installer.install_path(embedding_file, config=ModelRecordChanges())

    assert not escaped.exists()
    assert embedding_file.exists()
