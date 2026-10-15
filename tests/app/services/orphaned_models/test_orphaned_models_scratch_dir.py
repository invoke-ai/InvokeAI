"""The orphan scan must not report a model that is still being written.

An orphan is defined as model files under the models root with no database record — which is also
an exact description of a conversion in progress. Model conversion builds its diffusers copy on the
models volume (so the finished result can be moved into place rather than copied across a
filesystem boundary), and `DELETE /sync/orphaned` rmtrees whatever the scan reported. Before both
routes ran in the threadpool they could not overlap; now they can, so the scratch directory has to
be invisible to the scan by construction.
"""

import json
from pathlib import Path

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.model_install.model_install_common import INSTALL_RECOVERY_SENTINEL
from invokeai.app.services.model_records import ModelRecordServiceSQL
from invokeai.app.services.orphaned_models import CONVERSION_SCRATCH_DIRNAME, OrphanedModelsService
from invokeai.app.services.shared.database.database import Database
from invokeai.backend.model_manager.configs.textual_inversion import TI_File_SD1_Config
from invokeai.backend.model_manager.taxonomy import ModelSourceType
from invokeai.backend.util.logging import InvokeAILogger


@pytest.fixture
def store(database: Database) -> ModelRecordServiceSQL:
    return ModelRecordServiceSQL(database, InvokeAILogger.get_logger())


@pytest.fixture
def models_path(tmp_path: Path) -> Path:
    path = tmp_path / "models"
    path.mkdir()
    return path


def _service(models_path: Path, store: ModelRecordServiceSQL) -> OrphanedModelsService:
    config = InvokeAIAppConfig()
    config._root = models_path.parent
    assert config.models_path == models_path
    return OrphanedModelsService(config=config, store=store)


def _write_model_file(directory: Path, name: str = "model.safetensors") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"not really a model")


def test_conversion_scratch_directory_is_not_reported_as_orphaned(
    models_path: Path, store: ModelRecordServiceSQL
) -> None:
    # What a conversion looks like on disk while it runs: a TemporaryDirectory under the scratch
    # area, holding the diffusers copy it has written so far.
    _write_model_file(models_path / CONVERSION_SCRATCH_DIRNAME / "tmp8f2b1c" / "sd-v1-5")

    orphans = _service(models_path, store).find_orphaned_models()

    assert orphans == [], (
        "The scan reported a conversion's working directory. `DELETE /sync/orphaned` would delete "
        "it while the conversion is still writing into it."
    )


def test_a_real_orphan_is_still_reported(models_path: Path, store: ModelRecordServiceSQL) -> None:
    """The control: the skip must be targeted, not a scan that has stopped finding anything."""
    _write_model_file(models_path / "some-unregistered-model")

    orphans = _service(models_path, store).find_orphaned_models()

    assert [orphan.path for orphan in orphans] == ["some-unregistered-model"]


def test_conversion_scratch_directory_cannot_be_deleted(models_path: Path, store: ModelRecordServiceSQL) -> None:
    _write_model_file(models_path / CONVERSION_SCRATCH_DIRNAME / "active-conversion" / "sd-v1-5")

    result = _service(models_path, store).delete_orphaned_models([CONVERSION_SCRATCH_DIRNAME])

    assert result[CONVERSION_SCRATCH_DIRNAME].startswith("error:")
    assert (models_path / CONVERSION_SCRATCH_DIRNAME).exists()


def test_a_registered_model_is_no_orphan_even_when_its_config_no_longer_validates(
    models_path: Path, store: ModelRecordServiceSQL, database: Database
) -> None:
    _write_model_file(models_path / "future-model")
    # A record of a model type this version does not know, as after going back to an older version: deleting its
    # files would break the model for the version that wrote it.
    record = {
        "key": "future",
        "hash": "blake3:0",
        "base": "any",
        "type": "from_the_future",
        "format": "checkpoint",
        "name": "future model",
        "source": "somewhere",
        "source_type": "path",
        "file_size": 18,
        "path": "future-model/model.safetensors",
    }
    database.queries.models.insert("future", json.dumps(record))

    assert _service(models_path, store).find_orphaned_models() == []


def test_a_model_registered_by_its_absolute_path_is_no_orphan(models_path: Path, store: ModelRecordServiceSQL) -> None:
    # An in-place install records the file where it is, which may be under the models root.
    _write_model_file(models_path / "in-place")
    store.add_model(
        TI_File_SD1_Config(
            path=str(models_path / "in-place" / "model.safetensors"),
            name="in place",
            hash="ABC123",
            file_size=18,
            source="test/source/",
            source_type=ModelSourceType.Path,
        )
    )

    assert _service(models_path, store).find_orphaned_models() == []


def test_a_directory_holding_a_registered_model_is_no_orphan_whatever_else_it_holds(
    models_path: Path, store: ModelRecordServiceSQL
) -> None:
    # Orphans are reported, and deleted, by their directory under the models root: reporting this one for its loose
    # weights would delete the registered model with them.
    _write_model_file(models_path / "family" / "model-a")
    _write_model_file(models_path / "family" / "loose-weights")
    store.add_model(
        TI_File_SD1_Config(
            path="family/model-a/model.safetensors",
            name="model a",
            hash="ABC123",
            file_size=18,
            source="test/source/",
            source_type=ModelSourceType.Path,
        )
    )

    assert _service(models_path, store).find_orphaned_models() == []


@pytest.mark.parametrize("root_name", ["tmpinstall_recovery", "recovered-model"])
def test_install_recovery_root_is_not_reported_or_deleted(
    root_name: str, models_path: Path, store: ModelRecordServiceSQL
) -> None:
    recovery_root = models_path / root_name
    _write_model_file(recovery_root)
    sentinel = models_path / f".{root_name}{INSTALL_RECOVERY_SENTINEL}"
    sentinel.write_text("preserve", encoding="utf-8")
    service = _service(models_path, store)

    assert service.find_orphaned_models() == []
    result = service.delete_orphaned_models([root_name])

    assert result[root_name].startswith("error:")
    assert (recovery_root / "model.safetensors").exists()
    assert sentinel.exists()


def test_parent_of_recovery_root_cannot_be_deleted(models_path: Path, store: ModelRecordServiceSQL) -> None:
    parent = models_path / "model-group"
    _write_model_file(parent / "ordinary-orphan")
    recovery_root = parent / "recovered-model"
    _write_model_file(recovery_root)
    sentinel = parent / f".{recovery_root.name}{INSTALL_RECOVERY_SENTINEL}"
    sentinel.write_text("preserve", encoding="utf-8")

    service = _service(models_path, store)

    assert "model-group" not in {orphan.path for orphan in service.find_orphaned_models()}
    result = service.delete_orphaned_models(["model-group"])

    assert result["model-group"].startswith("error:")
    assert (parent / "ordinary-orphan" / "model.safetensors").exists()
    assert (recovery_root / "model.safetensors").exists()
    assert sentinel.exists()


def test_models_root_cannot_be_deleted_as_an_orphan(models_path: Path, store: ModelRecordServiceSQL) -> None:
    _write_model_file(models_path / "real-model")

    result = _service(models_path, store).delete_orphaned_models(["."])

    assert result["."].startswith("error:")
    assert (models_path / "real-model" / "model.safetensors").exists()
