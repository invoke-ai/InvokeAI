"""The orphan scan must not report a model that is still being written.

An orphan is defined as model files under the models root with no database record — which is also
an exact description of a conversion in progress. Model conversion builds its diffusers copy on the
models volume (so the finished result can be moved into place rather than copied across a
filesystem boundary), and `DELETE /sync/orphaned` rmtrees whatever the scan reported. Before both
routes ran in the threadpool they could not overlap; now they can, so the scratch directory has to
be invisible to the scan by construction.
"""

from logging import Logger
from pathlib import Path

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.model_install.model_install_common import INSTALL_RECOVERY_SENTINEL
from invokeai.app.services.orphaned_models import CONVERSION_SCRATCH_DIRNAME, OrphanedModelsService
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


@pytest.fixture
def db() -> SqliteDatabase:
    database = SqliteDatabase(db_path=None, logger=Logger("test_orphaned_models"), verbose=False)
    database._conn.execute("CREATE TABLE models (id TEXT PRIMARY KEY, config TEXT NOT NULL);")
    database._conn.commit()
    return database


@pytest.fixture
def models_path(tmp_path: Path) -> Path:
    path = tmp_path / "models"
    path.mkdir()
    return path


def _service(models_path: Path, db: SqliteDatabase) -> OrphanedModelsService:
    config = InvokeAIAppConfig()
    config._root = models_path.parent
    assert config.models_path == models_path
    return OrphanedModelsService(config=config, db=db)


def _write_model_file(directory: Path, name: str = "model.safetensors") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"not really a model")


def test_conversion_scratch_directory_is_not_reported_as_orphaned(models_path: Path, db: SqliteDatabase) -> None:
    # What a conversion looks like on disk while it runs: a TemporaryDirectory under the scratch
    # area, holding the diffusers copy it has written so far.
    _write_model_file(models_path / CONVERSION_SCRATCH_DIRNAME / "tmp8f2b1c" / "sd-v1-5")

    orphans = _service(models_path, db).find_orphaned_models()

    assert orphans == [], (
        "The scan reported a conversion's working directory. `DELETE /sync/orphaned` would delete "
        "it while the conversion is still writing into it."
    )


def test_a_real_orphan_is_still_reported(models_path: Path, db: SqliteDatabase) -> None:
    """The control: the skip must be targeted, not a scan that has stopped finding anything."""
    _write_model_file(models_path / "some-unregistered-model")

    orphans = _service(models_path, db).find_orphaned_models()

    assert [orphan.path for orphan in orphans] == ["some-unregistered-model"]


def test_conversion_scratch_directory_cannot_be_deleted(models_path: Path, db: SqliteDatabase) -> None:
    _write_model_file(models_path / CONVERSION_SCRATCH_DIRNAME / "active-conversion" / "sd-v1-5")

    result = _service(models_path, db).delete_orphaned_models([CONVERSION_SCRATCH_DIRNAME])

    assert result[CONVERSION_SCRATCH_DIRNAME].startswith("error:")
    assert (models_path / CONVERSION_SCRATCH_DIRNAME).exists()


@pytest.mark.parametrize("root_name", ["tmpinstall_recovery", "recovered-model"])
def test_install_recovery_root_is_not_reported_or_deleted(
    root_name: str, models_path: Path, db: SqliteDatabase
) -> None:
    recovery_root = models_path / root_name
    _write_model_file(recovery_root)
    sentinel = models_path / f".{root_name}{INSTALL_RECOVERY_SENTINEL}"
    sentinel.write_text("preserve", encoding="utf-8")
    service = _service(models_path, db)

    assert service.find_orphaned_models() == []
    result = service.delete_orphaned_models([root_name])

    assert result[root_name].startswith("error:")
    assert (recovery_root / "model.safetensors").exists()
    assert sentinel.exists()


def test_parent_of_recovery_root_cannot_be_deleted(models_path: Path, db: SqliteDatabase) -> None:
    parent = models_path / "model-group"
    _write_model_file(parent / "ordinary-orphan")
    recovery_root = parent / "recovered-model"
    _write_model_file(recovery_root)
    sentinel = parent / f".{recovery_root.name}{INSTALL_RECOVERY_SENTINEL}"
    sentinel.write_text("preserve", encoding="utf-8")

    service = _service(models_path, db)

    assert "model-group" not in {orphan.path for orphan in service.find_orphaned_models()}
    result = service.delete_orphaned_models(["model-group"])

    assert result["model-group"].startswith("error:")
    assert (parent / "ordinary-orphan" / "model.safetensors").exists()
    assert (recovery_root / "model.safetensors").exists()
    assert sentinel.exists()


def test_models_root_cannot_be_deleted_as_an_orphan(models_path: Path, db: SqliteDatabase) -> None:
    _write_model_file(models_path / "real-model")

    result = _service(models_path, db).delete_orphaned_models(["."])

    assert result["."].startswith("error:")
    assert (models_path / "real-model" / "model.safetensors").exists()
