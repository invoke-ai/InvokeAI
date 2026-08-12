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
