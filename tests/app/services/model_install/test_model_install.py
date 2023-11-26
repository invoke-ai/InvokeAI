"""
Test the model installer
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel, ValidationError

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.model_install import (
    InstallStatus,
    ModelInstallJob,
    ModelInstallService,
    ModelInstallServiceBase,
    UnknownInstallJobException,
)
from invokeai.app.services.model_records import ModelRecordServiceBase, ModelRecordServiceSQL, UnknownModelException
from invokeai.app.services.shared.sqlite import SqliteDatabase
from invokeai.backend.model_manager.config import BaseModelType, ModelType
from invokeai.backend.util.logging import InvokeAILogger


@pytest.fixture
def test_file(datadir: Path) -> Path:
    return datadir / "test_embedding.safetensors"


@pytest.fixture
def app_config(datadir: Path) -> InvokeAIAppConfig:
    return InvokeAIAppConfig(
        root=datadir / "root",
        models_dir=datadir / "root/models",
    )


@pytest.fixture
def store(app_config: InvokeAIAppConfig) -> ModelRecordServiceBase:
    database = SqliteDatabase(app_config, InvokeAILogger.get_logger(config=app_config))
    store: ModelRecordServiceBase = ModelRecordServiceSQL(database)
    return store


@pytest.fixture
def installer(app_config: InvokeAIAppConfig,
              store: ModelRecordServiceBase) -> ModelInstallServiceBase:
    return ModelInstallService(app_config=app_config,
                               record_store=store,
                               event_bus=DummyEventService(),
                               )


class DummyEvent(BaseModel):
    """Dummy Event to use with Dummy Event service."""

    event_name: str
    payload: Dict[str, Any]


class DummyEventService(EventServiceBase):
    """Dummy event service for testing."""

    events: List[DummyEvent]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    def dispatch(self, event_name: str, payload: Any) -> None:
        """Dispatch an event by appending it to self.events."""
        self.events.append(
            DummyEvent(event_name=payload['event'],
                       payload=payload['data'])
        )

def test_registration(installer: ModelInstallServiceBase, test_file: Path) -> None:
    store = installer.record_store
    matches = store.search_by_attr(model_name="test_embedding")
    assert len(matches) == 0
    key = installer.register_path(test_file)
    assert key is not None
    assert len(key) == 32

def test_registration_meta(installer: ModelInstallServiceBase, test_file: Path) -> None:
    store = installer.record_store
    key = installer.register_path(test_file)
    model_record = store.get_model(key)
    assert model_record is not None
    assert model_record.name == "test_embedding"
    assert model_record.type == ModelType.TextualInversion
    assert Path(model_record.path) == test_file
    assert model_record.base == BaseModelType('sd-1')
    assert model_record.description is not None
    assert model_record.source is not None
    assert Path(model_record.source) == test_file

def test_registration_meta_override_fail(installer: ModelInstallServiceBase, test_file: Path) -> None:
    key = None
    with pytest.raises(ValidationError):
        key = installer.register_path(test_file, {"name": "banana_sushi", "type": ModelType("lora")})
    assert key is None

def test_registration_meta_override_succeed(installer: ModelInstallServiceBase, test_file: Path) -> None:
    store = installer.record_store
    key = installer.register_path(test_file,
                                  {
                                      "name": "banana_sushi",
                                      "source": "fake/repo_id",
                                      "current_hash": "New Hash"
                                  }
                                  )
    model_record = store.get_model(key)
    assert model_record.name == "banana_sushi"
    assert model_record.source == "fake/repo_id"
    assert model_record.current_hash == "New Hash"

def test_install(installer: ModelInstallServiceBase, test_file: Path, app_config: InvokeAIAppConfig) -> None:
    store = installer.record_store
    key = installer.install_path(test_file)
    model_record = store.get_model(key)
    assert model_record.path == "sd-1/embedding/test_embedding.safetensors"
    assert model_record.source == test_file.as_posix()

def test_background_install(installer: ModelInstallServiceBase, test_file: Path, app_config: InvokeAIAppConfig) -> None:
    """Note: may want to break this down into several smaller unit tests."""
    source = test_file
    description = "Test of metadata assignment"
    job = installer.import_model(source, inplace=False, config={"description": description})
    assert job is not None
    assert isinstance(job, ModelInstallJob)

    # See if job is registered properly
    assert installer.get_job(source) == job

    # test that the job object tracked installation correctly
    jobs = installer.wait_for_installs()
    assert jobs[source] is not None
    assert jobs[source] == job
    assert jobs[source].status == InstallStatus.COMPLETED

    # test that the expected events were issued
    bus = installer.event_bus
    assert bus is not None                         # sigh - ruff is a stickler for type checking
    assert isinstance(bus, DummyEventService)
    assert len(bus.events) == 2
    event_names = [x.event_name for x in bus.events]
    assert "model_install_started" in event_names
    assert "model_install_completed" in event_names
    assert bus.events[0].payload["source"] == source.as_posix()
    assert bus.events[1].payload["source"] == source.as_posix()
    key = bus.events[1].payload["key"]
    assert key is not None

    # see if the thing actually got installed at the expected location
    model_record = installer.record_store.get_model(key)
    assert model_record is not None
    assert model_record.path == "sd-1/embedding/test_embedding.safetensors"
    assert Path(app_config.models_dir / model_record.path).exists()

    # see if metadata was properly passed through
    assert model_record.description == description

    # see if prune works properly
    installer.prune_jobs()
    with pytest.raises(UnknownInstallJobException):
        assert installer.get_job(source)

def test_delete_install(installer: ModelInstallServiceBase, test_file: Path, app_config: InvokeAIAppConfig):
    store = installer.record_store
    key = installer.install_path(test_file)
    model_record = store.get_model(key)
    assert Path(app_config.models_dir / model_record.path).exists()
    assert test_file.exists()  # original should still be there after installation
    installer.delete(key)
    assert not Path(app_config.models_dir / model_record.path).exists()  # after deletion, installed copy should not exist
    assert test_file.exists()  # but original should still be there
    with pytest.raises(UnknownModelException):
        store.get_model(key)

def test_delete_register(installer: ModelInstallServiceBase, test_file: Path, app_config: InvokeAIAppConfig):
    store = installer.record_store
    key = installer.register_path(test_file)
    model_record = store.get_model(key)
    assert Path(app_config.models_dir / model_record.path).exists()
    assert test_file.exists()  # original should still be there after installation
    installer.delete(key)
    assert Path(app_config.models_dir / model_record.path).exists()
    with pytest.raises(UnknownModelException):
        store.get_model(key)
