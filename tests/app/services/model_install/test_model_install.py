"""
Test the model installer
"""

import platform
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError
from pydantic_core import Url

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.events.events_common import (
    ModelInstallCompleteEvent,
    ModelInstallDownloadProgressEvent,
    ModelInstallDownloadsCompleteEvent,
    ModelInstallDownloadStartedEvent,
    ModelInstallErrorEvent,
    ModelInstallStartedEvent,
)
from invokeai.app.services.model_install import (
    HFModelSource,
    ModelInstallServiceBase,
)
from invokeai.app.services.model_install.model_install_common import (
    InstallStatus,
    LocalModelSource,
    ModelInstallJob,
    URLModelSource,
)
from invokeai.app.services.model_records import ModelRecordChanges, UnknownModelException
from invokeai.backend.model_manager.config import (
    BaseModelType,
    InvalidModelConfigException,
    ModelFormat,
    ModelRepoVariant,
    ModelType,
)
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403
from tests.test_nodes import TestEventService

OS = platform.uname().system


def test_registration(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    matches = store.search_by_attr(model_name="test_embedding")
    assert len(matches) == 0
    key = mm2_installer.register_path(embedding_file)
    # Not raising here is sufficient - key should be UUIDv4
    uuid.UUID(key, version=4)


def test_registration_meta(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record is not None
    assert model_record.name == "test_embedding"
    assert model_record.type == ModelType.TextualInversion
    assert Path(model_record.path) == embedding_file
    assert Path(model_record.path).exists()
    assert model_record.base == BaseModelType("sd-1")
    assert model_record.description is not None
    assert model_record.source is not None
    assert Path(model_record.source) == embedding_file


def test_registration_meta_override_fail(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    key = None
    with pytest.raises((ValidationError, InvalidModelConfigException)):
        key = mm2_installer.register_path(
            embedding_file, ModelRecordChanges(name="banana_sushi", type=ModelType("lora"))
        )
    assert key is None


def test_registration_meta_override_succeed(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(
        embedding_file, ModelRecordChanges(name="banana_sushi", source="fake/repo_id", key="xyzzy")
    )
    model_record = store.get_model(key)
    assert model_record.name == "banana_sushi"
    assert model_record.source == "fake/repo_id"
    assert model_record.key == "xyzzy"


def test_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record.path.endswith("sd-1/embedding/test_embedding.safetensors")
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert model_record.source == embedding_file.as_posix()


def test_rename(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record.path.endswith("sd-1/embedding/test_embedding.safetensors")
    store.update_model(key, ModelRecordChanges(name="new model name", base=BaseModelType("sd-2")))
    new_model_record = mm2_installer.sync_model_path(key)
    # Renaming the model record shouldn't rename the file
    assert new_model_record.name == "new model name"
    assert new_model_record.path.endswith("sd-2/embedding/test_embedding.safetensors")


@pytest.mark.parametrize(
    "fixture_name,size,destination",
    [
        ("embedding_file", 15440, "sd-1/embedding/test_embedding.safetensors"),
        ("diffusers_dir", 8241 if OS == "Windows" else 7907, "sdxl/main/test-diffusers-main"),  # EOL chars
    ],
)
def test_background_install(
    mm2_installer: ModelInstallServiceBase,
    fixture_name: str,
    size: int,
    destination: str,
    mm2_app_config: InvokeAIAppConfig,
    request: pytest.FixtureRequest,
) -> None:
    """Note: may want to break this down into several smaller unit tests."""
    path: Path = request.getfixturevalue(fixture_name)
    description = "Test of metadata assignment"
    source = LocalModelSource(path=path, inplace=False)
    job = mm2_installer.import_model(source, config=ModelRecordChanges(description=description))
    assert job is not None
    assert isinstance(job, ModelInstallJob)

    # See if job is registered properly
    assert job in mm2_installer.get_job_by_source(source)

    # test that the job object tracked installation correctly
    jobs = mm2_installer.wait_for_installs()
    assert len(jobs) > 0
    my_job = [x for x in jobs if x.source == source]
    assert len(my_job) == 1
    assert job == my_job[0]
    assert job.status == InstallStatus.COMPLETED
    assert job.total_bytes == size

    # test that the expected events were issued
    bus: TestEventService = mm2_installer.event_bus
    assert bus
    assert hasattr(bus, "events")

    assert len(bus.events) == 2
    assert isinstance(bus.events[0], ModelInstallStartedEvent)
    assert isinstance(bus.events[1], ModelInstallCompleteEvent)
    assert Path(bus.events[0].source) == source
    assert Path(bus.events[1].source) == source
    key = bus.events[1].key
    assert key is not None

    # see if the thing actually got installed at the expected location
    model_record = mm2_installer.record_store.get_model(key)
    assert model_record is not None
    assert model_record.path.endswith(destination)
    assert (mm2_app_config.models_path / model_record.path).exists()

    # see if metadata was properly passed through
    assert model_record.description == description

    # see if job filtering works
    assert mm2_installer.get_job_by_source(source)[0] == job

    # see if prune works properly
    mm2_installer.prune_jobs()
    assert not mm2_installer.get_job_by_source(source)


def test_not_inplace_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    source = LocalModelSource(path=embedding_file, inplace=False)
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs()
    assert job is not None
    assert job.config_out is not None
    assert Path(job.config_out.path) != embedding_file
    assert (mm2_app_config.models_path / job.config_out.path).exists()


def test_inplace_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    source = LocalModelSource(path=embedding_file, inplace=True)
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs()
    assert job is not None
    assert job.config_out is not None
    assert Path(job.config_out.path) == embedding_file
    assert Path(job.config_out.path).exists()


def test_delete_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert embedding_file.exists()  # original should still be there after installation
    mm2_installer.delete(key)
    assert not (
        mm2_app_config.models_path / model_record.path
    ).exists()  # after deletion, installed copy should not exist
    assert embedding_file.exists()  # but original should still be there
    with pytest.raises(UnknownModelException):
        store.get_model(key)


def test_delete_register(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(embedding_file)
    model_record = store.get_model(key)
    assert Path(model_record.path).exists()
    assert embedding_file.exists()  # original should still be there after installation
    mm2_installer.delete(key)
    assert Path(model_record.path).exists()
    with pytest.raises(UnknownModelException):
        store.get_model(key)


@pytest.mark.timeout(timeout=10, method="thread")
def test_simple_download(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    bus: TestEventService = mm2_installer.event_bus
    store = mm2_installer.record_store
    assert store is not None
    assert bus is not None
    assert hasattr(bus, "events")  # the dummy event service has this

    job = mm2_installer.import_model(source)
    assert job.source == source
    job_list = mm2_installer.wait_for_installs(timeout=10)
    assert len(job_list) == 1
    assert job.complete
    assert job.config_out

    key = job.config_out.key
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()

    assert len(bus.events) == 5
    assert isinstance(bus.events[0], ModelInstallDownloadStartedEvent)  # download starts
    assert isinstance(bus.events[1], ModelInstallDownloadProgressEvent)  # download progresses
    assert isinstance(bus.events[2], ModelInstallDownloadsCompleteEvent)  # download completed
    assert isinstance(bus.events[3], ModelInstallStartedEvent)  # install started
    assert isinstance(bus.events[4], ModelInstallCompleteEvent)  # install completed


@pytest.mark.timeout(timeout=10, method="thread")
def test_huggingface_install(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://huggingface.co/stabilityai/sdxl-turbo"))

    bus: TestEventService = mm2_installer.event_bus
    store = mm2_installer.record_store
    assert isinstance(bus, EventServiceBase)
    assert store is not None

    job = mm2_installer.import_model(source)
    job_list = mm2_installer.wait_for_installs(timeout=10)
    assert len(job_list) == 1
    assert job.complete
    assert job.config_out

    key = job.config_out.key
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert model_record.type == ModelType.Main
    assert model_record.format == ModelFormat.Diffusers

    assert any(isinstance(x, ModelInstallStartedEvent) for x in bus.events)
    assert any(isinstance(x, ModelInstallDownloadProgressEvent) for x in bus.events)
    assert any(isinstance(x, ModelInstallCompleteEvent) for x in bus.events)
    assert len(bus.events) >= 3


@pytest.mark.timeout(timeout=10, method="thread")
def test_huggingface_repo_id(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = HFModelSource(repo_id="stabilityai/sdxl-turbo", variant=ModelRepoVariant.Default)

    bus = mm2_installer.event_bus
    store = mm2_installer.record_store
    assert isinstance(bus, EventServiceBase)
    assert store is not None

    job = mm2_installer.import_model(source)
    job_list = mm2_installer.wait_for_installs(timeout=10)
    assert len(job_list) == 1
    assert job.complete
    assert job.config_out

    key = job.config_out.key
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert model_record.type == ModelType.Main
    assert model_record.format == ModelFormat.Diffusers

    assert hasattr(bus, "events")  # the dummyeventservice has this
    assert len(bus.events) >= 3
    event_types = [type(x) for x in bus.events]
    assert all(
        x in event_types
        for x in [
            ModelInstallDownloadProgressEvent,
            ModelInstallDownloadsCompleteEvent,
            ModelInstallStartedEvent,
            ModelInstallCompleteEvent,
        ]
    )

    completed_events = [x for x in bus.events if isinstance(x, ModelInstallCompleteEvent)]
    downloading_events = [x for x in bus.events if isinstance(x, ModelInstallDownloadProgressEvent)]
    assert completed_events[0].total_bytes == downloading_events[-1].bytes
    assert job.total_bytes == completed_events[0].total_bytes
    print(downloading_events[-1])
    print(job.download_parts)
    assert job.total_bytes == sum(x["total_bytes"] for x in downloading_events[-1].parts)


def test_404_download(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://test.com/missing_model.safetensors"))
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status == InstallStatus.ERROR
    assert job.errored
    assert job.error_type == "HTTPError"
    assert job.error
    assert "NOT FOUND" in job.error
    assert job.error_traceback is not None
    assert job.error_traceback.startswith("Traceback")
    bus = mm2_installer.event_bus
    assert bus is not None
    assert hasattr(bus, "events")  # the dummyeventservice has this
    event_types = [type(x) for x in bus.events]
    assert ModelInstallErrorEvent in event_types


def test_other_error_during_install(
    monkeypatch: pytest.MonkeyPatch, mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(
        "invokeai.app.services.model_install.model_install_default.ModelInstallService._register_or_install",
        raise_runtime_error,
    )
    source = LocalModelSource(path=Path("tests/data/embedding/test_embedding.safetensors"))
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status == InstallStatus.ERROR
    assert job.errored
    assert job.error_type == "RuntimeError"
    assert job.error == "Test error"


@pytest.mark.parametrize(
    "model_params",
    [
        # SDXL, Lora
        {
            "repo_id": "InvokeAI-test/textual_inversion_tests::learned_embeds-steps-1000.safetensors",
            "name": "test_lora",
            "type": "embedding",
        },
        # SDXL, Lora - incorrect type
        {
            "repo_id": "InvokeAI-test/textual_inversion_tests::learned_embeds-steps-1000.safetensors",
            "name": "test_lora",
            "type": "lora",
        },
    ],
)
@pytest.mark.timeout(timeout=10, method="thread")
def test_heuristic_import_with_type(mm2_installer: ModelInstallServiceBase, model_params: Dict[str, str]):
    """Test whether or not type is respected on configs when passed to heuristic import."""
    assert "name" in model_params and "type" in model_params
    config1: Dict[str, Any] = {
        "name": f"{model_params['name']}_1",
        "type": model_params["type"],
        "hash": "placeholder1",
    }
    config2: Dict[str, Any] = {
        "name": f"{model_params['name']}_2",
        "type": ModelType(model_params["type"]),
        "hash": "placeholder2",
    }
    assert "repo_id" in model_params
    install_job1 = mm2_installer.heuristic_import(source=model_params["repo_id"], config=config1)
    mm2_installer.wait_for_job(install_job1, timeout=10)
    if model_params["type"] != "embedding":
        assert install_job1.errored
        assert install_job1.error_type == "InvalidModelConfigException"
        return
    assert install_job1.complete
    assert install_job1.config_out if model_params["type"] == "embedding" else not install_job1.config_out

    install_job2 = mm2_installer.heuristic_import(source=model_params["repo_id"], config=config2)
    mm2_installer.wait_for_job(install_job2, timeout=10)
    assert install_job2.complete
    assert install_job2.config_out if model_params["type"] == "embedding" else not install_job2.config_out
