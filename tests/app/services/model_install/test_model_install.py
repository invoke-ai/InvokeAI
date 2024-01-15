"""
Test the model installer
"""

import platform
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic.networks import Url

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.model_install import (
    InstallStatus,
    LocalModelSource,
    ModelInstallJob,
    ModelInstallServiceBase,
    URLModelSource,
)
from invokeai.app.services.model_records import UnknownModelException
from invokeai.backend.model_manager.config import BaseModelType, ModelFormat, ModelType
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403

OS = platform.uname().system


def test_registration(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    matches = store.search_by_attr(model_name="test_embedding")
    assert len(matches) == 0
    key = mm2_installer.register_path(embedding_file)
    assert key is not None
    assert len(key) == 32


def test_registration_meta(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record is not None
    assert model_record.name == "test_embedding"
    assert model_record.type == ModelType.TextualInversion
    assert Path(model_record.path) == embedding_file
    assert model_record.base == BaseModelType("sd-1")
    assert model_record.description is not None
    assert model_record.source is not None
    assert Path(model_record.source) == embedding_file


def test_registration_meta_override_fail(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    key = None
    with pytest.raises(ValidationError):
        key = mm2_installer.register_path(embedding_file, {"name": "banana_sushi", "type": ModelType("lora")})
    assert key is None


def test_registration_meta_override_succeed(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(
        embedding_file, {"name": "banana_sushi", "source": "fake/repo_id", "current_hash": "New Hash"}
    )
    model_record = store.get_model(key)
    assert model_record.name == "banana_sushi"
    assert model_record.source == "fake/repo_id"
    assert model_record.current_hash == "New Hash"


def test_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record.path == "sd-1/embedding/test_embedding.safetensors"
    assert model_record.source == embedding_file.as_posix()


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
    job = mm2_installer.import_model(source, config={"description": description})
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
    bus = mm2_installer.event_bus
    assert bus
    assert hasattr(bus, "events")

    assert len(bus.events) == 2
    event_names = [x.event_name for x in bus.events]
    assert "model_install_running" in event_names
    assert "model_install_completed" in event_names
    assert Path(bus.events[0].payload["source"]) == source
    assert Path(bus.events[1].payload["source"]) == source
    key = bus.events[1].payload["key"]
    assert key is not None

    # see if the thing actually got installed at the expected location
    model_record = mm2_installer.record_store.get_model(key)
    assert model_record is not None
    assert model_record.path == destination
    assert Path(mm2_app_config.models_dir / model_record.path).exists()

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
    assert Path(mm2_app_config.models_dir / job.config_out.path).exists()


def test_inplace_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    source = LocalModelSource(path=embedding_file, inplace=True)
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs()
    assert job is not None
    assert job.config_out is not None
    assert Path(job.config_out.path) == embedding_file


def test_delete_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)
    model_record = store.get_model(key)
    assert Path(mm2_app_config.models_dir / model_record.path).exists()
    assert embedding_file.exists()  # original should still be there after installation
    mm2_installer.delete(key)
    assert not Path(
        mm2_app_config.models_dir / model_record.path
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
    assert Path(mm2_app_config.models_dir / model_record.path).exists()
    assert embedding_file.exists()  # original should still be there after installation
    mm2_installer.delete(key)
    assert Path(mm2_app_config.models_dir / model_record.path).exists()
    with pytest.raises(UnknownModelException):
        store.get_model(key)


def test_simple_download(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    bus = mm2_installer.event_bus
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
    assert Path(mm2_app_config.models_dir / model_record.path).exists()

    assert len(bus.events) == 3
    event_names = [x.event_name for x in bus.events]
    assert event_names == ["model_install_downloading", "model_install_running", "model_install_completed"]


def test_huggingface_download(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://huggingface.co/stabilityai/sdxl-turbo"))

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
    assert Path(mm2_app_config.models_dir / model_record.path).exists()
    assert model_record.type == ModelType.Main
    assert model_record.format == ModelFormat.Diffusers

    assert hasattr(bus, "events")  # the dummyeventservice has this
    assert len(bus.events) >= 3
    event_names = {x.event_name for x in bus.events}
    assert event_names == {"model_install_downloading", "model_install_running", "model_install_completed"}


def test_404_download(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://test.com/missing_model.safetensors"))
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status == InstallStatus.ERROR
    assert job.errored
    assert job.error_type == "HTTPError"
    assert job.error
    assert "NOT FOUND" in job.error
    assert "Traceback" in job.error
