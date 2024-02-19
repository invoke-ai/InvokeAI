# Fixtures to support testing of the model_manager v2 installer, metadata and record store

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel
from pytest import FixtureRequest
from requests.sessions import Session
from requests_testadapter import TestAdapter, TestSession

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadQueueService, DownloadQueueServiceBase
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.model_install import ModelInstallService, ModelInstallServiceBase
from invokeai.app.services.model_load import ModelLoadService, ModelLoadServiceBase
from invokeai.app.services.model_manager import ModelManagerService, ModelManagerServiceBase
from invokeai.app.services.model_records import ModelRecordServiceBase, ModelRecordServiceSQL
from invokeai.backend.model_manager.config import (
    BaseModelType,
    LoRADiffusersConfig,
    MainCheckpointConfig,
    MainDiffusersConfig,
    ModelFormat,
    ModelSourceType,
    ModelType,
    ModelVariantType,
    VAEDiffusersConfig,
)
from invokeai.backend.model_manager.load import ModelCache, ModelConvertCache
from invokeai.backend.util.logging import InvokeAILogger
from tests.backend.model_manager.model_metadata.metadata_examples import (
    HFTestLoraMetadata,
    RepoCivitaiModelMetadata1,
    RepoCivitaiVersionMetadata1,
    RepoHFMetadata1,
    RepoHFMetadata1_nofp16,
    RepoHFModelJson1,
)
from tests.fixtures.sqlite_database import create_mock_sqlite_database


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
        self.events.append(DummyEvent(event_name=payload["event"], payload=payload["data"]))


# Create a temporary directory using the contents of `./data/invokeai_root` as the template
@pytest.fixture
def mm2_root_dir(tmp_path_factory) -> Path:
    root_template = Path(__file__).resolve().parent / "data" / "invokeai_root"
    temp_dir: Path = tmp_path_factory.mktemp("data") / "invokeai_root"
    shutil.copytree(root_template, temp_dir)
    return temp_dir


@pytest.fixture
def mm2_model_files(tmp_path_factory) -> Path:
    root_template = Path(__file__).resolve().parent / "data" / "test_files"
    temp_dir: Path = tmp_path_factory.mktemp("data") / "test_files"
    shutil.copytree(root_template, temp_dir)
    return temp_dir


@pytest.fixture
def embedding_file(mm2_model_files: Path) -> Path:
    return mm2_model_files / "test_embedding.safetensors"


@pytest.fixture
def diffusers_dir(mm2_model_files: Path) -> Path:
    return mm2_model_files / "test-diffusers-main"


@pytest.fixture
def mm2_app_config(mm2_root_dir: Path) -> InvokeAIAppConfig:
    app_config = InvokeAIAppConfig(models_dir=mm2_root_dir / "models", log_level="info")
    app_config.set_root(mm2_root_dir)
    return app_config


@pytest.fixture
def mm2_download_queue(mm2_session: Session, request: FixtureRequest) -> DownloadQueueServiceBase:
    download_queue = DownloadQueueService(requests_session=mm2_session)
    download_queue.start()

    def stop_queue() -> None:
        download_queue.stop()

    request.addfinalizer(stop_queue)
    return download_queue


@pytest.fixture
def mm2_loader(mm2_app_config: InvokeAIAppConfig) -> ModelLoadServiceBase:
    ram_cache = ModelCache(
        logger=InvokeAILogger.get_logger(),
        max_cache_size=mm2_app_config.ram,
        max_vram_cache_size=mm2_app_config.vram,
    )
    convert_cache = ModelConvertCache(mm2_app_config.convert_cache_path)
    return ModelLoadService(
        app_config=mm2_app_config,
        ram_cache=ram_cache,
        convert_cache=convert_cache,
    )


@pytest.fixture
def mm2_installer(
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue: DownloadQueueServiceBase,
    mm2_session: Session,
    request: FixtureRequest,
) -> ModelInstallServiceBase:
    logger = InvokeAILogger.get_logger()
    db = create_mock_sqlite_database(mm2_app_config, logger)
    events = DummyEventService()
    store = ModelRecordServiceSQL(db)

    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=store,
        download_queue=mm2_download_queue,
        event_bus=events,
        session=mm2_session,
    )
    installer.start()

    def stop_installer() -> None:
        installer.stop()
        time.sleep(0.1)  # avoid error message from the logger when it is closed before thread prints final message

    request.addfinalizer(stop_installer)
    return installer


@pytest.fixture
def mm2_record_store(mm2_app_config: InvokeAIAppConfig) -> ModelRecordServiceBase:
    logger = InvokeAILogger.get_logger(config=mm2_app_config)
    db = create_mock_sqlite_database(mm2_app_config, logger)
    store = ModelRecordServiceSQL(db)
    # add five simple config records to the database
    config1 = VAEDiffusersConfig(
        key="test_config_1",
        path="/tmp/foo1",
        format=ModelFormat.Diffusers,
        name="test2",
        base=BaseModelType.StableDiffusion2,
        type=ModelType.VAE,
        hash="111222333444",
        source="stabilityai/sdxl-vae",
        source_type=ModelSourceType.HFRepoID,
    )
    config2 = MainCheckpointConfig(
        key="test_config_2",
        path="/tmp/foo2.ckpt",
        name="model1",
        format=ModelFormat.Checkpoint,
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        config_path="/tmp/foo.yaml",
        variant=ModelVariantType.Normal,
        hash="111222333444",
        source="https://civitai.com/models/206883/split",
        source_type=ModelSourceType.Url,
    )
    config3 = MainDiffusersConfig(
        key="test_config_3",
        path="/tmp/foo3",
        format=ModelFormat.Diffusers,
        name="test3",
        base=BaseModelType.StableDiffusionXL,
        type=ModelType.Main,
        hash="111222333444",
        source="author3/model3",
        description="This is test 3",
        source_type=ModelSourceType.HFRepoID,
    )
    config4 = LoRADiffusersConfig(
        key="test_config_4",
        path="/tmp/foo4",
        format=ModelFormat.Diffusers,
        name="test4",
        base=BaseModelType.StableDiffusionXL,
        type=ModelType.LoRA,
        hash="111222333444",
        source="author4/model4",
        source_type=ModelSourceType.HFRepoID,
    )
    config5 = LoRADiffusersConfig(
        key="test_config_5",
        path="/tmp/foo5",
        format=ModelFormat.Diffusers,
        name="test5",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.LoRA,
        hash="111222333444",
        source="author4/model5",
        source_type=ModelSourceType.HFRepoID,
    )
    store.add_model(config1)
    store.add_model(config2)
    store.add_model(config3)
    store.add_model(config4)
    store.add_model(config5)
    return store


@pytest.fixture
def mm2_model_manager(
    mm2_record_store: ModelRecordServiceBase, mm2_installer: ModelInstallServiceBase, mm2_loader: ModelLoadServiceBase
) -> ModelManagerServiceBase:
    return ModelManagerService(store=mm2_record_store, install=mm2_installer, load=mm2_loader)


@pytest.fixture
def mm2_session(embedding_file: Path, diffusers_dir: Path) -> Session:
    """This fixtures defines a series of mock URLs for testing download and installation."""
    sess: Session = TestSession()
    sess.mount(
        "https://test.com/missing_model.safetensors",
        TestAdapter(
            b"missing",
            status=404,
        ),
    )
    sess.mount(
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo",
        TestAdapter(
            RepoHFMetadata1,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": len(RepoHFMetadata1)},
        ),
    )
    sess.mount(
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo-nofp16",
        TestAdapter(
            RepoHFMetadata1_nofp16,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": len(RepoHFMetadata1_nofp16)},
        ),
    )
    sess.mount(
        "https://civitai.com/api/v1/model-versions/242807",
        TestAdapter(
            RepoCivitaiVersionMetadata1,
            headers={
                "Content-Length": len(RepoCivitaiVersionMetadata1),
            },
        ),
    )
    sess.mount(
        "https://civitai.com/api/v1/models/215485",
        TestAdapter(
            RepoCivitaiModelMetadata1,
            headers={
                "Content-Length": len(RepoCivitaiModelMetadata1),
            },
        ),
    )
    sess.mount(
        "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/model_index.json",
        TestAdapter(
            RepoHFModelJson1,
            headers={
                "Content-Length": len(RepoHFModelJson1),
            },
        ),
    )
    with open(embedding_file, "rb") as f:
        data = f.read()  # file is small - just 15K
    sess.mount(
        "https://www.test.foo/download/test_embedding.safetensors",
        TestAdapter(data, headers={"Content-Type": "application/octet-stream", "Content-Length": len(data)}),
    )
    sess.mount(
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo",
        TestAdapter(
            RepoHFMetadata1,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": len(RepoHFMetadata1)},
        ),
    )
    sess.mount(
        "https://huggingface.co/api/models/InvokeAI-test/textual_inversion_tests?blobs=True",
        TestAdapter(
            HFTestLoraMetadata,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": len(HFTestLoraMetadata)},
        ),
    )
    sess.mount(
        "https://huggingface.co/InvokeAI-test/textual_inversion_tests/resolve/main/learned_embeds-steps-1000.safetensors",
        TestAdapter(
            data,
            headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": len(data)},
        ),
    )
    for root, _, files in os.walk(diffusers_dir):
        for name in files:
            path = Path(root, name)
            url_base = path.relative_to(diffusers_dir).as_posix()
            url = f"https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/{url_base}"
            with open(path, "rb") as f:
                data = f.read()
            sess.mount(
                url,
                TestAdapter(
                    data,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "Content-Length": len(data),
                    },
                ),
            )
    return sess
