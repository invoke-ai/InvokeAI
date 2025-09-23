# Fixtures to support testing of the model_manager v2 installer, metadata and record store

import os
import shutil
from pathlib import Path

import pytest
from requests.sessions import Session
import requests_mock

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadQueueService, DownloadQueueServiceBase
from invokeai.app.services.model_install import ModelInstallService, ModelInstallServiceBase
from invokeai.app.services.model_load import ModelLoadService, ModelLoadServiceBase
from invokeai.app.services.model_manager import ModelManagerService, ModelManagerServiceBase
from invokeai.app.services.model_records import ModelRecordServiceBase, ModelRecordServiceSQL
from invokeai.backend.model_manager import BaseModelType, ModelFormat, ModelType, ModelVariantType
from invokeai.backend.model_manager.config import (
    LoRADiffusersConfig,
    MainCheckpointConfig,
    MainDiffusersConfig,
    VAEDiffusersConfig,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.taxonomy import ModelSourceType
from invokeai.backend.util.devices import TorchDevice
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
from tests.test_nodes import TestEventService


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


# Can be used to test diffusers model directory loading, but
# the test file adds ~10MB of space.
# @pytest.fixture
# def vae_directory(mm2_model_files: Path) -> Path:
#     return mm2_model_files / "taesdxl"


@pytest.fixture
def diffusers_dir(mm2_model_files: Path) -> Path:
    return mm2_model_files / "test-diffusers-main"


@pytest.fixture
def mm2_app_config(mm2_root_dir: Path) -> InvokeAIAppConfig:
    app_config = InvokeAIAppConfig(models_dir=mm2_root_dir / "models", log_level="info")
    app_config._root = mm2_root_dir
    return app_config


@pytest.fixture
def mm2_download_queue(mm2_session: Session) -> DownloadQueueServiceBase:
    download_queue = DownloadQueueService(requests_session=mm2_session)
    download_queue.start()
    yield download_queue
    download_queue.stop()


@pytest.fixture
def mm2_loader(mm2_app_config: InvokeAIAppConfig) -> ModelLoadServiceBase:
    ram_cache = ModelCache(
        execution_device_working_mem_gb=mm2_app_config.device_working_mem_gb,
        enable_partial_loading=mm2_app_config.enable_partial_loading,
        keep_ram_copy_of_weights=mm2_app_config.keep_ram_copy_of_weights,
        max_ram_cache_size_gb=mm2_app_config.max_cache_ram_gb,
        max_vram_cache_size_gb=mm2_app_config.max_cache_vram_gb,
        execution_device=TorchDevice.choose_torch_device(),
        logger=InvokeAILogger.get_logger(),
    )
    return ModelLoadService(
        app_config=mm2_app_config,
        ram_cache=ram_cache,
    )


@pytest.fixture
def mm2_installer(
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue: DownloadQueueServiceBase,
    mm2_session: Session,
) -> ModelInstallServiceBase:
    logger = InvokeAILogger.get_logger()
    db = create_mock_sqlite_database(mm2_app_config, logger)
    events = TestEventService()
    store = ModelRecordServiceSQL(db, logger)

    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=store,
        download_queue=mm2_download_queue,
        event_bus=events,
        session=mm2_session,
    )
    installer.start()
    yield installer
    installer.stop()


@pytest.fixture
def mm2_record_store(mm2_app_config: InvokeAIAppConfig) -> ModelRecordServiceBase:
    logger = InvokeAILogger.get_logger(config=mm2_app_config)
    db = create_mock_sqlite_database(mm2_app_config, logger)
    store = ModelRecordServiceSQL(db, logger)
    # add five simple config records to the database
    config1 = VAEDiffusersConfig(
        key="test_config_1",
        path="/tmp/foo1",
        format=ModelFormat.Diffusers,
        name="test2",
        base=BaseModelType.StableDiffusion2,
        type=ModelType.VAE,
        hash="111222333444",
        file_size=4096,
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
        file_size=8192,
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
        file_size=8193,
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
        file_size=5000,
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
        file_size=5001,
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
    sess = Session()
    adapter = requests_mock.Adapter()
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    adapter.register_uri(
        "GET",
        "https://test.com/missing_model.safetensors",
        text="missing",
        status_code=404,
        reason="NOT FOUND",
    )
    adapter.register_uri(
        "GET",
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo",
        content=RepoHFMetadata1,
        headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(RepoHFMetadata1))},
    )
    (
        adapter.register_uri(
            "GET",
            "https://huggingface.co/api/models/stabilityai/sdxl-turbo-nofp16",
            content=RepoHFMetadata1_nofp16,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(RepoHFMetadata1_nofp16)),
            },
        ),
    )
    adapter.register_uri(
        "GET",
        "https://civitai.com/api/v1/model-versions/242807",
        content=RepoCivitaiVersionMetadata1,
        headers={"Content-Length": str(len(RepoCivitaiVersionMetadata1))},
    )
    adapter.register_uri(
        "GET",
        "https://civitai.com/api/v1/models/215485",
        content=RepoCivitaiModelMetadata1,
        headers={"Content-Length": str(len(RepoCivitaiModelMetadata1))},
    )
    adapter.register_uri(
        "GET",
        "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/model_index.json",
        content=RepoHFModelJson1,
        headers={"Content-Length": str(len(RepoHFModelJson1))},
    )
    with open(embedding_file, "rb") as f:
        data = f.read()  # file is small - just 15K
    adapter.register_uri(
        "GET",
        "https://www.test.foo/download/test_embedding.safetensors",
        content=data,
        headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(data))},
    )
    adapter.register_uri(
        "GET",
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo",
        content=RepoHFMetadata1,
        headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(RepoHFMetadata1))},
    )
    adapter.register_uri(
        "GET",
        "https://huggingface.co/api/models/stabilityai/sdxl-turbo/revision/ModelRepoVariant.Default?blobs=True",
        content=RepoHFMetadata1,
        headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(RepoHFMetadata1))},
    )
    adapter.register_uri(
        "GET",
        "https://huggingface.co/api/models/InvokeAI-test/textual_inversion_tests?blobs=True",
        content=HFTestLoraMetadata,
        headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(HFTestLoraMetadata))},
    )
    adapter.register_uri(
        "GET",
        "https://huggingface.co/InvokeAI-test/textual_inversion_tests/resolve/main/learned_embeds-steps-1000.safetensors",
        content=data,
        headers={"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(data))},
    )
    for root, _, files in os.walk(diffusers_dir):
        for name in files:
            path = Path(root, name)
            url_base = path.relative_to(diffusers_dir).as_posix()
            url = f"https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/{url_base}"
            with open(path, "rb") as f:
                data = f.read()
            adapter.register_uri(
                "GET",
                url,
                content=data,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Content-Length": str(len(data)),
                },
            )

    for i in ["12345", "9999", "54321"]:
        content = (
            b"I am a safetensors file " + bytearray(i, "utf-8") + bytearray(32_000)
        )  # for pause tests, must make content large
        adapter.register_uri(
            "GET",
            f"http://www.civitai.com/models/{i}",
            content=content,
            headers={
                "Content-Length": str(len(content)),
                "Content-Disposition": f'filename="mock{i}.safetensors"',
            },
        )

    adapter.register_uri(
        "GET",
        "http://www.huggingface.co/foo.txt",
        content=content,
        headers={
            "Content-Length": str(len(content)),
            "Content-Disposition": 'filename="foo.safetensors"',
        },
    )

    # here are some malformed URLs to test
    # missing the content length
    adapter.register_uri(
        "GET",
        "http://www.civitai.com/models/missing",
        text="Missing content length",
        headers={
            "Content-Disposition": 'filename="missing.txt"',
        },
    )
    # not found test
    adapter.register_uri(
        "GET",
        "http://www.civitai.com/models/broken",
        text="Not found",
        status_code=404,
        reason="NOT FOUND",
    )

    return sess
