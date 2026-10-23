"""Model record edits and re-identification through the model manager: they write the record and drop cached
instances of the model exactly when the change affects how it loads. The load fence these changes announce
themselves to is covered by `tests/app/services/model_load/test_stale_model_load_fence.py`."""

import gc
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

import pytest
import torch
from safetensors.torch import save_file

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadQueueService
from invokeai.app.services.model_install import ModelInstallService
from invokeai.app.services.model_load import ModelLoadService
from invokeai.app.services.model_manager import ModelManagerService
from invokeai.app.services.model_records import (
    InvalidModelException,
    ModelRecordChanges,
    ModelRecordServiceSQL,
    UnknownModelException,
)
from invokeai.backend.model_manager.configs.clip_embed import CLIPEmbed_Checkpoint_L_Config
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.configs.ip_adapter import IPAdapter_InvokeAI_SD1_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_SD1_Config, Main_Checkpoint_SDXL_Config
from invokeai.backend.model_manager.configs.spandrel import Spandrel_Checkpoint_Config
from invokeai.backend.model_manager.load import LoadedModel, ModelCache, ModelLoader
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelSourceType,
    ModelVariantType,
    SubModelType,
)
from invokeai.backend.util.logging import InvokeAILogger
from tests.backend.model_manager.load.state_dicts import clip_g_comfy_keys, clip_l_comfy_keys
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors
from tests.fixtures.spandrel_models import write_compact_upscaler
from tests.fixtures.sqlite_database import create_mock_sqlite_database

KEY = "model-key"
MODEL_PATH = "sd-1/main/model.safetensors"


class _RecordingLoader(ModelLoader):
    """Builds a tiny module and records the config class each construction was built from."""

    built_from: list[str] = []

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        self.built_from.append(type(config).__name__)
        return torch.nn.Linear(2, 2)


class _Registry:
    @classmethod
    def get_implementation(cls, config: AnyModelConfig, submodel_type: Optional[SubModelType]):
        return _RecordingLoader, config, submodel_type


class _NoEvents:
    def emit_model_load_started(self, *_args) -> None:
        pass

    def emit_model_load_complete(self, *_args) -> None:
        pass


class _Harness:
    def __init__(self, models_path: Path) -> None:
        self.models_path = models_path
        app_config = InvokeAIAppConfig(use_memory_db=True, models_dir=models_path)
        logger = InvokeAILogger.get_logger()
        self.store = ModelRecordServiceSQL(create_mock_sqlite_database(app_config, logger), logger)
        self.caches = [
            ModelCache(
                execution_device_working_mem_gb=1.0,
                enable_partial_loading=False,
                keep_ram_copy_of_weights=True,
                max_ram_cache_size_gb=1.0,
                max_vram_cache_size_gb=0.0,
                execution_device="cpu",
                logger=logger,
                # Keep the process-global weight store out of it, so no other test's model can be adopted.
                shared_cpu_weights=None,
            )
            for _ in range(2)
        ]
        load = ModelLoadService(
            app_config=app_config,
            ram_cache=self.caches[0],
            registry=_Registry,  # type: ignore[arg-type]
            ram_caches={"cpu": self.caches[0], "cpu:1": self.caches[1]},
        )
        install = ModelInstallService(
            app_config=app_config, record_store=self.store, download_queue=DownloadQueueService()
        )
        self.manager = ModelManagerService(store=self.store, install=install, load=load)
        # Only the load service is started: it reads the record through the invoker, and the installer's start
        # would scan the models directory.
        load.start(SimpleNamespace(services=SimpleNamespace(model_manager=self.manager, events=_NoEvents())))  # type: ignore[arg-type]

    def cache_model(self, key: str = KEY) -> None:
        """Put a model and one of its submodels into every device cache, beside an unrelated model."""
        for cache in self.caches:
            cache.put(key, torch.ones(2))
            cache.put(f"{key}:vae", torch.ones(2))
            cache.put("other-model", torch.ones(2))

    def is_cached(self, key: str = KEY) -> bool:
        """Whether the model is cached, failing the test if the device caches or its entries disagree."""
        states = {entry in cache._cached_models for cache in self.caches for entry in (key, f"{key}:vae")}
        assert len(states) == 1, "the change dropped only some of the model's entries"
        assert all("other-model" in cache._cached_models for cache in self.caches)
        return states.pop()


@pytest.fixture(autouse=True)
def _collect_only_this_tests_objects():
    """Every drop runs gc.collect(), and a full collection walks the worker's whole heap. Freezing what existed
    before the test leaves each collection to the objects the test created."""
    gc.freeze()
    yield
    gc.unfreeze()


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[_Harness]:
    _RecordingLoader.built_from = []
    models_path = tmp_path / "models"
    models_path.mkdir()
    harness = _Harness(models_path)
    yield harness
    for cache in harness.caches:
        cache.shutdown()


def _sd1_record(**fields) -> Main_Checkpoint_SD1_Config:
    defaults = {
        "key": KEY,
        "path": MODEL_PATH,
        "name": "model",
        "hash": "hash",
        "file_size": 1,
        "source": "local-source",
        "source_type": ModelSourceType.Path,
        "prediction_type": "epsilon",
        "variant": ModelVariantType.Normal,
    }
    return Main_Checkpoint_SD1_Config(**{**defaults, **fields})


def _write_sd_checkpoint(path: Path, base: BaseModelType) -> Path:
    """The few keys SD checkpoint identification reads, at the widths that tell SD-1 from SDXL."""
    cross_attention = {
        BaseModelType.StableDiffusion1: ("model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2", 768),
        BaseModelType.StableDiffusionXL: ("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2", 2048),
    }
    prefix, width = cross_attention[base]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            f"{prefix}.to_k.weight": torch.zeros(1, width),
            "model.diffusion_model.input_blocks.0.0.weight": torch.zeros(1, 4, 1, 1),
        },
        path,
    )
    return path


def _identify(model_file: Path, overrides: Optional[dict[str, object]] = None, /, **record_fields) -> AnyModelConfig:
    """The record an install of `model_file` would have written, with `record_fields` then set on it."""
    config = ModelConfigFactory.from_model_on_disk(
        model_file,
        {"source": "local-source", "source_type": ModelSourceType.Path, **(overrides or {})},
        allow_unknown=False,
    ).config
    assert config is not None
    for field, value in record_fields.items():
        setattr(config, field, value)
    return config


@pytest.mark.parametrize(
    ("changes", "load_affecting"),
    [
        pytest.param(ModelRecordChanges(name="renamed"), False, id="name"),
        pytest.param(ModelRecordChanges(description="described"), False, id="description"),
        pytest.param(ModelRecordChanges(trigger_phrases={"a phrase"}), False, id="trigger_phrases"),
        pytest.param(
            ModelRecordChanges(default_settings=MainModelDefaultSettings(steps=12)), False, id="generation_default"
        ),
        pytest.param(
            ModelRecordChanges(default_settings=MainModelDefaultSettings(fp8_storage=True)), True, id="fp8_storage"
        ),
        pytest.param(ModelRecordChanges(default_settings=MainModelDefaultSettings(cpu_only=True)), True, id="cpu_only"),
        pytest.param(ModelRecordChanges(variant=ModelVariantType.Inpaint), True, id="variant"),
        pytest.param(ModelRecordChanges(base=BaseModelType.StableDiffusionXL), True, id="config_class"),
        pytest.param(ModelRecordChanges(path="sd-1/main/other.safetensors"), True, id="other_file"),
    ],
)
def test_update_drops_cached_instances_only_when_the_change_affects_loading(
    harness: _Harness, changes: ModelRecordChanges, load_affecting: bool
) -> None:
    harness.store.add_model(_sd1_record())
    harness.cache_model()

    updated = harness.manager.update_model_record(KEY, changes)

    assert harness.is_cached() is not load_affecting
    assert harness.store.get_model(KEY) == updated
    for field in changes.model_fields_set:
        assert getattr(updated, field) == getattr(changes, field)


def test_update_rewriting_a_path_to_the_same_file_in_absolute_form_keeps_cached_instances(harness: _Harness) -> None:
    """The load fence resolves paths against the models directory, so a models-relative path and the absolute path
    of the same file load the same model. The edit must compare them the same way, or it evicts a model it did not
    change and rejects the loads that overlapped it."""
    harness.store.add_model(_sd1_record())
    harness.cache_model()

    harness.manager.update_model_record(KEY, ModelRecordChanges(path=str(harness.models_path / MODEL_PATH)))

    assert harness.is_cached()


def test_cached_instances_are_dropped_under_the_model_load_write_lock(harness: _Harness) -> None:
    """Holding the construction lock across the drop keeps a concurrent load from re-registering state it built
    from the previous record."""
    harness.store.add_model(_sd1_record())
    harness.cache_model()
    drops_under_write_lock: list[bool] = []
    for cache in harness.caches:
        cache.on_cache_models_cleared(lambda **_: drops_under_write_lock.append(MODEL_LOAD_LOCK._writer_active))

    harness.manager.update_model_record(
        KEY, ModelRecordChanges(default_settings=MainModelDefaultSettings(cpu_only=True))
    )

    assert drops_under_write_lock == [True, True]


def test_update_that_the_store_rejects_changes_nothing(harness: _Harness) -> None:
    record = _sd1_record()
    harness.store.add_model(record)
    harness.cache_model()

    with pytest.raises(ValueError, match="at most"):
        harness.manager.update_model_record(KEY, ModelRecordChanges(path="x" * 800))

    assert harness.store.get_model(KEY) == record
    assert harness.is_cached()


def test_update_of_an_unknown_model_raises_and_drops_nothing(harness: _Harness) -> None:
    harness.cache_model()

    with pytest.raises(UnknownModelException):
        harness.manager.update_model_record(KEY, ModelRecordChanges(name="renamed"))

    assert harness.is_cached()


@pytest.mark.parametrize("cpu_only", [True, False, None])
@pytest.mark.parametrize("fp8_storage", [True, False, None])
def test_reidentify_refiles_a_misidentified_model_and_keeps_what_the_user_set(
    harness: _Harness, cpu_only: Optional[bool], fp8_storage: Optional[bool]
) -> None:
    """Re-identification replaces what the files decide (here SD-1 becomes SDXL, a load-affecting change) and keeps
    the user's fields. False and None are user choices too: probe defaults must not replace them."""
    _write_sd_checkpoint(harness.models_path / MODEL_PATH, BaseModelType.StableDiffusionXL)
    harness.store.add_model(
        _sd1_record(
            name="My model",
            description="mine",
            cover_image="cover.webp",
            trigger_phrases={"a phrase"},
            default_settings=MainModelDefaultSettings(cpu_only=cpu_only, fp8_storage=fp8_storage, steps=12),
        )
    )
    harness.cache_model()

    updated = harness.manager.reidentify_model(KEY)

    assert isinstance(updated, Main_Checkpoint_SDXL_Config)
    assert harness.store.get_model(KEY) == updated
    assert (updated.key, updated.path, updated.name, updated.description) == (KEY, MODEL_PATH, "My model", "mine")
    assert updated.trigger_phrases == {"a phrase"}
    assert updated.cover_image == "cover.webp"
    assert (updated.source, updated.source_type) == ("local-source", ModelSourceType.Path)
    assert updated.default_settings is not None
    assert (updated.default_settings.cpu_only, updated.default_settings.fp8_storage) == (cpu_only, fp8_storage)
    # Other defaults are the architecture's: only the load preferences are the user's to keep.
    assert updated.default_settings.width == 1024
    assert not harness.is_cached()
    # The probe hands out an architecture's shared defaults; the kept preferences must not leak into them.
    later_install = _identify(harness.models_path / MODEL_PATH)
    assert later_install.default_settings is not None
    assert (later_install.default_settings.cpu_only, later_install.default_settings.fp8_storage) == (None, None)


def test_reidentify_while_a_load_of_the_old_record_is_queued_rebuilds_it_from_the_new_record(
    harness: _Harness,
) -> None:
    """A load that checked the SD-1 record and queued for the construction lock while re-identification files the
    model as SDXL must not build from the record it read: re-identification is announced to loads from before its
    commit until after its cache drop."""
    _write_sd_checkpoint(harness.models_path / MODEL_PATH, BaseModelType.StableDiffusionXL)
    harness.store.add_model(_sd1_record())
    stale = harness.store.get_model(KEY)
    loads: list[object] = []
    errors: list[BaseException] = []

    def load() -> None:
        try:
            loads.append(harness.manager.load.load_model(stale))
        except BaseException as e:
            loads.append(e)

    def reidentify() -> None:
        try:
            harness.manager.reidentify_model(KEY)
        except BaseException as e:
            errors.append(e)

    # Daemon threads: a regression that deadlocks must fail the test, not hang it.
    loader = threading.Thread(target=load, daemon=True)
    editor = threading.Thread(target=reidentify, daemon=True)
    with MODEL_LOAD_LOCK.write_lock():  # an unrelated construction in progress
        loader.start()
        deadline = time.monotonic() + 10
        while MODEL_LOAD_LOCK._writers_waiting == 0:
            assert time.monotonic() < deadline, "loader never queued for the construction lock"
            time.sleep(0.001)
        editor.start()
        # Until re-identification has committed and waits to drop the model, or has ended without dropping it.
        while MODEL_LOAD_LOCK._writers_waiting < 2 and editor.is_alive():
            assert time.monotonic() < deadline + 10, "re-identification neither finished nor reached its drop"
            time.sleep(0.001)
    editor.join(timeout=10)
    loader.join(timeout=10)

    assert not editor.is_alive() and not errors
    assert not loader.is_alive() and len(loads) == 1
    assert isinstance(loads[0], LoadedModel) and isinstance(loads[0].config, Main_Checkpoint_SDXL_Config)
    assert _RecordingLoader.built_from == ["Main_Checkpoint_SDXL_Config"]


def test_reidentify_that_changes_nothing_that_loads_keeps_cached_instances(harness: _Harness) -> None:
    path = _write_sd_checkpoint(harness.models_path / MODEL_PATH, BaseModelType.StableDiffusion1)
    installed = _identify(path, key=KEY, path=MODEL_PATH, name="My model")
    harness.store.add_model(installed)
    harness.cache_model()

    updated = harness.manager.reidentify_model(KEY)

    assert updated == installed
    assert harness.is_cached()


@pytest.mark.parametrize("cpu_only", [True, False])
def test_reidentify_keeps_a_cpu_only_choice_held_on_the_config_itself(harness: _Harness, cpu_only: bool) -> None:
    """Text encoders keep `cpu_only` beside their other fields rather than in default settings."""
    path = write_header_only_safetensors(harness.models_path / "clip-l.safetensors", clip_l_comfy_keys.state_dict_keys)
    installed = _identify(path, key=KEY, path="clip-l.safetensors", name="CLIP-L")
    assert isinstance(installed, CLIPEmbed_Checkpoint_L_Config)
    installed.cpu_only = cpu_only
    harness.store.add_model(installed)

    updated = harness.manager.reidentify_model(KEY)

    assert isinstance(updated, CLIPEmbed_Checkpoint_L_Config)
    assert updated.cpu_only is cpu_only


def test_reidentify_keeps_an_ip_adapter_encoder_override_its_files_leave_blank(harness: _Harness) -> None:
    model_path = harness.models_path / "ip-adapter"
    model_path.mkdir()
    torch.save({"ip_adapter": {"1.to_k_ip.weight": torch.zeros(1, 768)}}, model_path / "ip_adapter.bin")
    (model_path / "image_encoder.txt").write_text("  \n", encoding="utf-8")
    installed = _identify(
        model_path, {"image_encoder_model_id": "custom/image-encoder"}, key=KEY, path="ip-adapter", name="IP-Adapter"
    )
    assert isinstance(installed, IPAdapter_InvokeAI_SD1_Config)
    harness.store.add_model(installed)

    updated = harness.manager.reidentify_model(KEY)

    assert isinstance(updated, IPAdapter_InvokeAI_SD1_Config)
    assert updated.image_encoder_model_id == "custom/image-encoder"


def test_reidentify_records_the_scale_of_an_upscaler_installed_before_it_was_recorded(harness: _Harness) -> None:
    """The recorded scale is part of the record, so the rewrite counts as load-affecting: a cached instance built
    from the scale-less record is dropped."""
    path = write_compact_upscaler(harness.models_path / "4x-compact.safetensors", 4)
    installed = _identify(path, key=KEY, path="4x-compact.safetensors", scale=None)
    assert isinstance(installed, Spandrel_Checkpoint_Config)
    harness.store.add_model(installed)
    harness.cache_model()

    updated = harness.manager.reidentify_model(KEY)

    assert isinstance(updated, Spandrel_Checkpoint_Config) and updated.scale == 4
    assert harness.store.get_model(KEY) == updated
    assert not harness.is_cached()


def test_reidentify_keeps_a_backbone_only_the_install_source_names(harness: _Harness) -> None:
    """A 16-channel VAE is FLUX.1's or SD3's by its name alone. Copied into the models folder it keeps a generic
    file name in a folder named for its key, and only the Hugging Face source still says SD3."""
    path = harness.models_path / "7c1f9d2e" / "diffusion_pytorch_model.safetensors"
    path.parent.mkdir()
    save_file(
        {"encoder.conv_in.weight": torch.zeros(8, 3, 3, 3), "decoder.conv_in.weight": torch.zeros(8, 16, 3, 3)}, path
    )
    installed = ModelConfigFactory.from_model_on_disk(
        path,
        {
            "key": KEY,
            "source": "stabilityai/stable-diffusion-3.5-large::vae/diffusion_pytorch_model.safetensors",
            "source_type": ModelSourceType.HFRepoID,
        },
        allow_unknown=False,
    ).config
    assert installed is not None and installed.base is BaseModelType.StableDiffusion3
    harness.store.add_model(installed)

    assert harness.manager.reidentify_model(KEY).base is BaseModelType.StableDiffusion3


def test_reidentify_of_files_no_config_accepts_raises_and_keeps_the_record(harness: _Harness) -> None:
    keys = dict(clip_g_comfy_keys.state_dict_keys)
    del keys["text_projection.weight"]
    write_header_only_safetensors(harness.models_path / "clip-g.safetensors", keys)
    record = _sd1_record(path="clip-g.safetensors")
    harness.store.add_model(record)
    harness.cache_model()

    with pytest.raises(InvalidModelException):
        harness.manager.reidentify_model(KEY)

    assert harness.store.get_model(KEY) == record
    assert harness.is_cached()


def test_reidentify_of_an_unknown_model_raises(harness: _Harness) -> None:
    with pytest.raises(UnknownModelException):
        harness.manager.reidentify_model(KEY)
