"""A load that read a model record before a load-affecting edit must not repopulate the cache after
that edit's invalidation; later loads must build from the updated record."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_load.model_load_default import ModelLoadService
from invokeai.app.services.model_records import ModelRecordChanges, ModelRecordServiceSQL
from invokeai.app.services.shared.invocation_context import ModelsInterface
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.siglip import SigLIP_Diffusers_Config
from invokeai.backend.model_manager.load import LoadedModel, ModelCache, ModelLoader, StaleModelConfigError
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK
from invokeai.backend.model_manager.taxonomy import AnyModel, SubModelType
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database

KEY = "siglip-key"


class _RecordingLoader(ModelLoader):
    """Builds a tiny module and records the `cpu_only` value each construction was built from."""

    built_from: list[Optional[bool]] = []

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        self.built_from.append(getattr(config, "cpu_only", None))
        return torch.nn.Linear(2, 2)


class _Registry:
    @classmethod
    def get_implementation(cls, config: AnyModelConfig, submodel_type: Optional[SubModelType]):
        return _RecordingLoader, config, submodel_type


def _noop(*args, **kwargs) -> None:
    return None


@pytest.fixture
def harness(tmp_path: Path):
    _RecordingLoader.built_from = []
    app_config = InvokeAIAppConfig(use_memory_db=True)
    logger = InvokeAILogger.get_logger()
    store = ModelRecordServiceSQL(create_mock_sqlite_database(app_config, logger), logger)
    model_dir = tmp_path / "siglip"
    model_dir.mkdir()
    store.add_model(
        SigLIP_Diffusers_Config(
            key=KEY, path=str(model_dir), name="siglip", hash="abc", file_size=1, source="test", source_type="path"
        )
    )
    cache = ModelCache(
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
    service = ModelLoadService(app_config=app_config, ram_cache=cache, registry=_Registry)  # type: ignore[arg-type]
    events = SimpleNamespace(emit_model_load_started=_noop, emit_model_load_complete=_noop)
    service.start(SimpleNamespace(services=SimpleNamespace(events=events, model_manager=SimpleNamespace(store=store))))  # type: ignore[arg-type]
    return service, store, cache


def _edit_and_invalidate(store: ModelRecordServiceSQL, cache: ModelCache, changes: ModelRecordChanges) -> None:
    """The record edit followed by the cache drop, as `update_model_record` performs them."""
    store.update_model(KEY, changes=changes, allow_class_change=True)
    with MODEL_LOAD_LOCK.write_lock():
        cache.drop_model(KEY)


def _is_cached(cache: ModelCache) -> bool:
    try:
        cache.get(KEY)
    except IndexError:
        return False
    return True


def test_load_from_record_read_before_invalidating_edit_is_rejected_and_not_cached(harness):
    service, store, cache = harness
    stale = store.get_model(KEY)

    _edit_and_invalidate(store, cache, ModelRecordChanges(cpu_only=True))

    with pytest.raises(StaleModelConfigError):
        service.load_model(stale)
    assert not _is_cached(cache)
    assert _RecordingLoader.built_from == []

    loaded = service.load_model(store.get_model(KEY))
    assert loaded.config.cpu_only is True
    assert _RecordingLoader.built_from == [True]
    assert _is_cached(cache)


def test_load_from_record_read_before_metadata_only_edit_still_loads(harness):
    service, store, cache = harness
    previous = store.get_model(KEY)

    # A rename does not invalidate the cache, so nothing fences a load that read the old name.
    store.update_model(KEY, changes=ModelRecordChanges(name="renamed"))

    service.load_model(previous)
    assert _RecordingLoader.built_from == [None]
    assert _is_cached(cache)


def _run_with_edit_while_queued_on_construction_lock(
    store: ModelRecordServiceSQL, cache: ModelCache, load: Callable[[], object]
) -> object:
    """The #9674 ordering: `load` reads a current record, then waits for the construction lock while
    a `cpu_only` edit commits and invalidates under it. Returns what `load` returned or raised."""
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(load())
        except BaseException as e:
            outcome.append(e)

    loader = threading.Thread(target=run)
    with MODEL_LOAD_LOCK.write_lock():
        loader.start()
        deadline = time.monotonic() + 10
        while MODEL_LOAD_LOCK._writers_waiting == 0:
            assert time.monotonic() < deadline, "loader never queued for the construction lock"
            time.sleep(0.001)
        store.update_model(KEY, changes=ModelRecordChanges(cpu_only=True))
        cache.drop_model(KEY)
    loader.join(timeout=10)
    assert len(outcome) == 1
    return outcome[0]


def test_load_waiting_on_construction_lock_during_invalidating_edit_is_rejected(harness):
    service, store, cache = harness

    outcome = _run_with_edit_while_queued_on_construction_lock(
        store, cache, lambda: service.load_model(store.get_model(KEY))
    )

    assert isinstance(outcome, StaleModelConfigError)
    assert _RecordingLoader.built_from == []
    assert not _is_cached(cache)


def test_invocation_load_superseded_mid_load_loads_the_updated_record(harness):
    service, store, cache = harness
    services = SimpleNamespace(model_manager=SimpleNamespace(store=store, load=service))
    models = ModelsInterface(services=services, data=MagicMock(), util=MagicMock())  # type: ignore[arg-type]

    outcome = _run_with_edit_while_queued_on_construction_lock(store, cache, lambda: models.load(KEY))

    assert isinstance(outcome, LoadedModel)
    assert outcome.config.cpu_only is True
    assert _RecordingLoader.built_from == [True]


def test_config_object_reused_after_eviction_still_loads(harness):
    """The loader rewrites `config.path` to an absolute path in place; a service that keeps its config
    object across loads must not see that as a load-affecting edit."""
    service, store, cache = harness
    config = store.get_model(KEY)
    # Use it once so its first-use hold is released and the drop below evicts it outright.
    with service.load_model(config):
        pass
    with MODEL_LOAD_LOCK.write_lock():
        cache.drop_model(KEY)

    service.load_model(config)
    assert _RecordingLoader.built_from == [None, None]
