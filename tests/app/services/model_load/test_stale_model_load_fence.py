"""A load that read a model record before a load-affecting edit must not repopulate the cache after
that edit's invalidation; later loads must build from the updated record."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import pytest
import torch

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_load.model_load_default import ModelLoadService
from invokeai.app.services.model_records import ModelRecordChanges, ModelRecordServiceSQL
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


class _LoadEvents:
    """Records the model load events in order, as (event, model name)."""

    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []

    def emit_model_load_started(self, config: AnyModelConfig, submodel_type, user_id: str) -> None:
        self.log.append(("started", config.name))

    def emit_model_load_complete(self, config: AnyModelConfig, submodel_type, user_id: str) -> None:
        self.log.append(("complete", config.name))


@pytest.fixture
def harness(tmp_path: Path):
    _RecordingLoader.built_from = []
    app_config = InvokeAIAppConfig(use_memory_db=True, models_dir=tmp_path)
    logger = InvokeAILogger.get_logger()
    store = ModelRecordServiceSQL(create_mock_sqlite_database(app_config, logger), logger)
    model_dir = tmp_path / "siglip"
    model_dir.mkdir()
    store.add_model(
        SigLIP_Diffusers_Config(
            key=KEY, path="siglip", name="siglip", hash="abc", file_size=1, source="test", source_type="path"
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
    events = _LoadEvents()
    service.start(SimpleNamespace(services=SimpleNamespace(events=events, model_manager=SimpleNamespace(store=store))))  # type: ignore[arg-type]
    return service, store, cache, events


def _edit_and_invalidate(
    service: ModelLoadService, store: ModelRecordServiceSQL, cache: ModelCache, changes: ModelRecordChanges
) -> None:
    """The record edit followed by the cache drop, as `update_model_record` performs them."""
    with service.record_edit(KEY):
        store.update_model(KEY, changes=changes, allow_class_change=True)
        with MODEL_LOAD_LOCK.write_lock():
            cache.drop_model(KEY)


def _is_cached(cache: ModelCache) -> bool:
    try:
        cache.get(KEY)
    except IndexError:
        return False
    return True


def test_load_from_record_read_before_invalidating_edit_loads_the_updated_record(harness):
    service, store, cache, events = harness
    stale = store.get_model(KEY)

    _edit_and_invalidate(service, store, cache, ModelRecordChanges(cpu_only=True))

    loaded = service.load_model(stale)
    assert loaded.config.cpu_only is True
    assert _RecordingLoader.built_from == [True]
    assert _is_cached(cache)
    # One load as far as the UI can tell: a retry that re-announced itself would leave a load showing.
    assert events.log == [("started", "siglip"), ("complete", "siglip")]


def test_load_from_record_read_before_metadata_only_edit_still_loads(harness):
    service, store, cache, _ = harness
    previous = store.get_model(KEY)

    # A rename does not invalidate the cache, so nothing fences a load that read the old name.
    store.update_model(KEY, changes=ModelRecordChanges(name="renamed"))

    service.load_model(previous)
    assert _RecordingLoader.built_from == [None]
    assert _is_cached(cache)


def _run_with_edit_while_queued_on_construction_lock(
    service: ModelLoadService, store: ModelRecordServiceSQL, cache: ModelCache, load: Callable[[], object]
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
        with service.record_edit(KEY):
            store.update_model(KEY, changes=ModelRecordChanges(cpu_only=True))
            cache.drop_model(KEY)
    loader.join(timeout=10)
    assert len(outcome) == 1
    return outcome[0]


def test_load_waiting_on_construction_lock_during_invalidating_edit_loads_the_updated_record(harness):
    service, store, cache, events = harness

    outcome = _run_with_edit_while_queued_on_construction_lock(
        service, store, cache, lambda: service.load_model(store.get_model(KEY))
    )

    assert isinstance(outcome, LoadedModel)
    assert outcome.config.cpu_only is True
    # Built once, from the updated record: the stale construction never ran.
    assert _RecordingLoader.built_from == [True]
    # One load as far as the UI can tell: a retry that re-announced itself would leave a load showing.
    assert events.log == [("started", "siglip"), ("complete", "siglip")]


def test_edit_committed_while_load_waits_but_invalidating_after_it_is_not_built_from_the_old_record(harness):
    """The load reads a current record and queues behind an unrelated construction; the edit commits
    meanwhile but reaches its invalidation only after the load holds the construction lock."""
    service, store, cache, _ = harness
    outcome: list[LoadedModel] = []
    loader = threading.Thread(target=lambda: outcome.append(service.load_model(store.get_model(KEY))))

    with service.record_edit(KEY):
        with MODEL_LOAD_LOCK.write_lock():  # an unrelated construction in progress
            loader.start()
            deadline = time.monotonic() + 10
            while MODEL_LOAD_LOCK._writers_waiting == 0:
                assert time.monotonic() < deadline, "loader never queued for the construction lock"
                time.sleep(0.001)
            store.update_model(KEY, changes=ModelRecordChanges(cpu_only=True))
        loader.join(timeout=10)
        with MODEL_LOAD_LOCK.write_lock():
            cache.drop_model(KEY)

    assert len(outcome) == 1
    assert outcome[0].config.cpu_only is True
    assert _RecordingLoader.built_from == [True]


def test_config_object_reused_after_eviction_still_loads(harness):
    """The loader rewrites `config.path` to an absolute path in place; a service that keeps its config
    object across loads must not see that as a load-affecting edit."""
    service, store, cache, _ = harness
    config = store.get_model(KEY)
    # Use it once so its first-use hold is released and the drop below evicts it outright.
    with service.load_model(config):
        pass
    with MODEL_LOAD_LOCK.write_lock():
        cache.drop_model(KEY)

    # Checked directly: a spurious rejection would otherwise be hidden by the retry.
    assert service._config_is_current(config)
    service.load_model(config)
    assert _RecordingLoader.built_from == [None, None]


def test_record_superseded_again_during_retry_raises(harness, monkeypatch):
    service, store, _, _ = harness
    fence_checks: list[str] = []

    def never_current(config: AnyModelConfig) -> bool:
        fence_checks.append(config.key)
        return False

    monkeypatch.setattr(service, "_config_is_current", never_current)

    with pytest.raises(StaleModelConfigError):
        service.load_model(store.get_model(KEY))
    assert fence_checks == [KEY, KEY]
    assert _RecordingLoader.built_from == []


def test_database_transaction_starting_while_a_cold_load_is_queued_does_not_hold_the_model_load_lock(
    harness, monkeypatch
):
    """A long DB transaction (e.g. VACUUM) that starts after a cold load read its record, while that load
    waits for the construction lock, must not leave the load holding MODEL_LOAD_LOCK while it waits on the
    database: unrelated MODEL_LOAD_LOCK users, such as other devices' VRAM moves, would stall with it."""
    service, store, cache, _ = harness
    config = store.get_model(KEY)
    record_reads = threading.Semaphore(0)
    store_get_model = store.get_model

    def get_model(key: str) -> AnyModelConfig:
        record_reads.release()
        return store_get_model(key)

    monkeypatch.setattr(store, "get_model", get_model)
    loads: list[LoadedModel] = []
    acquired = threading.Event()

    def take_model_load_lock() -> None:
        with MODEL_LOAD_LOCK.write_lock():
            acquired.set()

    loader = threading.Thread(target=lambda: loads.append(service.load_model(config)))
    other = threading.Thread(target=take_model_load_lock)
    with MODEL_LOAD_LOCK.write_lock():  # an unrelated construction in progress
        loader.start()
        assert record_reads.acquire(timeout=10)
        deadline = time.monotonic() + 10
        while MODEL_LOAD_LOCK._writers_waiting == 0:
            assert time.monotonic() < deadline, "loader never queued for the construction lock"
            time.sleep(0.001)
        store._db._lock.acquire()  # the transaction starts; VACUUM holds this for its whole run
    try:
        # The loader now holds the construction lock: it either finishes without the database, or reads
        # its record again under the lock and blocks there.
        deadline = time.monotonic() + 10
        while loader.is_alive() and not record_reads.acquire(timeout=0.01):
            assert time.monotonic() < deadline, "loader neither finished nor read its record again"
        other.start()
        lock_was_free = acquired.wait(timeout=5)
    finally:
        store._db._lock.release()
    loader.join(timeout=10)
    other.join(timeout=10)

    assert lock_was_free
    assert len(loads) == 1
    assert _is_cached(cache)
