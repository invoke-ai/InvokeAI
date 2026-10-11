"""A load that read a model record before a load-affecting edit must not repopulate the cache after
that edit's invalidation; later loads must build from the updated record."""

import contextlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, ContextManager, Iterator, Optional

import pytest
import torch

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_load.model_load_default import ModelLoadService
from invokeai.app.services.model_records import ModelRecordChanges, ModelRecordServiceSQL
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.siglip import SigLIP_Diffusers_Config
from invokeai.backend.model_manager.load import (
    LoadedModel,
    ModelCache,
    ModelLoader,
    ModelLoaderBase,
    StaleModelConfigError,
)
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


def _long_transaction(store: ModelRecordServiceSQL) -> ContextManager[object]:
    """A write transaction held open, as VACUUM holds the database for its whole run; on SQLite it blocks every
    other transaction, reads included."""
    return store._queries._database.begin(write=True)


def _is_cached(cache: ModelCache) -> bool:
    try:
        cache.get(KEY)
    except IndexError:
        return False
    return True


@pytest.fixture
def record_reads_under_lock(harness, monkeypatch) -> list[bool]:
    """For each read of the model record by a worker thread (the test's own edits run on the main thread),
    whether the reading thread held MODEL_LOAD_LOCK's write lock."""
    _, store, _, _ = harness
    holders: set[int] = set()
    write_lock = MODEL_LOAD_LOCK.write_lock

    @contextlib.contextmanager
    def tracked_write_lock() -> Iterator[None]:
        with write_lock():
            holders.add(threading.get_ident())
            try:
                yield
            finally:
                holders.discard(threading.get_ident())

    reads: list[bool] = []
    store_get_model = store.get_model

    def get_model(key: str) -> AnyModelConfig:
        if threading.current_thread() is not threading.main_thread():
            reads.append(threading.get_ident() in holders)
        return store_get_model(key)

    monkeypatch.setattr(MODEL_LOAD_LOCK, "write_lock", tracked_write_lock)
    monkeypatch.setattr(store, "get_model", get_model)
    return reads


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


def test_metadata_only_edit_while_a_load_waits_on_construction_lock_does_not_reject_it(harness, monkeypatch):
    """A rename changes nothing that loads, so a load that read the old name and is queued behind an
    unrelated construction while the rename lands is built as it is, without a retry."""
    service, store, cache, events = harness
    waits = _waits_for_edits(service, monkeypatch)
    outcome: list[LoadedModel] = []
    loader = threading.Thread(target=lambda: outcome.append(service.load_model(store.get_model(KEY))))

    with MODEL_LOAD_LOCK.write_lock():  # an unrelated construction in progress
        loader.start()
        deadline = time.monotonic() + 10
        while MODEL_LOAD_LOCK._writers_waiting == 0:
            assert time.monotonic() < deadline, "loader never queued for the construction lock"
            time.sleep(0.001)
        # As `update_model_record` performs a rename: bracketed, and cleared as changing nothing that loads.
        with service.record_edit(KEY) as edit:
            store.update_model(KEY, changes=ModelRecordChanges(name="renamed"))
            edit.load_affecting = False
    loader.join(timeout=10)

    assert len(outcome) == 1
    assert _RecordingLoader.built_from == [None]
    assert _is_cached(cache)
    # Checked once, never retried.
    assert waits.acquire(timeout=0) and not waits.acquire(timeout=0)


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


def test_load_waiting_on_construction_lock_during_invalidating_edit_loads_the_updated_record(
    harness, record_reads_under_lock
):
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
    # Rejected without reading the record under the lock: that read would stall every MODEL_LOAD_LOCK user
    # behind any long DB transaction.
    assert record_reads_under_lock and not any(record_reads_under_lock)


def _waits_for_edits(service: ModelLoadService, monkeypatch) -> threading.Semaphore:
    """Released each time a load starts waiting for edits of its model to finish."""
    entered = threading.Semaphore(0)
    record_edits = service._record_edits
    watch, wait_idle = record_edits.watch, record_edits.wait_idle

    def observed_watch(key: str):
        entered.release()
        return watch(key)

    def observed_wait_idle(key: str) -> None:
        entered.release()
        wait_idle(key)

    monkeypatch.setattr(record_edits, "watch", observed_watch)
    monkeypatch.setattr(record_edits, "wait_idle", observed_wait_idle)
    return entered


@pytest.mark.parametrize("commit_before_rejection", [True, False])
def test_edit_starting_while_load_waits_but_invalidating_after_it_is_not_built_from_the_old_record(
    harness, monkeypatch, record_reads_under_lock, commit_before_rejection: bool
):
    """The load checks a current record and queues behind an unrelated construction; an edit then starts,
    and reaches its invalidation only after the load holds the construction lock. Its commit lands either
    before the load is rejected or while the rejected load waits to retry."""
    service, store, cache, _ = harness
    waits = _waits_for_edits(service, monkeypatch)
    outcome: list[LoadedModel] = []
    loader = threading.Thread(target=lambda: outcome.append(service.load_model(store.get_model(KEY))))

    with MODEL_LOAD_LOCK.write_lock():  # an unrelated construction in progress
        loader.start()
        deadline = time.monotonic() + 10
        while MODEL_LOAD_LOCK._writers_waiting == 0:
            assert time.monotonic() < deadline, "loader never queued for the construction lock"
            time.sleep(0.001)
        edit = service.record_edit(KEY)
        edit.__enter__()
        if commit_before_rejection:
            store.update_model(KEY, changes=ModelRecordChanges(cpu_only=True))
    try:
        # Rejected under the lock, the load waits for the edit before reading the record again.
        assert waits.acquire(timeout=10) and waits.acquire(timeout=10)
        if not commit_before_rejection:
            store.update_model(KEY, changes=ModelRecordChanges(cpu_only=True))
        with MODEL_LOAD_LOCK.write_lock():
            cache.drop_model(KEY)
    finally:
        edit.__exit__(None, None, None)
    loader.join(timeout=10)

    assert len(outcome) == 1
    assert outcome[0].config.cpu_only is True
    assert _RecordingLoader.built_from == [True]
    assert not any(record_reads_under_lock)


def test_load_overlapping_an_edit_stuck_on_the_database_does_not_hold_the_model_load_lock(
    harness, monkeypatch, record_reads_under_lock
):
    """A cold load of a model whose edit is waiting out a long DB transaction (e.g. VACUUM) waits too, but
    without MODEL_LOAD_LOCK, so unrelated model builds and VRAM moves carry on; it then loads the edit."""
    service, store, cache, events = harness
    waits = _waits_for_edits(service, monkeypatch)
    stale = store.get_model(KEY)
    committing = threading.Event()

    def edit() -> None:
        with service.record_edit(KEY):
            committing.set()
            store.update_model(KEY, changes=ModelRecordChanges(cpu_only=True))
            with MODEL_LOAD_LOCK.write_lock():
                cache.drop_model(KEY)

    loads: list[LoadedModel] = []
    acquired = threading.Event()

    def take_model_load_lock() -> None:
        with MODEL_LOAD_LOCK.write_lock():
            acquired.set()

    editor = threading.Thread(target=edit)
    loader = threading.Thread(target=lambda: loads.append(service.load_model(stale)))
    other = threading.Thread(target=take_model_load_lock)
    with _long_transaction(store):  # the transaction the edit's commit waits on
        editor.start()
        assert committing.wait(timeout=10)
        loader.start()
        assert waits.acquire(timeout=10)
        other.start()
        lock_was_free = acquired.wait(timeout=5)
    editor.join(timeout=10)
    loader.join(timeout=10)
    other.join(timeout=10)

    assert lock_was_free
    assert len(loads) == 1 and loads[0].config.cpu_only is True
    assert _RecordingLoader.built_from == [True]
    assert events.log == [("started", "siglip"), ("complete", "siglip")]
    assert not any(record_reads_under_lock)


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
    service, store, _, events = harness
    fence_checks: list[str] = []

    def never_current(config: AnyModelConfig) -> bool:
        fence_checks.append(config.key)
        return False

    monkeypatch.setattr(service, "_config_is_current", never_current)

    with pytest.raises(StaleModelConfigError):
        service.load_model(store.get_model(KEY))
    assert fence_checks == [KEY, KEY]
    assert _RecordingLoader.built_from == []
    # The load ended, so it must not be left showing as in progress.
    assert events.log == [("started", "siglip"), ("complete", "siglip")]


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
        transaction = _long_transaction(store)
        transaction.__enter__()
    try:
        # The loader now holds the construction lock: it either finishes without the database, or reads
        # its record again under the lock and blocks there.
        deadline = time.monotonic() + 10
        while loader.is_alive() and not record_reads.acquire(timeout=0.01):
            assert time.monotonic() < deadline, "loader neither finished nor read its record again"
        other.start()
        lock_was_free = acquired.wait(timeout=5)
    finally:
        transaction.__exit__(None, None, None)
    loader.join(timeout=10)
    other.join(timeout=10)

    assert lock_was_free
    assert len(loads) == 1
    assert _is_cached(cache)


def test_load_from_record_read_before_the_model_was_moved_loads_from_its_new_path(harness, tmp_path):
    service, store, cache, events = harness
    stale = store.get_model(KEY)
    (tmp_path / "siglip").rename(tmp_path / "moved")
    _edit_and_invalidate(service, store, cache, ModelRecordChanges(path="moved"))

    loaded = service.load_model(stale)

    assert Path(loaded.config.path) == tmp_path / "moved"
    assert events.log == [("started", "siglip"), ("complete", "siglip")]


def test_missing_files_under_an_unchanged_record_still_fail(harness, tmp_path):
    service, store, _, events = harness
    (tmp_path / "siglip").rmdir()

    with pytest.raises(FileNotFoundError):
        service.load_model(store.get_model(KEY))
    assert _RecordingLoader.built_from == []
    assert events.log == [("started", "siglip"), ("complete", "siglip")]


def test_edit_tracking_is_retired_once_no_edit_or_load_needs_it(harness):
    """Edited and deleted keys must not accumulate: a key is tracked only while an edit of it is in progress
    or a load is between its record read and its check under the construction lock."""
    service, store, _, _ = harness
    for i in range(100):
        with service.record_edit(f"deleted-{i}"):
            pass
    service.load_model(store.get_model(KEY))
    with service.record_edit(KEY):
        pass

    assert service._record_edits._keys == {}


def test_rename_then_load_affecting_edit_while_a_load_waits_rejects_it(harness, monkeypatch):
    """A load stays tracked between its record read and its check: the rename ending must not retire what the
    load watches, or the load-affecting edit that follows would go unnoticed."""
    service, store, cache, _ = harness
    waits = _waits_for_edits(service, monkeypatch)
    outcome: list[LoadedModel] = []
    loader = threading.Thread(target=lambda: outcome.append(service.load_model(store.get_model(KEY))))

    with MODEL_LOAD_LOCK.write_lock():  # an unrelated construction in progress
        loader.start()
        deadline = time.monotonic() + 10
        while MODEL_LOAD_LOCK._writers_waiting == 0:
            assert time.monotonic() < deadline, "loader never queued for the construction lock"
            time.sleep(0.001)
        with service.record_edit(KEY) as rename:
            store.update_model(KEY, changes=ModelRecordChanges(name="renamed"))
            rename.load_affecting = False
        edit = service.record_edit(KEY)
        edit.__enter__()
        store.update_model(KEY, changes=ModelRecordChanges(cpu_only=True))
    try:
        # Rejected under the lock, the load waits for the edit before reading the record again.
        assert waits.acquire(timeout=10) and waits.acquire(timeout=10)
        with MODEL_LOAD_LOCK.write_lock():
            cache.drop_model(KEY)
    finally:
        edit.__exit__(None, None, None)
    loader.join(timeout=10)

    assert len(outcome) == 1 and outcome[0].config.cpu_only is True
    assert _RecordingLoader.built_from == [True]


class _UnfencedLoader(ModelLoaderBase):
    """Built directly on the base, so nothing checks its config against record edits before it caches."""

    def __init__(self, app_config, logger, ram_cache: ModelCache) -> None:
        self._ram_cache = ram_cache

    @property
    def ram_cache(self) -> ModelCache:
        return self._ram_cache

    def get_size_fs(self, config, model_path, submodel_type=None) -> int:
        return 0

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        self._ram_cache.put(model_config.key, torch.nn.Linear(2, 2))
        return LoadedModel(
            config=model_config, cache_record=self._ram_cache.get(model_config.key), cache=self._ram_cache
        )


def test_loader_built_directly_on_the_base_is_refused_before_it_can_cache(harness, monkeypatch):
    """A registry is free to hand the service any class, so the service refuses one that skips the check."""
    service, store, cache, events = harness

    class _UnfencedRegistry:
        @classmethod
        def get_implementation(cls, config: AnyModelConfig, submodel_type: Optional[SubModelType]):
            return _UnfencedLoader, config, submodel_type

    monkeypatch.setattr(service, "_registry", _UnfencedRegistry)

    with pytest.raises(TypeError, match="not a ModelLoader"):
        service.load_model(store.get_model(KEY))
    assert not _is_cached(cache)
    assert events.log == [("started", "siglip"), ("complete", "siglip")]
