import asyncio
import threading
from logging import getLogger
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.api import dependencies
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_index.image_index_default import ImageIndexService
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_processor import session_processor_default
from invokeai.app.services.shared.database.database import Database
from invokeai.backend.util.device_pool import GENERATION_DEVICE_POOL
from invokeai.backend.util.devices import TorchDevice


def _skip_process_wide_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The architecture registry, the attention warm-up, the JWT secret and the model manager: process-wide or heavy,
    and not what these tests check."""
    monkeypatch.setattr(dependencies, "validate_architectures", lambda: None)
    monkeypatch.setattr(dependencies, "warm_up_attention", lambda *_: None)
    monkeypatch.setattr(dependencies, "set_jwt_secret", lambda _: None)
    monkeypatch.setattr(
        dependencies.ModelManagerService,
        "build_model_manager",
        MagicMock(return_value=MagicMock()),
    )


def test_initialize_does_not_vacuum_database(monkeypatch, tmp_path) -> None:
    _skip_process_wide_setup(monkeypatch)

    original_init_database = dependencies.init_database
    initialized_databases: list[Database] = []

    def initialize_database(*args, **kwargs) -> Database:
        database = original_init_database(*args, **kwargs)
        initialized_databases.append(database)
        return database

    monkeypatch.setattr(dependencies, "init_database", initialize_database)
    clean_calls: list[Database] = []
    monkeypatch.setattr(Database, "clean", lambda database: clean_calls.append(database))

    config = InvokeAIAppConfig(
        use_memory_db=True,
        image_index_enabled=False,
        node_cache_size=0,
        scan_models_on_startup=False,
    )
    config._root = tmp_path
    loop = asyncio.new_event_loop()
    original_invoker = getattr(ApiDependencies, "invoker", None)

    try:
        ApiDependencies.initialize(config=config, event_handler_id=0, loop=loop, logger=getLogger("test"))

        assert clean_calls == []
        assert len(initialized_databases) == 1
        assert ApiDependencies.invoker.services.database is initialized_databases[0]
    finally:
        invoker = getattr(ApiDependencies, "invoker", None)
        if invoker is not None and invoker is not original_invoker:
            invoker.stop()
            if original_invoker is None:
                delattr(ApiDependencies, "invoker")
            else:
                ApiDependencies.invoker = original_invoker
        for database in initialized_databases:
            database.dispose()
        ApiDependencies.database = None
        loop.close()


def test_the_generation_devices_are_registered_before_any_service_starts(monkeypatch, tmp_path) -> None:
    """Work outside the session queue borrows an idle GPU from the moment its service starts -- the image index's first
    embedding batch runs before the session processor, which starts last -- and keeps it across the processor's start."""
    _skip_process_wide_setup(monkeypatch)
    # Two GPUs, on a machine that may have none: a worker pins its GPU in CUDA only once it claims a queue item, and
    # the empty queue gives it none.
    gpus = [torch.device("cuda:0"), torch.device("cuda:1")]
    monkeypatch.setattr(TorchDevice, "get_generation_devices", classmethod(lambda cls, generation_devices: list(gpus)))
    # Process-wide torch state, which the processor sets for more than one CUDA worker.
    monkeypatch.setattr(session_processor_default, "disable_conv_benchmark_empty_cache", lambda: None)
    borrowed: list[Optional[torch.device]] = []
    image_index_start = ImageIndexService.start

    def start_and_borrow(self: ImageIndexService, invoker: Invoker) -> None:
        image_index_start(self, invoker)
        borrowed.append(GENERATION_DEVICE_POOL.try_borrow_off_queue("cuda"))

    monkeypatch.setattr(ImageIndexService, "start", start_and_borrow)
    config = InvokeAIAppConfig(
        use_memory_db=True,
        image_index_enabled=False,
        node_cache_size=0,
        scan_models_on_startup=False,
    )
    config._root = tmp_path
    loop = asyncio.new_event_loop()
    original_invoker = getattr(ApiDependencies, "invoker", None)

    try:
        ApiDependencies.initialize(config=config, event_handler_id=0, loop=loop, logger=getLogger("test"))

        assert borrowed == [gpus[0]]
        assert GENERATION_DEVICE_POOL.is_lent(gpus[0]), "the processor's start kept the borrow"
        GENERATION_DEVICE_POOL.release_borrow(gpus[0])
    finally:
        invoker = getattr(ApiDependencies, "invoker", None)
        if invoker is not None and invoker is not original_invoker:
            ApiDependencies.shutdown()
            for worker in invoker.services.session_processor._workers:
                assert worker.thread is not None
                worker.thread.join(timeout=5)
            if original_invoker is None:
                delattr(ApiDependencies, "invoker")
            else:
                ApiDependencies.invoker = original_invoker
        ApiDependencies.database = None
        # The processor registered the devices and a release listener in the process-wide pool.
        GENERATION_DEVICE_POOL.reset()
        loop.close()


def test_a_restart_refused_for_a_worker_the_last_stop_gave_up_on_opens_no_database(monkeypatch, tmp_path) -> None:
    _skip_process_wide_setup(monkeypatch)
    opened: list[Database] = []
    original_init_database = dependencies.init_database

    def init_database(*args, **kwargs) -> Database:
        opened.append(original_init_database(*args, **kwargs))
        return opened[-1]

    monkeypatch.setattr(dependencies, "init_database", init_database)
    # A session worker the previous invoker's stop gave up on: still in its node.
    release = threading.Event()
    stuck = threading.Thread(target=release.wait, name="session_processor_0", daemon=True)
    stuck.start()
    monkeypatch.setattr(session_processor_default, "_abandoned_workers", [stuck])
    config = InvokeAIAppConfig(use_memory_db=True, image_index_enabled=False, scan_models_on_startup=False)
    config._root = tmp_path
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RuntimeError, match="still running"):
            ApiDependencies.initialize(config=config, event_handler_id=0, loop=loop, logger=getLogger("test"))
        assert opened == []
    finally:
        release.set()
        stuck.join(timeout=30)
        for database in opened:
            database.dispose()
        GENERATION_DEVICE_POOL.reset()
        loop.close()
