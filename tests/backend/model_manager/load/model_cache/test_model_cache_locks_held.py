"""`ModelCache.locks_held`: every pin on the cache's records, whichever thread took it and of whichever kind. The
working-memory probe compares it with its own thread's locks to see another thread sharing the device."""

import logging
import threading
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache


@pytest.fixture
def cache():
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logger,
        shared_cpu_weights=None,
    )
    yield cache
    cache.shutdown()


def test_counts_device_and_ram_locks_of_every_thread(cache: ModelCache):
    cache.put("vae", torch.nn.Linear(2, 2))
    cache.put("lora", torch.nn.Linear(2, 2))
    vae, lora = cache.get("vae"), cache.get("lora")

    cache.lock(vae, None)
    cache.lock(vae, None)
    other = threading.Thread(target=cache.lock_in_ram, args=(lora,))
    other.start()
    other.join()

    assert cache.locks_held() == 3

    cache.unlock(vae)
    cache.unlock(lora)
    assert cache.locks_held() == 1
    cache.unlock(vae)
    assert cache.locks_held() == 0
