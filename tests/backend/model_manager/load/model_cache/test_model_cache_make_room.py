"""Tests for make_room()'s eviction guards."""

import logging
from unittest.mock import MagicMock

from invokeai.backend.model_manager.load.load_base import LoadedModelWithoutConfig
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from tests.backend.model_manager.load.model_cache.cached_model.utils import DummyModule


def _make_cache() -> ModelCache:
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logger,
    )


def test_held_record_survives_sibling_admissions_until_it_locks():
    """Multi-model invocations load their whole set before locking any of it: an earlier
    sibling's wrapper must survive later puts (whose sweep clears the unowned admission grace)
    and the make_rooms they trigger, for as long as the wrapper is alive and un-entered."""
    cache = _make_cache()
    cache.put("first", DummyModule())
    record, claim = cache.get_with_first_use_claim("first")
    handle = LoadedModelWithoutConfig(cache_record=record, cache=cache, first_use_claim=claim)

    # Two more siblings admitted — each put() runs the stale-grace sweep and a make_room.
    cache.put("second", DummyModule())
    cache.put("third", DummyModule())
    assert record.first_use_holds > 0, "a sibling's put() stripped a live wrapper's first-use hold"

    cache.make_room(10**15)
    assert "first" in cache._cached_models

    # The handle finally locks: hold released, ordinary cache content again.
    with handle:
        pass
    assert record.first_use_holds == 0
    cache.make_room(10**15)
    assert "first" not in cache._cached_models


def test_make_room_reports_what_it_cleared():
    """The clear-model-cache route sums each device cache's result, so make_room must report
    the entries it dropped and the bytes they held."""
    cache = _make_cache()
    cache.put("fresh", DummyModule())
    record = cache.get("fresh")
    size = record.cached_model.total_bytes()

    result = cache.make_room(10**15)
    assert "fresh" not in cache._cached_models
    assert result.models_cleared == 1
    assert result.bytes_freed == size
