"""Tests for `ModelCache.drop_model` — used by the model_manager API to invalidate cached
entries when a setting that changes how a model loads (e.g. `fp8_storage`, `cpu_only`) is
toggled. Without this, the toggle is silently a no-op until the entry is evicted by other
means (clear cache, eviction under memory pressure, restart).

Also covers:
- Locked entries are marked stale and evicted by `unlock()` — without that, a setting toggled
  during an in-flight generation would survive on the locked entry and silently be reused.
- `stats.cleared` and the `cleared` callbacks fire on invalidation, mirroring the eviction
  path through `_make_room_internal`, so observers and stats stay accurate.
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cache_stats import CacheStats
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache


@pytest.fixture
def mock_logger():
    logger = MagicMock()
    logger.getEffectiveLevel.return_value = logging.INFO
    return logger


@pytest.fixture
def cache(mock_logger):
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=mock_logger,
    )
    yield cache
    cache.shutdown()


def test_drop_model_removes_all_submodel_entries(cache: ModelCache):
    """A model with multiple submodels has multiple cache keys (`<key>` and `<key>:<submodel>`);
    drop_model must drop them all together so the next load rebuilds with the new settings.
    """
    model_key = "abc123"
    cache.put(model_key, torch.randn(4))
    cache.put(f"{model_key}:unet", torch.randn(4))
    cache.put(f"{model_key}:text_encoder", torch.randn(4))
    cache.put("other_model", torch.randn(4))
    cache.put("other_model:unet", torch.randn(4))

    dropped = cache.drop_model(model_key)

    assert dropped == 3
    assert model_key not in cache._cached_models
    assert f"{model_key}:unet" not in cache._cached_models
    assert f"{model_key}:text_encoder" not in cache._cached_models
    # Unrelated model is left alone.
    assert "other_model" in cache._cached_models
    assert "other_model:unet" in cache._cached_models


def test_drop_model_marks_locked_entries_stale_without_evicting(cache: ModelCache):
    """Locked entries are in active use; we must not yank them out from under inference.
    But we also must not silently retain them after the lock releases — otherwise a setting
    toggle that happened during inference would survive and the next generation would reuse
    the pre-change cached module. drop_model marks locked entries `is_stale=True`; unlock
    evicts them as soon as the last lock releases.
    """
    model_key = "abc123"
    cache.put(model_key, torch.randn(4))
    cache.put(f"{model_key}:unet", torch.randn(4))

    locked_entry = cache._cached_models[f"{model_key}:unet"]
    locked_entry.lock()

    dropped = cache.drop_model(model_key)

    assert dropped == 1
    assert model_key not in cache._cached_models
    assert f"{model_key}:unet" in cache._cached_models
    assert locked_entry.is_stale is True


def test_unlock_evicts_stale_entry(cache: ModelCache):
    """The flip side of `marks_locked_entries_stale`: the next `unlock` after a stale-marking
    invalidation must actually remove the entry.
    """
    model_key = "abc123"
    cache.put(model_key, torch.randn(4))
    entry = cache._cached_models[model_key]
    entry.lock()

    cache.drop_model(model_key)

    # Entry still here while locked.
    assert model_key in cache._cached_models
    assert entry.is_stale is True

    cache.unlock(entry)

    assert model_key not in cache._cached_models


def test_unlock_does_not_evict_non_stale_entry(cache: ModelCache):
    """The stale-eviction path must not affect ordinary unlock — only stale-marked entries
    should be evicted on unlock.
    """
    model_key = "abc123"
    cache.put(model_key, torch.randn(4))
    entry = cache._cached_models[model_key]
    entry.lock()

    cache.unlock(entry)

    # No drop_model was called, so entry should still be there.
    assert model_key in cache._cached_models


def test_unlock_only_evicts_when_last_lock_releases(cache: ModelCache):
    """If the entry is held by multiple locks (the cache supports re-entrant locking via
    `_locks`), eviction must wait until they all release. Otherwise we'd yank the entry out
    from under a caller that still expects it loaded.
    """
    model_key = "abc123"
    cache.put(model_key, torch.randn(4))
    entry = cache._cached_models[model_key]
    entry.lock()
    entry.lock()

    cache.drop_model(model_key)
    assert entry.is_stale is True

    cache.unlock(entry)
    # Still locked by one holder — must remain.
    assert model_key in cache._cached_models

    cache.unlock(entry)
    # Now fully released — eviction happens.
    assert model_key not in cache._cached_models


def test_drop_model_updates_stats_and_fires_callbacks(cache: ModelCache):
    """drop_model is a real eviction path — observers watching for cache changes (stats,
    cleared callbacks) must see it just like the make_room eviction path. Otherwise the UI
    cache-stats panel and any external observer would miss invalidations.
    """
    model_key = "abc123"
    # Use real nn.Modules so `total_bytes()` is non-zero (raw tensors are sized as 0 by
    # `calc_model_size_by_data` since the cache doesn't know what they are).
    cache.put(model_key, torch.nn.Linear(4, 4))
    cache.put(f"{model_key}:unet", torch.nn.Linear(4, 4))

    cache.stats = CacheStats()
    callback = MagicMock()
    cache.on_cache_models_cleared(callback)

    dropped = cache.drop_model(model_key)

    assert dropped == 2
    assert cache.stats.cleared == 2
    callback.assert_called_once()
    kwargs = callback.call_args.kwargs
    assert kwargs["models_cleared"] == 2
    assert kwargs["bytes_requested"] == 0  # not a make-room call
    assert kwargs["bytes_freed"] > 0


def test_unlock_stale_eviction_updates_stats_and_fires_callbacks(cache: ModelCache):
    """Stale-entry eviction is also a cache change observers care about."""
    model_key = "abc123"
    cache.put(model_key, torch.randn(4))
    entry = cache._cached_models[model_key]
    entry.lock()

    cache.drop_model(model_key)

    cache.stats = CacheStats()
    callback = MagicMock()
    cache.on_cache_models_cleared(callback)

    cache.unlock(entry)

    assert model_key not in cache._cached_models
    assert cache.stats.cleared == 1
    callback.assert_called_once()


def test_drop_model_with_no_matches_does_not_fire_callbacks(cache: ModelCache):
    """No-op invalidations should be silent — don't spam observers with empty events."""
    cache.put("other_model", torch.randn(4))
    callback = MagicMock()
    cache.on_cache_models_cleared(callback)

    dropped = cache.drop_model("does_not_exist")

    assert dropped == 0
    callback.assert_not_called()


def test_drop_model_with_no_matches_is_noop(cache: ModelCache):
    cache.put("other_model", torch.randn(4))

    dropped = cache.drop_model("does_not_exist")

    assert dropped == 0
    assert "other_model" in cache._cached_models


def test_drop_model_does_not_match_prefix_substring(cache: ModelCache):
    """`drop_model("abc")` must not drop `abcd` — only the exact key or `abc:<submodel>`."""
    cache.put("abc", torch.randn(4))
    cache.put("abcd", torch.randn(4))
    cache.put("abc:unet", torch.randn(4))

    dropped = cache.drop_model("abc")

    assert dropped == 2
    assert "abc" not in cache._cached_models
    assert "abc:unet" not in cache._cached_models
    assert "abcd" in cache._cached_models
