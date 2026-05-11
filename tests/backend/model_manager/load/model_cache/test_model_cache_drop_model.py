"""Tests for `ModelCache.drop_model` — used by the model_manager API to invalidate cached
entries when a setting that changes how a model loads (e.g. `fp8_storage`, `cpu_only`) is
toggled. Without this, the toggle is silently a no-op until the entry is evicted by other
means (clear cache, eviction under memory pressure, restart).
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch

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


def test_drop_model_does_not_drop_locked_entries(cache: ModelCache):
    """Locked entries are in active use; we must not yank them out from under inference.
    The caller's expectation is that the next load rebuilds — that still happens once the
    lock releases and the entry is evicted (or is correctly re-loaded after another flush).
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
