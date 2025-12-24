"""Tests for model cache keep-alive timeout functionality."""
import time
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return MagicMock()


@pytest.fixture
def model_cache_with_timeout(mock_logger):
    """Create a ModelCache instance with a short timeout for testing."""
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=mock_logger,
        keep_alive_minutes=0.01,  # 0.6 seconds for fast testing
    )
    yield cache
    cache.shutdown()


@pytest.fixture
def model_cache_no_timeout(mock_logger):
    """Create a ModelCache instance without timeout (default behavior)."""
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=mock_logger,
        keep_alive_minutes=0,  # 0 means no timeout
    )
    yield cache
    cache.shutdown()


def test_timeout_clears_cache(model_cache_with_timeout):
    """Test that the cache is cleared after the timeout expires."""
    cache = model_cache_with_timeout

    # Add a simple tensor to the cache
    test_tensor = torch.randn(10, 10)
    cache.put("test_model", test_tensor)

    # Verify the model is in the cache
    assert "test_model" in cache._cached_models

    # Wait for the timeout to expire (0.01 minutes = 0.6 seconds + buffer)
    time.sleep(1.5)

    # Verify the cache has been cleared
    assert len(cache._cached_models) == 0


def test_activity_resets_timeout(model_cache_with_timeout):
    """Test that model activity resets the timeout."""
    cache = model_cache_with_timeout

    # Add a simple tensor to the cache
    test_tensor = torch.randn(10, 10)
    cache.put("test_model", test_tensor)

    # Wait half the timeout
    time.sleep(0.4)

    # Access the model to reset the timeout
    cache.get("test_model")

    # Wait another half timeout (model should still be in cache)
    time.sleep(0.4)

    # Verify the model is still in the cache
    assert "test_model" in cache._cached_models


def test_no_timeout_keeps_models(model_cache_no_timeout):
    """Test that models are kept indefinitely when timeout is 0."""
    cache = model_cache_no_timeout

    # Add a simple tensor to the cache
    test_tensor = torch.randn(10, 10)
    cache.put("test_model", test_tensor)

    # Verify the model is in the cache
    assert "test_model" in cache._cached_models

    # Wait longer than what would be a timeout
    time.sleep(1.0)

    # Verify the model is still in the cache
    assert "test_model" in cache._cached_models


def test_shutdown_cancels_timer(model_cache_with_timeout):
    """Test that shutdown properly cancels the timeout timer."""
    cache = model_cache_with_timeout

    # Add a model to start the timer
    test_tensor = torch.randn(10, 10)
    cache.put("test_model", test_tensor)

    # Shutdown the cache
    cache.shutdown()

    # Wait for what would be the timeout
    time.sleep(1.0)

    # The model should still be in the cache since shutdown was called
    assert "test_model" in cache._cached_models
