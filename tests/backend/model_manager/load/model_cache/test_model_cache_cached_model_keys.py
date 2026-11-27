"""Tests for ModelCache.cached_model_keys(), which feeds the session queue's device-affinity
heuristic and must therefore (a) report only model-key-shaped keys and (b) never block."""

import logging
import threading

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache


@pytest.fixture
def cache() -> ModelCache:
    return ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logging.getLogger(__name__),
    )


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Linear(2, 2)


def test_cached_model_keys_strips_submodel_suffix(cache: ModelCache) -> None:
    key = "aaaaaaaa-1111-2222-3333-444444444444"
    cache.put(f"{key}:unet", _tiny_model())
    cache.put(f"{key}:vae", _tiny_model())
    assert cache.cached_model_keys() == {key}


def test_cached_model_keys_excludes_path_shaped_keys(cache: ModelCache) -> None:
    """load_model_from_path keys entries by str(model_path); neither a POSIX path, a Windows
    path, nor the bare drive letter left by splitting 'C:\\...' on ':' may leak into affinity
    scoring, where a short string like 'C' would substring-match nearly every session."""
    model_key = "aaaaaaaa-1111-2222-3333-444444444444"
    cache.put(model_key, _tiny_model())
    cache.put("/home/user/models/upscaler.pth", _tiny_model())
    cache.put("C:\\Users\\user\\models\\dw-ll_ucoco.onnx", _tiny_model())
    assert cache.cached_model_keys() == {model_key}


def test_cached_model_keys_does_not_block_on_contended_lock(cache: ModelCache) -> None:
    """The cache lock is held across long operations (VRAM transfers, cache clears); the keys
    lookup must return an empty set immediately rather than stall a worker's dequeue."""
    cache.put("aaaaaaaa-1111-2222-3333-444444444444", _tiny_model())

    lock_held = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with cache._lock:
            lock_held.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    try:
        assert lock_held.wait(timeout=10)
        assert cache.cached_model_keys() == set()
    finally:
        release.set()
        holder.join(timeout=10)

    # Uncontended again: the real contents are visible.
    assert cache.cached_model_keys() == {"aaaaaaaa-1111-2222-3333-444444444444"}
