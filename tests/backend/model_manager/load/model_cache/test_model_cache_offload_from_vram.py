"""`ModelCache.offload_models_from_vram_except`: what an architecture switch moves out of VRAM, and what it keeps.

The entries stay cached (in RAM); locked entries and every submodel of a kept model stay where they are.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache

MB = 2**20
MODULE = "invokeai.backend.model_manager.load.model_cache.model_cache"


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
    )
    yield cache
    cache.shutdown()


@pytest.fixture
def moved(cache: ModelCache):
    """Record which entries the cache moves to RAM. A CPU cache stands in for a GPU one: it reports its entries as
    resident and its device as having dedicated VRAM, but nothing really moves."""
    keys: list[str] = []

    def move(cache_entry, vram_bytes_to_free, keep_required_weights_in_vram=None):
        keys.append(cache_entry.key)
        return cache_entry.cached_model.total_bytes()

    with (
        patch.object(cache, "_move_model_to_ram", side_effect=move),
        patch(f"{MODULE}._has_dedicated_vram", return_value=True),
        patch(f"{MODULE}.TorchDevice.empty_cache") as empty_cache,
    ):
        yield keys, empty_cache


def _put(cache: ModelCache, key: str, mb: int, in_vram: bool = True) -> None:
    cache.put(key, torch.nn.Linear(mb * MB // 4, 1, bias=False))
    cached_model = cache._cached_models[key].cached_model
    vram = cached_model.total_bytes() if in_vram else 0
    cached_model.cur_vram_bytes = lambda: vram  # type: ignore[method-assign]


def test_models_the_session_does_not_name_move_to_ram_and_stay_cached(cache: ModelCache, moved):
    keys, empty_cache = moved
    _put(cache, "flux:transformer", 8)
    _put(cache, "flux:vae", 1)
    _put(cache, "qwen3:text_encoder", 4)  # shared with the next session
    _put(cache, "zimage:transformer", 2)
    _put(cache, r"D:\models\upscaler.safetensors", 1)  # loaded by path, names no model key

    freed = cache.offload_models_from_vram_except({"zimage", "qwen3"})

    assert sorted(keys) == sorted(["flux:transformer", "flux:vae", r"D:\models\upscaler.safetensors"])
    assert freed == 10 * MB
    assert len(cache._cached_models) == 5
    empty_cache.assert_called_once()


def test_locked_entries_stay(cache: ModelCache, moved):
    keys, _ = moved
    _put(cache, "flux:transformer", 8)
    _put(cache, "borrowed:text_encoder", 4)
    cache._cached_models["borrowed:text_encoder"].lock()

    cache.offload_models_from_vram_except(set())

    assert keys == ["flux:transformer"]


def test_entries_already_in_ram_are_not_moved_again(cache: ModelCache, moved):
    keys, empty_cache = moved
    _put(cache, "flux:transformer", 8, in_vram=False)

    assert cache.offload_models_from_vram_except(set()) == 0
    assert keys == []
    empty_cache.assert_not_called()


def test_nothing_to_move_spends_no_empty_cache(cache: ModelCache, moved):
    keys, empty_cache = moved
    _put(cache, "zimage:transformer", 2)

    assert cache.offload_models_from_vram_except({"zimage"}) == 0
    assert keys == []
    empty_cache.assert_not_called()


def test_offload_model_from_vram_still_moves_one_model_with_its_submodels(cache: ModelCache, moved):
    keys, _ = moved
    _put(cache, "flux", 1)
    _put(cache, "flux:vae", 1)
    _put(cache, "fluxier:vae", 1)  # shares the prefix text, not the key

    cache.offload_model_from_vram("flux")

    assert sorted(keys) == ["flux", "flux:vae"]


def test_a_device_whose_vram_is_system_ram_moves_nothing(cache: ModelCache):
    """CPU, MPS and integrated GPUs: moving a model "to RAM" would copy it within the same memory."""
    cache.put("flux:transformer", torch.nn.Linear(MB // 4, 1, bias=False))
    with patch.object(cache, "_move_model_to_ram") as move:
        assert cache.offload_models_from_vram_except(set()) == 0
    move.assert_not_called()
