"""Unit tests for the ModelCache working-memory reserve logic (smart_partial_loading)."""

from unittest.mock import MagicMock, patch

from invokeai.backend.model_manager.load.model_cache.model_cache import (
    GB,
    MB,
    ModelCache,
    PARTIAL_LOAD_HEADROOM_MULTIPLIER,
)


def _make_cache(**kwargs) -> ModelCache:
    defaults = {
        "execution_device_working_mem_gb": 3,
        "enable_partial_loading": True,
        "keep_ram_copy_of_weights": False,
        "smart_partial_loading": True,
        "execution_device": "cuda",
    }
    defaults.update(kwargs)
    return ModelCache(**defaults)


def _make_partial_entry(total_bytes: int, cur_vram_bytes: int) -> MagicMock:
    entry = MagicMock()
    entry.key = "test-model"
    entry.cached_model.total_bytes.return_value = total_bytes
    entry.cached_model.cur_vram_bytes.return_value = cur_vram_bytes
    return entry


# --- partial-load headroom in _load_locked_model ---


def test_partial_load_headroom_survives_self_unload():
    """Re-lock scenario: the model is still resident from a previous (lower-reserve) run, and the new
    op's headroom reserve drives vram_available negative. The last-resort branch unloads the deficit
    from the locked model itself — the reload budget after that unload must be computed with the SAME
    effective (headroom) reserve. Recomputing with the bare estimate would hand the just-freed bytes
    straight back, reloading the model into the thrash zone the headroom exists to avoid."""
    cache = _make_cache()
    estimate = 1 * GB
    headroom_reserve = cache._partial_load_reserve(estimate)
    assert headroom_reserve > estimate  # sanity: the scenario exercises two distinct reserves

    # Model 10GB, 9.6GB already resident -> 0.4GB still needed (genuinely partial).
    entry = _make_partial_entry(total_bytes=10 * GB, cur_vram_bytes=int(9.6 * GB))

    # Stateful availability: with the bare estimate reserve we are 100MB short; with the headroom
    # reserve we are 900MB short. Bytes freed by the self-unload increase availability.
    state = {"freed": 0}
    base_available = {estimate: -100 * MB, headroom_reserve: -900 * MB}

    def fake_get_vram_available(working_mem_bytes):
        return base_available[working_mem_bytes] + state["freed"]

    def fake_move_model_to_ram(_entry, vram_bytes_to_free):
        state["freed"] += vram_bytes_to_free
        return vram_bytes_to_free

    vram_budgets = []

    def fake_move_model_to_vram(_entry, vram_available):
        vram_budgets.append(vram_available)
        return max(0, vram_available)

    with (
        patch.object(cache, "_get_vram_available", side_effect=fake_get_vram_available),
        patch.object(cache, "_offload_unlocked_models", return_value=0),
        patch.object(cache, "_move_model_to_ram", side_effect=fake_move_model_to_ram),
        patch.object(cache, "_move_model_to_vram", side_effect=fake_move_model_to_vram),
    ):
        cache._load_locked_model(entry, working_mem_bytes=estimate)

    # The self-unload freed the 900MB headroom deficit...
    assert state["freed"] == 900 * MB
    # ...so the reload budget must be ~0 (the +1MB tracking-error allowance), not deficit + 800MB.
    assert len(vram_budgets) == 1
    assert vram_budgets[0] <= 1 * MB, (
        f"reload budget {vram_budgets[0] / MB:.0f}MB hands the freed headroom deficit straight back"
    )


def test_partial_load_headroom_multiplier_is_active():
    """Guard for the scenario above: the headroom multiplier must actually produce a larger reserve
    than the bare estimate for a mid-sized estimate, or the headroom path is inert."""
    assert PARTIAL_LOAD_HEADROOM_MULTIPLIER > 1.0
