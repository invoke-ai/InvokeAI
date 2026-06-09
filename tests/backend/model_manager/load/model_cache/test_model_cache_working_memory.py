"""Unit tests for the ModelCache working-memory reserve logic (smart_partial_loading)."""

from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.load.model_cache.model_cache import (
    GB,
    MB,
    MIN_DEVICE_WORKING_MEM_GB,
    ModelCache,
    PARTIAL_LOAD_HEADROOM_MULTIPLIER,
)

# Free VRAM reported by the mocked CUDA queries below.
FREE = 8 * GB
ALLOCATED = 2 * GB


def _vram_available(cache: ModelCache, working_mem_bytes) -> int:
    """Call _get_vram_available with the CUDA memory queries mocked out.

    With memory_allocated mocked to a single value, the math reduces to
    available = free - reserve (the cache's own usage cancels out), so each test
    asserts the RESERVE the cache chose for the given inputs.
    """
    with (
        patch("torch.cuda.memory_allocated", return_value=ALLOCATED),
        patch("torch.cuda.mem_get_info", return_value=(FREE, 24 * GB)),
    ):
        return cache._get_vram_available(working_mem_bytes)


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


@pytest.mark.parametrize(
    ("smart", "working_mem_bytes", "max_vram_cache_size_gb", "model_fits"),
    [
        (False, 1 * GB, None, False),  # smart disabled
        (True, None, None, False),  # un-instrumented op (no estimate)
        (True, 1 * GB, None, True),  # model fully fits under the small reserve
        (True, 1 * GB, 8, False),  # fixed VRAM cap overrides the reserve
    ],
)
def test_partial_load_headroom_only_engages_when_applicable(smart, working_mem_bytes, max_vram_cache_size_gb, model_fits):
    """The headroom reserve must only be consulted for a smart-loaded, estimate-carrying, genuinely
    partial model with no fixed VRAM cap. In every other case _get_vram_available must only ever see
    the op's own reserve value."""
    cache = _make_cache(smart_partial_loading=smart, max_vram_cache_size_gb=max_vram_cache_size_gb)
    available = 10 * GB if model_fits else 100 * MB
    entry = _make_partial_entry(total_bytes=10 * GB, cur_vram_bytes=int(9.6 * GB))  # 0.4GB still needed

    reserves_requested = []

    def fake_get_vram_available(wm):
        reserves_requested.append(wm)
        return available

    with (
        patch.object(cache, "_get_vram_available", side_effect=fake_get_vram_available),
        patch.object(cache, "_offload_unlocked_models", return_value=0),
        patch.object(cache, "_move_model_to_vram", return_value=0),
    ):
        cache._load_locked_model(entry, working_mem_bytes=working_mem_bytes)

    assert set(reserves_requested) == {working_mem_bytes}


# --- _get_vram_available reserve semantics ---


def test_reserve_legacy_is_one_directional_floor():
    """smart_partial_loading off: callers may only RAISE the reserve above the configured default."""
    cache = _make_cache(smart_partial_loading=False)
    assert _vram_available(cache, None) == FREE - 3 * GB
    assert _vram_available(cache, 1 * GB) == FREE - 3 * GB  # lowering is ignored
    assert _vram_available(cache, 5 * GB) == FREE - 5 * GB  # raising is honored


def test_reserve_smart_uninstrumented_op_keeps_default():
    """smart on, no estimate: the full default reserve stays as the safety net."""
    cache = _make_cache()
    assert _vram_available(cache, None) == FREE - 3 * GB


def test_reserve_smart_honors_estimate_down_to_floor():
    cache = _make_cache()
    floor = int(MIN_DEVICE_WORKING_MEM_GB * GB)
    assert _vram_available(cache, 512 * MB) == FREE - floor  # clamped up to the floor
    assert _vram_available(cache, 2 * GB) == FREE - 2 * GB  # between floor and default: honored
    assert _vram_available(cache, 4 * GB) == FREE - 4 * GB  # above default: honored (raise)


def test_reserve_smart_user_raised_default_is_the_floor():
    """A user-raised device_working_mem_gb is an OOM mitigation; smart mode must not go below it."""
    cache = _make_cache(execution_device_working_mem_gb=6, device_working_mem_raised=True)
    assert _vram_available(cache, 2 * GB) == FREE - 6 * GB
    assert _vram_available(cache, 7 * GB) == FREE - 7 * GB


def test_reserve_smart_user_lowered_default():
    """A user-lowered default applies to un-instrumented ops; instrumented ops keep the smart floor."""
    cache = _make_cache(execution_device_working_mem_gb=2)
    assert _vram_available(cache, None) == FREE - 2 * GB
    assert _vram_available(cache, 512 * MB) == FREE - int(MIN_DEVICE_WORKING_MEM_GB * GB)


def test_reserve_ignored_when_fixed_vram_cap_set():
    cache = _make_cache(max_vram_cache_size_gb=4)
    for wm in (None, 1 * GB, 5 * GB):
        assert _vram_available(cache, wm) == 4 * GB - ALLOCATED


# --- _partial_load_reserve clamps ---


def test_partial_load_reserve_scales_between_floor_and_default():
    cache = _make_cache()
    floor = int(MIN_DEVICE_WORKING_MEM_GB * GB)
    assert cache._partial_load_reserve(512 * MB) == floor  # scaled (768MB) still below floor
    assert cache._partial_load_reserve(1 * GB) == int(1 * GB * PARTIAL_LOAD_HEADROOM_MULTIPLIER)
    assert cache._partial_load_reserve(int(2.5 * GB)) == 3 * GB  # scaled (3.75GB) capped at default
    assert cache._partial_load_reserve(4 * GB) == 4 * GB  # estimate above default: cap = estimate


def test_partial_load_reserve_respects_user_raised_default():
    cache = _make_cache(execution_device_working_mem_gb=6, device_working_mem_raised=True)
    assert cache._partial_load_reserve(2 * GB) == 6 * GB  # floor = raised default
    assert cache._partial_load_reserve(7 * GB) == 7 * GB  # estimate above raised default: cap = estimate
