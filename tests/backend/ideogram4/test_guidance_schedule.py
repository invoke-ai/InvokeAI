"""Tests for Ideogram 4's effective guidance schedule.

The schedule is ``(polish_gw,)*N_polish + (main_gw,)*N_main`` in loop-index order (index 0 is the
final/polish step). A guidance_scale override must replace the main weight while preserving the
polish tail, and there must always be at least one main step so the override is never silently
dropped — the regression that motivated capping the polish tail at ``num_steps - 1``.
"""

import pytest

from invokeai.app.invocations.ideogram4.ideogram4_denoise import _effective_guidance_schedule
from invokeai.backend.ideogram4.sampler_configs import PRESETS

_QUALITY = PRESETS["V4_QUALITY_48"]  # num_steps=48, schedule=(3.0,)*3 + (7.0,)*45


def test_returns_base_schedule_unchanged_at_preset_steps_without_override():
    result = _effective_guidance_schedule(_QUALITY.guidance_schedule, _QUALITY.num_steps, _QUALITY.num_steps, None)
    assert result == _QUALITY.guidance_schedule


def test_guidance_override_replaces_main_weight_and_preserves_polish():
    result = _effective_guidance_schedule(_QUALITY.guidance_schedule, _QUALITY.num_steps, _QUALITY.num_steps, 12.0)
    assert len(result) == _QUALITY.num_steps
    # Polish tail (weight 3.0) preserved; every main step now carries the 12.0 override.
    assert result[0] == 3.0
    assert result[-1] == 12.0
    assert set(result) == {3.0, 12.0}


def test_two_step_override_keeps_one_polish_and_one_main():
    # The minimum allowed step count. Regression: previously polish_count could equal num_steps,
    # leaving main_count == 0 and silently dropping the guidance override.
    result = _effective_guidance_schedule(_QUALITY.guidance_schedule, _QUALITY.num_steps, 2, 15.0)
    assert result == (3.0, 15.0)


@pytest.mark.parametrize("num_steps", list(range(2, 60)))
def test_at_least_one_main_step_and_override_always_applied(num_steps: int):
    override = 9.5
    result = _effective_guidance_schedule(_QUALITY.guidance_schedule, _QUALITY.num_steps, num_steps, override)
    assert len(result) == num_steps
    main_count = sum(1 for gw in result if gw == override)
    polish_count = len(result) - main_count
    assert main_count >= 1, "the guidance override must always occupy at least one step"
    assert polish_count >= 1, "the polish tail must always be preserved"
    assert result[-1] == override
    assert result[0] == _QUALITY.guidance_schedule[0]
