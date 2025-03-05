import pytest

from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation

TIMESTEPS = [1.0, 0.75, 0.5, 0.25, 0.0]


@pytest.mark.parametrize(
    ["cfg_scale", "timesteps", "cfg_scale_start_step", "cfg_scale_end_step", "expected"],
    [
        # Test scalar cfg_scale.
        (2.0, TIMESTEPS, 0, -1, [2.0, 2.0, 2.0, 2.0]),
        # Test list cfg_scale.
        ([1.0, 2.0, 3.0, 4.0], TIMESTEPS, 0, -1, [1.0, 2.0, 3.0, 4.0]),
        # Test positive cfg_scale_start_step.
        (2.0, TIMESTEPS, 1, -1, [1.0, 2.0, 2.0, 2.0]),
        # Test positive cfg_scale_end_step.
        (2.0, TIMESTEPS, 0, 2, [2.0, 2.0, 2.0, 1.0]),
        # Test negative cfg_scale_start_step.
        (2.0, TIMESTEPS, -3, -1, [1.0, 2.0, 2.0, 2.0]),
        # Test negative cfg_scale_end_step.
        (2.0, TIMESTEPS, 0, -2, [2.0, 2.0, 2.0, 1.0]),
        # Test single step application.
        (2.0, TIMESTEPS, 2, 2, [1.0, 1.0, 2.0, 1.0]),
    ],
)
def test_prep_cfg_scale(
    cfg_scale: float | list[float],
    timesteps: list[float],
    cfg_scale_start_step: int,
    cfg_scale_end_step: int,
    expected: list[float],
):
    result = FluxDenoiseInvocation.prep_cfg_scale(cfg_scale, timesteps, cfg_scale_start_step, cfg_scale_end_step)
    assert result == expected


def test_prep_cfg_scale_invalid_type():
    with pytest.raises(ValueError, match="Unsupported cfg_scale type"):
        FluxDenoiseInvocation.prep_cfg_scale("invalid", [1.0, 0.5], 0, -1)  # type: ignore


@pytest.mark.parametrize("cfg_scale_start_step", [4, -5])
def test_prep_cfg_scale_invalid_start_step(cfg_scale_start_step: int):
    with pytest.raises(ValueError, match="Invalid cfg_scale_start_step"):
        FluxDenoiseInvocation.prep_cfg_scale(2.0, TIMESTEPS, cfg_scale_start_step, -1)


@pytest.mark.parametrize("cfg_scale_end_step", [4, -5])
def test_prep_cfg_scale_invalid_end_step(cfg_scale_end_step: int):
    with pytest.raises(ValueError, match="Invalid cfg_scale_end_step"):
        FluxDenoiseInvocation.prep_cfg_scale(2.0, TIMESTEPS, 0, cfg_scale_end_step)


def test_prep_cfg_scale_start_after_end():
    with pytest.raises(ValueError, match="cfg_scale_start_step .* must be before cfg_scale_end_step"):
        FluxDenoiseInvocation.prep_cfg_scale(2.0, TIMESTEPS, 3, 2)


def test_prep_cfg_scale_list_length_mismatch():
    with pytest.raises(AssertionError):
        FluxDenoiseInvocation.prep_cfg_scale([1.0, 2.0, 3.0], TIMESTEPS, 0, -1)
