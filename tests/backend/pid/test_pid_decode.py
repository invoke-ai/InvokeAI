"""Regression tests for the PiD distill schedule and decoder/base validation."""

from unittest.mock import patch

import pytest
import torch

from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.pid.decode import (
    PiDDecodeConfig,
    _get_t_list,
    _student_sample_loop,
    _velocity_to_x0,
    assert_pid_decoder_matches_base,
    estimate_pid_decode_working_memory,
)

_CPU = torch.device("cpu")


@pytest.mark.parametrize("num_steps", [1, 2, 3, 4])
def test_student_schedule_is_strictly_decreasing(num_steps: int) -> None:
    """Every permitted step count yields a strictly decreasing schedule with no duplicate timesteps.

    The student schedule has only four transitions; sub-sampling to >4 steps rounds distinct indices
    onto the same point and produces duplicates (e.g. 5 steps → [.999, .866, .634, .634, .342, 0]),
    which is why the public range is capped at 4.
    """
    t = _get_t_list(_CPU, num_steps=num_steps).tolist()
    assert len(t) == num_steps + 1
    assert t[-1] == pytest.approx(0.0, abs=1e-6)
    assert all(a > b for a, b in zip(t[:-1], t[1:], strict=True)), t
    assert len(set(t)) == len(t)


def test_default_schedule_matches_four_steps() -> None:
    assert _get_t_list(_CPU).tolist() == _get_t_list(_CPU, num_steps=4).tolist()


def test_out_of_range_step_count_trips_the_safety_net() -> None:
    """If an invalid count ever bypassed the field cap, the guard raises. Uses ValueError (not assert)
    so it still fires under `python -O`, where assertions are stripped."""
    with pytest.raises(ValueError, match="strictly decreasing"):
        _get_t_list(_CPU, num_steps=5)


@pytest.mark.parametrize(
    ("x_dtype", "net_output_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_velocity_to_x0_uses_float32_math_and_preserves_input_dtype(
    x_dtype: torch.dtype, net_output_dtype: torch.dtype
) -> None:
    x_t = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=x_dtype)
    net_output = torch.tensor([[[[0.5, -0.25], [0.125, -0.0625]]]], dtype=net_output_dtype)
    timestep = torch.tensor([0.634], dtype=torch.float32)
    expected = (x_t.float() - timestep.view(1, 1, 1, 1) * net_output.float()).to(x_dtype)

    # A float64 intermediate doubles memory for each full-resolution sampler tensor. PiD already
    # predicts under bf16 autocast, so perform this update in float32 without calling Tensor.double().
    with patch.object(torch.Tensor, "double", side_effect=AssertionError("unexpected float64 conversion")):
        actual = _velocity_to_x0(x_t, net_output, timestep, pid_memory_optimization=True)

    assert actual.dtype == x_dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_velocity_to_x0_uses_original_float64_math_when_optimization_is_disabled() -> None:
    x_t = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=torch.bfloat16)
    net_output = torch.tensor([[[[0.5, -0.25], [0.125, -0.0625]]]], dtype=torch.bfloat16)
    timestep = torch.tensor([0.634], dtype=torch.float32)
    expected = (x_t.double() - timestep.double().view(1, 1, 1, 1) * net_output.double()).to(x_t.dtype)

    with patch("torch.addcmul", side_effect=AssertionError("unexpected optimized path")):
        actual = _velocity_to_x0(x_t, net_output, timestep, pid_memory_optimization=False)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("pid_memory_optimization", "expected_chunk_size"), [(False, None), (True, 1024)])
def test_student_sample_loop_passes_per_call_activation_chunk_size(
    pid_memory_optimization: bool, expected_chunk_size: int | None
) -> None:
    activation_chunk_sizes: list[int | None] = []

    def net(x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        activation_chunk_sizes.append(kwargs.get("activation_chunk_size"))  # type: ignore[arg-type]
        return torch.zeros_like(x)

    _student_sample_loop(
        net,  # type: ignore[arg-type]
        noise=torch.zeros(1, 3, 2, 2),
        t_list=torch.tensor([0.999, 0.0]),
        caption_embs=torch.zeros(1, 1, 1),
        caption_mask=None,
        lq_latent=None,
        degrade_sigma=torch.zeros(1),
        pid_memory_optimization=pid_memory_optimization,
    )

    assert activation_chunk_sizes == [expected_chunk_size]


def test_pid_memory_optimization_defaults_to_disabled() -> None:
    assert PiDDecodeConfig().pid_memory_optimization is False


def test_working_memory_estimate_shrinks_when_the_optimization_is_enabled() -> None:
    """Wiring the flag into the decode but not into the estimate is worse than not wiring it at all.

    The cache takes `max(working_mem_bytes, device_working_mem_gb)` and subtracts it from the weight
    budget, so an estimate calibrated for the unoptimized peak withholds precisely the VRAM the
    optimization just freed, and PidNet partial-loads to CPU on the machines this feature targets.
    """
    latent = torch.zeros(1, 16, 64, 64)  # FLUX: 64 * 4 * 8 = 2048px output

    unoptimized = estimate_pid_decode_working_memory(latent, BaseModelType.Flux)
    optimized = estimate_pid_decode_working_memory(latent, BaseModelType.Flux, True)

    assert optimized < unoptimized
    # Measured peaks at 2048px on an RTX 4090: 3.68 GiB unoptimized, 1.50 GiB optimized. The estimates
    # must sit above their own peak (headroom) and below the other mode's (or the flag buys nothing).
    gib = 1024**3
    assert 1.50 * gib < optimized < 2.50 * gib
    assert 3.68 * gib < unoptimized < 4.50 * gib


def test_working_memory_estimate_keeps_a_fixed_term_for_the_chunk_working_set() -> None:
    """The optimized peak is not a pure multiple of the output size.

    Chunking bounds the per-block activations to a fixed working set, so halving the output area does
    not halve the peak (measured: 509 MiB at 1024px vs 1533 MiB at 2048px — a factor of 3.0, not 4).
    A pure scaling constant would therefore under-reserve at small sizes or over-reserve at large ones.
    """
    small = estimate_pid_decode_working_memory(torch.zeros(1, 16, 32, 32), BaseModelType.Flux, True)
    large = estimate_pid_decode_working_memory(torch.zeros(1, 16, 64, 64), BaseModelType.Flux, True)

    assert large < 4 * small, "the estimate scales purely with area — the fixed chunk term is missing"
    assert large > 2 * small, "the per-pixel term has been lost"


def test_working_memory_estimate_never_exceeds_the_unoptimized_one() -> None:
    """Below the chunk size the pixel blocks run unchunked, so the fixed chunk-working-set term must
    not be charged: a small output would otherwise reserve *more* with the optimization enabled than
    without it.
    """
    # 8 * 4 * 8 = 256px output -> 256 patch tokens, well under the 1024-token chunk size.
    small_latent = torch.zeros(1, 16, 8, 8)

    optimized = estimate_pid_decode_working_memory(small_latent, BaseModelType.Flux, True)
    unoptimized = estimate_pid_decode_working_memory(small_latent, BaseModelType.Flux)

    assert optimized <= unoptimized


def test_working_memory_estimate_still_returns_zero_for_unsupported_backbones() -> None:
    latent = torch.zeros(1, 16, 64, 64)
    for optimized in (False, True):
        assert estimate_pid_decode_working_memory(latent, BaseModelType.StableDiffusion1, optimized) == 0


def test_matching_decoder_base_is_accepted() -> None:
    assert_pid_decoder_matches_base(BaseModelType.Flux, BaseModelType.Flux, node_title="FLUX PiD Decode")


def test_mismatched_decoder_base_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires a"):
        assert_pid_decoder_matches_base(
            BaseModelType.StableDiffusion3, BaseModelType.Flux, node_title="FLUX PiD Decode"
        )


def test_z_image_node_accepts_flux_decoder() -> None:
    """Z-Image reuses the FLUX decoder, so its node passes node_base=FLUX and accepts a FLUX decoder."""
    assert_pid_decoder_matches_base(BaseModelType.Flux, BaseModelType.Flux, node_title="Z-Image PiD Decode")
