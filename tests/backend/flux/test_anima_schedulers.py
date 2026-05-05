"""Tests for Anima scheduler registry and ancestral-Euler helper."""

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from invokeai.backend.flux.schedulers import (
    ANIMA_SCHEDULER_LABELS,
    ANIMA_SCHEDULER_MAP,
    ANIMA_SCHEDULER_NAME_VALUES,
)
from invokeai.app.invocations.anima_denoise import _anima_euler_ancestral_step


def test_anima_scheduler_map_entries_are_class_kwargs_tuples():
    """Every entry must be (SchedulerClass, kwargs_dict)."""
    for name, entry in ANIMA_SCHEDULER_MAP.items():
        assert isinstance(entry, tuple), f"{name} is not a tuple"
        assert len(entry) == 2, f"{name} tuple has wrong arity"
        cls, kwargs = entry
        assert isinstance(cls, type) and issubclass(cls, SchedulerMixin), (
            f"{name} first element is not a SchedulerMixin subclass"
        )
        assert isinstance(kwargs, dict), f"{name} second element is not a dict"


def test_anima_scheduler_map_entries_can_be_constructed():
    """Every entry must construct cleanly by splatting its kwargs."""
    for name, (cls, kwargs) in ANIMA_SCHEDULER_MAP.items():
        scheduler = cls(num_train_timesteps=1000, **kwargs)
        assert isinstance(scheduler, SchedulerMixin), f"{name} did not produce a SchedulerMixin"


def test_anima_scheduler_labels_cover_every_map_key():
    for name in ANIMA_SCHEDULER_MAP.keys():
        assert name in ANIMA_SCHEDULER_LABELS, f"{name} has no label"


def test_anima_scheduler_map_includes_new_dpmpp_entries():
    assert "dpmpp_2m" in ANIMA_SCHEDULER_MAP
    assert "dpmpp_2m_sde" in ANIMA_SCHEDULER_MAP


def test_anima_dpmpp_2m_uses_flow_prediction():
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT

    cls, kwargs = ANIMA_SCHEDULER_MAP["dpmpp_2m"]
    assert kwargs["prediction_type"] == "flow_prediction"
    assert kwargs["use_flow_sigmas"] is True
    assert kwargs["flow_shift"] == ANIMA_SHIFT
    assert kwargs["solver_order"] == 2
    assert "algorithm_type" not in kwargs  # deterministic, default algorithm


def test_anima_dpmpp_2m_sde_uses_sde_algorithm():
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT

    cls, kwargs = ANIMA_SCHEDULER_MAP["dpmpp_2m_sde"]
    assert kwargs["prediction_type"] == "flow_prediction"
    assert kwargs["use_flow_sigmas"] is True
    assert kwargs["flow_shift"] == ANIMA_SHIFT
    assert kwargs["algorithm_type"] == "sde-dpmsolver++"
    assert kwargs["solver_order"] == 2


def test_anima_dpmpp_2m_produces_anima_compatible_sigma_schedule():
    """The DPM++ 2M scheduler, when run through the same dispatch logic as
    anima_denoise._run_diffusion, must produce a sigma schedule equivalent
    to Anima's reference schedule (loglinear_timestep_shift with shift=3.0).

    On diffusers 0.35.1, DPMSolverMultistepScheduler.set_timesteps does not
    accept `sigmas=`, so the runtime falls back to num_inference_steps and
    relies on the scheduler's internal flow_shift=3.0 to compute equivalent
    sigmas. This test verifies that equivalence end-to-end.
    """
    import inspect
    from invokeai.app.invocations.anima_denoise import loglinear_timestep_shift
    from invokeai.backend.flux.schedulers import ANIMA_SHIFT

    num_steps = 10

    # Reference: Anima's own pre-shifted sigma schedule.
    anima_sigmas = [
        loglinear_timestep_shift(ANIMA_SHIFT, 1.0 - i / num_steps)
        for i in range(num_steps + 1)
    ]

    cls, kwargs = ANIMA_SCHEDULER_MAP["dpmpp_2m"]
    scheduler = cls(num_train_timesteps=1000, **kwargs)

    # Mirror anima_denoise.py:502-506 dispatch.
    sig = inspect.signature(scheduler.set_timesteps)
    if "sigmas" in sig.parameters:
        scheduler.set_timesteps(sigmas=anima_sigmas, device="cpu")
    else:
        scheduler.set_timesteps(num_inference_steps=num_steps, device="cpu")

    diffusers_sigmas = [float(s) for s in scheduler.sigmas[: len(anima_sigmas)]]
    max_diff = max(abs(a - b) for a, b in zip(anima_sigmas, diffusers_sigmas, strict=True))
    assert max_diff < 1e-3, (
        f"DPM++ 2M sigma schedule diverges from Anima reference (max abs diff = {max_diff:.6f})"
    )


def test_anima_literal_covers_every_map_key():
    """Catch the silent failure mode where a new entry lands in the map but
    the Literal isn't updated — Pydantic validation would still accept it
    via runtime introspection but type-check tooling would not."""
    import typing

    literal_values = set(typing.get_args(ANIMA_SCHEDULER_NAME_VALUES))
    for name in ANIMA_SCHEDULER_MAP:
        assert name in literal_values, f"{name} is in the map but missing from the Literal"


def _deterministic_euler_step(latents, v, sigma_curr, sigma_prev):
    """Reference deterministic Euler step (matches the built-in Euler loop)."""
    return latents + (sigma_prev - sigma_curr) * v


def test_anima_euler_ancestral_step_eta_zero_matches_deterministic_euler():
    """With eta=0, the ancestral step must be byte-identical to deterministic Euler."""
    torch.manual_seed(0)
    latents = torch.randn(1, 16, 1, 8, 8)
    v = torch.randn(1, 16, 1, 8, 8)
    noise = torch.randn_like(latents)
    sigma_curr, sigma_prev = 0.6, 0.4

    deterministic = _deterministic_euler_step(latents, v, sigma_curr, sigma_prev)
    ancestral_eta_zero = _anima_euler_ancestral_step(
        latents=latents, v=v, sigma_curr=sigma_curr, sigma_prev=sigma_prev,
        noise=noise, eta=0.0, s_noise=1.0,
    )
    assert torch.allclose(deterministic, ancestral_eta_zero, atol=1e-6)


def test_anima_euler_ancestral_step_eta_one_differs_from_deterministic():
    """With eta=1, output must differ from deterministic Euler (noise was injected)."""
    torch.manual_seed(0)
    latents = torch.randn(1, 16, 1, 8, 8)
    v = torch.randn(1, 16, 1, 8, 8)
    noise = torch.randn_like(latents)
    sigma_curr, sigma_prev = 0.6, 0.4

    deterministic = _deterministic_euler_step(latents, v, sigma_curr, sigma_prev)
    ancestral_eta_one = _anima_euler_ancestral_step(
        latents=latents, v=v, sigma_curr=sigma_curr, sigma_prev=sigma_prev,
        noise=noise, eta=1.0, s_noise=1.0,
    )
    assert not torch.allclose(deterministic, ancestral_eta_one, atol=1e-3)


def test_anima_euler_ancestral_step_preserves_shape_and_dtype():
    """Output must have the same shape and dtype as the input latents — no
    silent dtype promotion (e.g., float32 → float64) and no shape changes."""
    latents = torch.randn(1, 16, 1, 8, 8, dtype=torch.float32)
    v = torch.randn_like(latents)
    noise = torch.randn_like(latents)
    out = _anima_euler_ancestral_step(latents, v, 0.6, 0.4, noise, eta=1.0, s_noise=1.0)
    assert out.shape == latents.shape
    assert out.dtype == latents.dtype


def test_anima_euler_ancestral_step_at_final_step_is_clean():
    """When sigma_prev == 0, no noise should be injected even with eta=1."""
    latents = torch.randn(1, 16, 1, 8, 8)
    v = torch.randn_like(latents)
    noise = torch.randn_like(latents)
    sigma_curr, sigma_prev = 0.05, 0.0
    out = _anima_euler_ancestral_step(latents, v, sigma_curr, sigma_prev, noise, eta=1.0, s_noise=1.0)
    deterministic = _deterministic_euler_step(latents, v, sigma_curr, sigma_prev)
    assert torch.allclose(out, deterministic, atol=1e-6)
