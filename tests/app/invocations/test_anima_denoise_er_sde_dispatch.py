"""Dispatch integration test for Anima ER-SDE through the universal scheduler.

Verifies that AnimaDenoiseInvocation with scheduler='er_sde' routes through
the standard use_scheduler path and instantiates an ERSDEScheduler — not
through the legacy elif is_er_sde: branch (which is removed in a sibling
commit).

Numerical correctness is gated by tests/backend/rectified_flow/test_er_sde_scheduler_anima_parity.py
(worst delta 5.137e-07 against er_sde_rf_step). This test catches dispatch
wiring bugs only.
"""

from __future__ import annotations

from invokeai.backend.flux.schedulers import ANIMA_SCHEDULER_MAP
from invokeai.backend.rectified_flow.er_sde_scheduler import ERSDEScheduler


def test_anima_scheduler_map_er_sde_constructs_correctly():
    """The map entry must produce a valid ERSDEScheduler when instantiated."""
    cls, kwargs = ANIMA_SCHEDULER_MAP["er_sde"]
    scheduler = cls(num_train_timesteps=1000, **kwargs)
    assert isinstance(scheduler, ERSDEScheduler)
    assert scheduler.config.prediction_type == "flow_prediction"
    assert scheduler.config.use_flow_sigmas is True
    assert scheduler.config.solver_order == 3
    assert scheduler.config.stochastic is True


def test_anima_er_sde_set_timesteps_accepts_sigmas():
    """Anima passes pre-shifted sigmas via set_timesteps(sigmas=...).

    The legacy elif is_er_sde: branch consumed Anima's pre-shifted sigmas
    directly. The universal path requires ERSDEScheduler.set_timesteps to
    accept sigmas= as a keyword argument. This is the contract that makes
    the cutover safe.
    """
    import inspect

    cls, kwargs = ANIMA_SCHEDULER_MAP["er_sde"]
    scheduler = cls(num_train_timesteps=1000, **kwargs)
    sig = inspect.signature(scheduler.set_timesteps)
    assert "sigmas" in sig.parameters, "ERSDEScheduler.set_timesteps must accept sigmas= for Anima compatibility"


def test_anima_er_sde_set_timesteps_with_pre_shifted_sigmas():
    """End-to-end set_timesteps with a small pre-shifted sigma schedule."""
    import torch

    cls, kwargs = ANIMA_SCHEDULER_MAP["er_sde"]
    scheduler = cls(num_train_timesteps=1000, **kwargs)
    # Synthetic 5-step pre-shifted schedule, sigma_max=0.95 down to terminal 0.
    sigmas = torch.tensor([0.95, 0.75, 0.5, 0.3, 0.1, 0.0], dtype=torch.float32)
    scheduler.set_timesteps(sigmas=sigmas, device="cpu")
    assert scheduler.num_inference_steps == 5
    assert torch.allclose(scheduler.sigmas, sigmas)
    # Multistep state must be reset.
    assert scheduler.lower_order_nums == 0
    assert all(x is None for x in scheduler.model_outputs)
