"""Qwen-Image-2.1's sigma schedules: resolution-shifted for the base model, a fixed table for Turbo."""

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from invokeai.backend.model_manager.taxonomy import QwenImage21VariantType

# scheduler/scheduler_config.json of Qwen/Qwen-Image-2.1.
BASE_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Qwen/Qwen-Image-2.1-Turbo samples its own table unshifted: the same config with dynamic shifting and the
# terminal shift turned off, and model_index.json's `sample_sigmas`.
TURBO_SCHEDULER_CONFIG = {**BASE_SCHEDULER_CONFIG, "shift_terminal": None, "use_dynamic_shifting": False}
TURBO_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


def calculate_mu(image_seq_len: int) -> float:
    """The pipeline's `calculate_shift` over target-image tokens only (condition images are not counted)."""
    c = BASE_SCHEDULER_CONFIG
    m = (c["max_shift"] - c["base_shift"]) / (c["max_image_seq_len"] - c["base_image_seq_len"])
    return image_seq_len * m + (c["base_shift"] - m * c["base_image_seq_len"])


def turbo_sigmas(steps: int) -> list[float]:
    """Turbo's table, or the table resampled along its index for another step count.

    Only the shipped 8 steps are evaluated by its authors. Resampling keeps the table's shape and its
    start at 1.0, which is what lets fewer or more steps still land in the noise range it was distilled for.
    """
    if steps == len(TURBO_SIGMAS):
        return list(TURBO_SIGMAS)
    positions = np.linspace(0, len(TURBO_SIGMAS) - 1, steps)
    return np.interp(positions, np.arange(len(TURBO_SIGMAS)), TURBO_SIGMAS).tolist()


def build_sigmas(variant: QwenImage21VariantType | None, steps: int, image_seq_len: int) -> torch.Tensor:
    """The full schedule, `steps + 1` sigmas ending in 0, exactly as `QwenImage21Pipeline` samples it.

    `image_seq_len` is the target image's latent token count, `(height // 16) * (width // 16)`; only the base
    model's shift reads it.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if variant is QwenImage21VariantType.Turbo:
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(TURBO_SCHEDULER_CONFIG)
        scheduler.set_timesteps(sigmas=turbo_sigmas(steps), device="cpu")
    elif steps == 1:
        # One step is the whole range; the pipeline's stretch to `shift_terminal` divides by zero on it.
        return torch.tensor([1.0, 0.0])
    else:
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(BASE_SCHEDULER_CONFIG)
        scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 1 / steps, steps).tolist(), mu=calculate_mu(image_seq_len), device="cpu"
        )
    return scheduler.sigmas.float()
