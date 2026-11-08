"""The shared schedule-shift fit: its line, its hold, its two forms, and where it reads its anchors from."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from invokeai.backend.rectified_flow.schedule_shift import ShiftFit, load_scheduler, time_shift

ONE_MEGAPIXEL_FIT = ShiftFit(base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
# A checkpoint's config may make the line fall. Holding the sequence length, not mu, keeps the hold at its end.
FALLING_FIT = ShiftFit(base_seq_len=256, max_seq_len=1024, base_shift=1.5, max_shift=0.3)
HELD_BOTH_ENDS_FIT = ShiftFit(
    base_seq_len=1024, max_seq_len=4096, base_shift=0.95, max_shift=2.05, hold_below_base=True
)


@pytest.mark.parametrize(
    ("fit", "seq_len", "mu"),
    [
        (ONE_MEGAPIXEL_FIT, 1, 0.5 - 0.65 * 255 / 3840),  # below the lower anchor the line keeps running
        (ONE_MEGAPIXEL_FIT, 256, 0.5),
        (ONE_MEGAPIXEL_FIT, 1024, 0.63),
        (ONE_MEGAPIXEL_FIT, 4095, 0.5 + 0.65 * 3839 / 3840),
        (ONE_MEGAPIXEL_FIT, 4096, 1.15),
        (ONE_MEGAPIXEL_FIT, 4097, 1.15),  # held from the upper anchor on
        (ONE_MEGAPIXEL_FIT, 65536, 1.15),  # 11.55 extrapolated
        (FALLING_FIT, 512, 1.5 - 1.2 * 256 / 768),
        (FALLING_FIT, 4096, 0.3),
        (HELD_BOTH_ENDS_FIT, 1, 0.95),  # held below the lower anchor too
        (HELD_BOTH_ENDS_FIT, 1024, 0.95),
        (HELD_BOTH_ENDS_FIT, 2560, 1.5),
        (HELD_BOTH_ENDS_FIT, 13728, 2.05),
    ],
)
def test_mu_follows_the_line_through_the_anchors_and_holds_where_it_ends(
    fit: ShiftFit, seq_len: int, mu: float
) -> None:
    assert fit.mu(seq_len) == pytest.approx(mu, abs=1e-12)


def test_the_linear_form_takes_e_to_the_mu() -> None:
    assert ONE_MEGAPIXEL_FIT.shift(65536) == pytest.approx(3.158192909689768, abs=1e-12)  # e^1.15


def test_the_time_shift_keeps_the_ends_and_moves_the_middle_towards_noise() -> None:
    t = torch.tensor([1.0, 0.5, 0.25, 0.0], dtype=torch.float64)
    torch.testing.assert_close(time_shift(3.0, t), torch.tensor([1.0, 0.75, 0.5, 0.0], dtype=torch.float64))
    assert time_shift(3.0, 0.25) == 0.5


def test_the_linear_form_of_e_to_the_mu_is_the_schedulers_exponential_shift() -> None:
    mu = 1.15
    sigmas = np.linspace(1.0, 1 / 8, 8).tolist()
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, time_shift_type="exponential")
    scheduler.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")

    expected = time_shift(math.exp(mu), torch.tensor(sigmas, dtype=torch.float64)).float()
    torch.testing.assert_close(scheduler.sigmas[:-1], expected)


STOCK = {
    "use_dynamic_shifting": True,
    "base_image_seq_len": 256,
    "max_image_seq_len": 8192,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "time_shift_type": "exponential",
}


STOCK_FIT = ShiftFit(base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9)


@pytest.mark.parametrize(
    ("scheduler_folder", "checkpoint_config", "fit"),
    [
        # GGUF and single-file transformers ship no scheduler/ folder.
        (False, None, STOCK_FIT),
        (True, None, STOCK_FIT),
        (
            True,
            {"base_image_seq_len": 100, "max_image_seq_len": 1000, "base_shift": 0.3, "max_shift": 1.5},
            ShiftFit(base_seq_len=100, max_seq_len=1000, base_shift=0.3, max_shift=1.5),
        ),
        # A key the checkpoint omits takes diffusers' default, not the stock value: 4096 and 1.15, not 8192 and 0.9.
        (
            True,
            {"base_image_seq_len": 100, "base_shift": 0.3},
            ShiftFit(base_seq_len=100, max_seq_len=4096, base_shift=0.3, max_shift=1.15),
        ),
    ],
    ids=["no-scheduler-folder", "folder-without-config", "checkpoint-config", "checkpoint-config-missing-keys"],
)
def test_the_fit_comes_from_the_checkpoints_scheduler_config_else_the_stock_one(
    tmp_path: Path, scheduler_folder: bool, checkpoint_config: dict | None, fit: ShiftFit
) -> None:
    if scheduler_folder:
        (tmp_path / "scheduler").mkdir()
    if checkpoint_config is not None:
        config = {"_class_name": "FlowMatchEulerDiscreteScheduler", "use_dynamic_shifting": True, **checkpoint_config}
        (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps(config))

    assert ShiftFit.from_scheduler_config(load_scheduler(tmp_path, STOCK).config) == fit
