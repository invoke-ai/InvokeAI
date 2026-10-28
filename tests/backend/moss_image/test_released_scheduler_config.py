"""`get_moss_sigmas` builds the schedule from the vendored scheduler's defaults, not from a pipeline folder.

That holds only while the released `scheduler/scheduler_config.json` describes the same schedule. Its values are copied
here so a release that changes the shift or the training grid fails this test instead of quietly diverging.
"""

import pytest

from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageScheduler
from invokeai.backend.moss_image.sampling import get_moss_sigmas

# `OpenMOSS-Team/MOSS-Image-Base` `scheduler/scheduler_config.json` at revision
# 1ceede80badddd6a7812069619edaa5633587192, verbatim. MOSS-Image-SFT ships the identical file.
RELEASED_SCHEDULER_CONFIG = {
    "_class_name": "MOSSImageScheduler",
    "_diffusers_version": "0.38.0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 5.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "training_shift": 1.0,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


@pytest.mark.parametrize("num_steps", [1, 8, 50, 250])
def test_the_released_scheduler_config_gives_the_schedule_the_defaults_give(num_steps: int) -> None:
    released = MOSSImageScheduler.from_config(RELEASED_SCHEDULER_CONFIG)
    released.set_timesteps(num_steps)

    assert released.sigmas.tolist() == get_moss_sigmas(num_steps)
