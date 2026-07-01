from typing import get_args

from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_MAP, SCHEDULER_NAME_VALUES


def test_scheduler_map_has_all_keys():
    # Assert that SCHEDULER_MAP has all keys from SCHEDULER_NAME_VALUES.
    # TODO(ryand): This feels like it should be a type check, but I couldn't find a clean way to do this and didn't want
    # to spend more time on it.
    assert set(SCHEDULER_MAP.keys()) == set(get_args(SCHEDULER_NAME_VALUES))
