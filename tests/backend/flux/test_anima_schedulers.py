"""Tests for Anima scheduler registry and ancestral-Euler helper."""

from diffusers.schedulers.scheduling_utils import SchedulerMixin

from invokeai.backend.flux.schedulers import (
    ANIMA_SCHEDULER_LABELS,
    ANIMA_SCHEDULER_MAP,
    ANIMA_SCHEDULER_NAME_VALUES,
)


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
