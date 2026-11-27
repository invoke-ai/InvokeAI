"""Every scheduler a denoise node will accept is one it can actually build.

`FlowMatchLCMScheduler` is imported under `try: ... except ImportError`, so `"lcm"` lands in each
`*_SCHEDULER_MAP` only when the installed diffusers has it. The `*_SCHEDULER_NAME_VALUES` literals
accept it unconditionally, and webv2 offers it unconditionally — so on a diffusers without it the
node validates the request, queues the graph, and raises `KeyError: 'lcm'` out of its
`MAP[self.scheduler]` lookup partway through generation. The user sees a failed queue item, not a
rejected setting.

This is the assertion that turns that into a red test on the machine that builds the wheel rather
than a bug report from whoever installs it. `test_anima_schedulers.py` already covers the opposite
direction (every map key is in the literal), which catches a scheduler that ships without a type
but not one whose type ships without the scheduler.
"""

import typing

import pytest

from invokeai.backend.flux.schedulers import (
    ANIMA_SCHEDULER_LABELS,
    ANIMA_SCHEDULER_MAP,
    ANIMA_SCHEDULER_NAME_VALUES,
    ERNIE_IMAGE_SCHEDULER_LABELS,
    ERNIE_IMAGE_SCHEDULER_MAP,
    ERNIE_IMAGE_SCHEDULER_NAME_VALUES,
    FLUX_SCHEDULER_LABELS,
    FLUX_SCHEDULER_MAP,
    FLUX_SCHEDULER_NAME_VALUES,
    ZIMAGE_SCHEDULER_LABELS,
    ZIMAGE_SCHEDULER_MAP,
    ZIMAGE_SCHEDULER_NAME_VALUES,
)

SCHEDULER_FAMILIES = [
    ("flux", FLUX_SCHEDULER_NAME_VALUES, FLUX_SCHEDULER_MAP, FLUX_SCHEDULER_LABELS),
    ("z-image", ZIMAGE_SCHEDULER_NAME_VALUES, ZIMAGE_SCHEDULER_MAP, ZIMAGE_SCHEDULER_LABELS),
    ("ernie-image", ERNIE_IMAGE_SCHEDULER_NAME_VALUES, ERNIE_IMAGE_SCHEDULER_MAP, ERNIE_IMAGE_SCHEDULER_LABELS),
    ("anima", ANIMA_SCHEDULER_NAME_VALUES, ANIMA_SCHEDULER_MAP, ANIMA_SCHEDULER_LABELS),
]


@pytest.mark.parametrize(
    ("names", "scheduler_map"),
    [pytest.param(family[1], family[2], id=family[0]) for family in SCHEDULER_FAMILIES],
)
def test_every_accepted_scheduler_name_can_be_built(names: object, scheduler_map: dict[str, object]) -> None:
    unbuildable = sorted(set(typing.get_args(names)) - set(scheduler_map))
    assert unbuildable == [], (
        f"accepted by the node's field but absent from its scheduler map: {unbuildable}. "
        f"The lookup is unguarded, so choosing one fails mid-generation."
    )


@pytest.mark.parametrize(
    ("names", "labels"),
    [pytest.param(family[1], family[3], id=family[0]) for family in SCHEDULER_FAMILIES],
)
def test_every_accepted_scheduler_name_has_a_label(names: object, labels: dict[str, str]) -> None:
    """The UI renders the label; a missing one shows the raw key or nothing at all."""
    unlabelled = sorted(set(typing.get_args(names)) - set(labels))
    assert unlabelled == []
