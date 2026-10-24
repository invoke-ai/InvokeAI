"""scripts/calibrate_z_image_working_memory.py: the forwards it plans.

The measurement needs a CUDA or ROCm device and runs by hand (see the script's docstring). The plumbing it shares with
the other calibration scripts -- the measuring process, its runtime, the lines and the point file, the records the
builder reads -- is `scripts/working_memory/fresh_points.py`'s, tested in `tests/test_fresh_points_working_memory.py`.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

pytestmark = pytest.mark.platform_independent

SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_z_image_working_memory.py"
MIB = 2**20


def _load_script():
    spec = spec_from_file_location("calibrate_z_image_working_memory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()
Point = script.Point

GRID = {
    "text_tokens": 512,
    "short_text_tokens": 77,
    "blocks": 2,
    "extra_blocks": 0,
    "control_blocks": 15,
}


def test_the_fit_grid_runs_every_size_in_every_dtype_and_the_short_caption_at_its_sizes() -> None:
    points = script.plan(
        "fit", sizes=[512, 1024, 1440], dtypes=["bfloat16", "float32"], short_text_sizes=[1024, 1440, 2048], **GRID
    )

    expected = [
        (dtype, px, text)
        for dtype in ("bfloat16", "float32")
        for px, text in ((512, 512), (1024, 512), (1440, 512), (1024, 77), (1440, 77))
    ]
    # 2048px runs no caption at all: it is not among the sizes.
    assert [(point.dtype, point.px, point.text_tokens) for point in points] == expected
    assert all(
        point.fit and point.blocks == 2 and not point.force_math and not point.control_blocks for point in points
    )


def test_extra_blocks_add_one_forward_outside_the_fit_at_1024px_in_the_first_dtype() -> None:
    points = script.plan(
        "fit", sizes=[512], dtypes=["float16", "bfloat16"], short_text_sizes=[], **(GRID | {"extra_blocks": 4})
    )

    assert [point for point in points if not point.fit] == [Point("blocks", 1024, 512, "float16", 4)]
    assert points[-1].kind == "blocks"


def test_each_check_pairs_its_forward_with_a_default_kernel_forward_in_every_dtype() -> None:
    dtypes = ["bfloat16", "float32"]
    checks = GRID | {"extra_blocks": 4, "control_blocks": 6}
    math = script.plan("math", sizes=[512], dtypes=dtypes, short_text_sizes=[], **checks)
    control = script.plan("control", sizes=[512], dtypes=dtypes, short_text_sizes=[], **checks)

    assert math == [
        Point("reference", 1024, 512, "bfloat16", 2),
        Point("math", 1024, 512, "bfloat16", 2, force_math=True),
        Point("reference", 1024, 512, "float32", 2),
        Point("math", 1024, 512, "float32", 2, force_math=True),
    ]
    assert control == [
        Point("reference", 1024, 512, "bfloat16", 2),
        Point("control", 1024, 512, "bfloat16", 2, control_blocks=6),
        Point("reference", 1024, 512, "float32", 2),
        Point("control", 1024, 512, "float32", 2, control_blocks=6),
    ]
    assert not any(point.fit for point in math + control)
