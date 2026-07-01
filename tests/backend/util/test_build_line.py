import math

import pytest

from invokeai.backend.util.build_line import build_line


@pytest.mark.parametrize(
    ["x1", "y1", "x2", "y2", "x3", "y3"],
    [
        (0, 0, 1, 1, 2, 2),  # y = x
        (0, 1, 1, 2, 2, 3),  # y = x + 1
        (0, 0, 1, 2, 2, 4),  # y = 2x
        (0, 1, 1, 0, 2, -1),  # y = -x + 1
        (0, 5, 1, 5, 2, 5),  # y = 0
    ],
)
def test_build_line(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    assert math.isclose(build_line(x1, y1, x2, y2)(x3), y3, rel_tol=1e-9)
