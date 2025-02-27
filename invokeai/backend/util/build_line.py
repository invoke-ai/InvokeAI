from typing import Callable


def build_line(x1: float, y1: float, x2: float, y2: float) -> Callable[[float], float]:
    """Build a linear function given two points on the line (x1, y1) and (x2, y2)."""
    return lambda x: (y2 - y1) / (x2 - x1) * (x - x1) + y1
