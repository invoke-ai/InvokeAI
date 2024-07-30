from dataclasses import dataclass
from typing import Any, Optional

import numpy.typing as npt


@dataclass
class BoundingBox:
    """Bounding box helper class."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_box(self) -> list[int]:
        """Convert to the array notation expected by SAM."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    """Detection result from Grounding DINO or Grounded SAM."""

    score: float
    label: str
    box: BoundingBox
    mask: Optional[npt.NDArray[Any]] = None

    @classmethod
    def from_dict(cls, detection_dict: dict[str, Any]):
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )
