from typing import Any, Optional

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class BoundingBox(BaseModel):
    """Bounding box helper class."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_box(self) -> list[int]:
        """Convert to the array notation expected by SAM."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]


class DetectionResult(BaseModel):
    """Detection result from Grounding DINO or Grounded SAM."""

    score: float
    label: str
    box: BoundingBox
    mask: Optional[npt.NDArray[Any]] = None
    model_config = ConfigDict(
        # Allow arbitrary types for mask, since it will be a numpy array.
        arbitrary_types_allowed=True
    )
