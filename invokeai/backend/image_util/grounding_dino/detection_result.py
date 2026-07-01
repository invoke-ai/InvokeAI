from pydantic import BaseModel, ConfigDict


class BoundingBox(BaseModel):
    """Bounding box helper class."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int


class DetectionResult(BaseModel):
    """Detection result from Grounding DINO."""

    score: float
    label: str
    box: BoundingBox
    model_config = ConfigDict(
        # Allow arbitrary types for mask, since it will be a numpy array.
        arbitrary_types_allowed=True
    )
