from enum import Enum

from pydantic import BaseModel, model_validator
from pydantic.fields import Field


class BoundingBox(BaseModel):
    x_min: int = Field(..., description="The minimum x-coordinate of the bounding box (inclusive).")
    x_max: int = Field(..., description="The maximum x-coordinate of the bounding box (exclusive).")
    y_min: int = Field(..., description="The minimum y-coordinate of the bounding box (inclusive).")
    y_max: int = Field(..., description="The maximum y-coordinate of the bounding box (exclusive).")

    @model_validator(mode="after")
    def check_coords(self):
        if self.x_min > self.x_max:
            raise ValueError(f"x_min ({self.x_min}) is greater than x_max ({self.x_max}).")
        if self.y_min > self.y_max:
            raise ValueError(f"y_min ({self.y_min}) is greater than y_max ({self.y_max}).")
        return self

    def tuple(self) -> tuple[int, int, int, int]:
        """
        Returns the bounding box as a tuple suitable for use with PIL's `Image.crop()` method.
        This method returns a tuple of the form (left, upper, right, lower) == (x_min, y_min, x_max, y_max).
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)


class SAMPointLabel(Enum):
    negative = -1
    neutral = 0
    positive = 1


class SAMPoint(BaseModel):
    x: int = Field(..., description="The x-coordinate of the point")
    y: int = Field(..., description="The y-coordinate of the point")
    label: SAMPointLabel = Field(..., description="The label of the point")


class SAMInput(BaseModel):
    bounding_box: BoundingBox | None = Field(None, description="The bounding box to use for segmentation")
    points: list[SAMPoint] | None = Field(None, description="The points to use for segmentation")

    @model_validator(mode="after")
    def check_input(self):
        if not self.bounding_box and not self.points:
            raise ValueError("Either bounding_box or points must be provided")
        return self
