from pydantic import ValidationInfo, field_validator

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation_output("image_panel_coordinate_output")
class ImagePanelCoordinateOutput(BaseInvocationOutput):
    x_left: int = OutputField(description="The left x-coordinate of the panel.")
    y_top: int = OutputField(description="The top y-coordinate of the panel.")
    width: int = OutputField(description="The width of the panel.")
    height: int = OutputField(description="The height of the panel.")


@invocation(
    "image_panel_layout",
    title="Image Panel Layout",
    tags=["image", "panel", "layout"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ImagePanelLayoutInvocation(BaseInvocation):
    """Get the coordinates of a single panel in a grid. (If the full image shape cannot be divided evenly into panels,
    then the grid may not cover the entire image.)
    """

    width: int = InputField(description="The width of the entire grid.")
    height: int = InputField(description="The height of the entire grid.")
    num_cols: int = InputField(ge=1, default=1, description="The number of columns in the grid.")
    num_rows: int = InputField(ge=1, default=1, description="The number of rows in the grid.")
    panel_col_idx: int = InputField(ge=0, default=0, description="The column index of the panel to be processed.")
    panel_row_idx: int = InputField(ge=0, default=0, description="The row index of the panel to be processed.")

    @field_validator("panel_col_idx")
    def validate_panel_col_idx(cls, v: int, info: ValidationInfo) -> int:
        if v < 0 or v >= info.data["num_cols"]:
            raise ValueError(f"panel_col_idx must be between 0 and {info.data['num_cols'] - 1}")
        return v

    @field_validator("panel_row_idx")
    def validate_panel_row_idx(cls, v: int, info: ValidationInfo) -> int:
        if v < 0 or v >= info.data["num_rows"]:
            raise ValueError(f"panel_row_idx must be between 0 and {info.data['num_rows'] - 1}")
        return v

    def invoke(self, context: InvocationContext) -> ImagePanelCoordinateOutput:
        x_left = self.panel_col_idx * (self.width // self.num_cols)
        y_top = self.panel_row_idx * (self.height // self.num_rows)
        width = self.width // self.num_cols
        height = self.height // self.num_rows
        return ImagePanelCoordinateOutput(x_left=x_left, y_top=y_top, width=width, height=height)
