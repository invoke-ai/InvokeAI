from typing import Annotated, Optional

from pydantic import BaseModel, Field, field_validator

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, UIComponent
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ideogram4.caption import build_ideogram4_caption

_IDEOGRAM4_COORD_MAX = 1000

# Exactly four normalized coordinates [y_min, x_min, y_max, x_max], each in 0..1000. Modeled as a
# constrained type so the generated OpenAPI schema advertises minItems/maxItems and per-item
# minimum/maximum — clients get the contract from the schema, not only from runtime validation.
Ideogram4Bbox = Annotated[
    list[Annotated[int, Field(ge=0, le=_IDEOGRAM4_COORD_MAX)]],
    Field(min_length=4, max_length=4),
]


class Ideogram4Region(BaseModel):
    """A single region of an Ideogram 4 structured caption (description + optional bounding box)."""

    prompt: str = Field(description="The region's description (becomes the element's `desc`).")
    bbox: Optional[Ideogram4Bbox] = Field(
        default=None,
        description="Normalized bounding box [y_min, x_min, y_max, x_max] (0–1000), or null for a region "
        "with no drawn content.",
    )

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        """Enforce the ordering the constrained type can't express: y_min <= y_max and x_min <= x_max.

        Length (exactly 4) and range (0..1000) are enforced by the Ideogram4Bbox type. The caption
        builder forwards the bbox verbatim into the structured JSON, so an inverted box would emit a
        malformed prompt the model may misapply — reject it here.
        """
        if v is None:
            return v
        y_min, x_min, y_max, x_max = v
        if y_min > y_max or x_min > x_max:
            raise ValueError(
                f"bbox must satisfy y_min <= y_max and x_min <= x_max, got [y_min={y_min}, x_min={x_min}, "
                f"y_max={y_max}, x_max={x_max}]"
            )
        return v


@invocation(
    "ideogram4_caption_builder",
    title="Caption Builder - Ideogram 4",
    tags=["prompt", "ideogram4"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Ideogram4CaptionBuilderInvocation(BaseInvocation):
    """Assembles the Ideogram 4 structured JSON caption at generation time.

    The caption is built here (not in the graph builder) so the batch-injectable global `prompt` — which
    dynamic prompts and prompt batching vary — is folded into the encoded caption. The regions and color
    palette are fixed per generation and supplied as inputs. If the prompt is already a JSON object it is
    passed through verbatim; otherwise it is always wrapped in the structured JSON schema (never bare
    plain text — Ideogram's safety filter false-positives far more on plain text).
    """

    prompt: str = InputField(
        default="",
        description="The global prompt (becomes `high_level_description`, or is used verbatim if it is "
        "already a JSON caption).",
        ui_component=UIComponent.Textarea,
    )
    regions: list[Ideogram4Region] = InputField(
        default=[],
        description="Regional descriptions and bounding boxes assembled from Canvas Regional Guidance layers.",
    )
    color_palette: list[str] = InputField(
        default=[],
        description="Optional color palette as hex colors (#RRGGBB).",
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        caption = build_ideogram4_caption(
            self.prompt,
            [(region.prompt, region.bbox) for region in self.regions],
            self.color_palette,
        )
        return StringOutput(value=caption)
