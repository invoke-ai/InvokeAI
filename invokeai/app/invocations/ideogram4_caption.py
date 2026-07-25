from typing import Optional

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, UIComponent
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ideogram4.caption import build_ideogram4_caption


class Ideogram4Region(BaseModel):
    """A single region of an Ideogram 4 structured caption (description + optional bounding box)."""

    prompt: str = Field(description="The region's description (becomes the element's `desc`).")
    bbox: Optional[list[int]] = Field(
        default=None,
        description="Normalized bounding box [y_min, x_min, y_max, x_max] (0–1000), or null for a region "
        "with no drawn content.",
    )


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
    passed through verbatim; with no regions or palette it falls back to the plain prompt.
    """

    prompt: str = InputField(
        default="",
        description="The global prompt (becomes `high_level_description`, or is used verbatim if it is "
        "already a JSON caption, or as plain text).",
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
