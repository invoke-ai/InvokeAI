from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, UIComponent
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux3.layout import Flux3LayoutRegion, build_flux3_layout_prompt


@invocation(
    "flux3_layout_prompt",
    title="Layout Prompt - FLUX.3",
    tags=["prompt", "layout", "flux3", "flux"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux3LayoutPromptInvocation(BaseInvocation):
    """Places regions on a FLUX.3 prompt as boxes, at generation time.

    The boxes are added here rather than by the graph builder so the prompt that dynamic prompts and batching vary is
    the one laid out. Without regions the prompt passes through unchanged; a prompt that already ends with a box table
    gets the regions appended to it, numbered after its highest `obj_N` id.
    """

    prompt: str = InputField(
        default="",
        description="The image prompt: the caption the boxes follow, optionally ending with a box table of its own.",
        ui_component=UIComponent.Textarea,
    )
    regions: list[Flux3LayoutRegion] = InputField(
        default=[],
        description="Boxes on the 0-1000 grid across the image, assembled from Canvas Regional Guidance layers.",
    )
    edit_targets: bool = InputField(
        default=False,
        description="Write each region as an edit row that places new content at its box in the output "
        '("from": "target"), the rows FLUX.3 reads when it edits reference images. Off: object rows.',
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=build_flux3_layout_prompt(self.prompt, self.regions, edit_targets=self.edit_targets))
