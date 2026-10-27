from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, UIComponent
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.moss_image.layout import MossImageLayoutRegion, build_moss_image_layout_prompt


@invocation(
    "moss_image_layout_prompt",
    title="Layout Prompt - MOSS-Image",
    tags=["prompt", "layout", "moss_image", "moss-image"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MossImageLayoutPromptInvocation(BaseInvocation):
    """Places regions on a MOSS-Image prompt as a JSON layout, at generation time.

    The layout is built here rather than by the graph builder so the prompt that dynamic prompts and batching vary is
    the one laid out. Without regions the prompt passes through unchanged; a prompt that already is a layout gets the
    regions appended to its items.
    """

    prompt: str = InputField(
        default="",
        description="The image prompt: plain text becomes the layout's summary caption, a compiled layout gets the "
        "regions appended.",
        ui_component=UIComponent.Textarea,
    )
    regions: list[MossImageLayoutRegion] = InputField(
        default=[],
        description="Layout elements, in pixels of the width x height image, assembled from Canvas Regional Guidance "
        "layers.",
    )
    width: int = InputField(default=1024, gt=0, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, gt=0, multiple_of=16, description="Height of the generated image.")

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(
            value=build_moss_image_layout_prompt(self.prompt, self.regions, width=self.width, height=self.height)
        )
