from invokeai.app.invocations.baseinvocation import BaseInvocation, Bottleneck, Classification, invocation
from invokeai.app.invocations.fields import InputField, UIComponent
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.external_generation.minimax_context_ir import MiniMaxTextRatio, expand_minimax_prompt
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "minimax_h3_context_ir",
    title="MiniMax H3 Context-IR (API)",
    tags=["minimax", "video", "prompt", "external"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
    bottleneck=Bottleneck.Network,
    use_cache=False,
)
class MiniMaxH3ContextIRInvocation(BaseInvocation):
    """Expand a video prompt with the paid MiniMax H3 Context-IR API. Connect the text to Prompt - MiniMax H3."""

    prompt: str = InputField(
        description="The video or sequence of shots to describe.",
        ui_component=UIComponent.Textarea,
        min_length=1,
        max_length=7000,
    )
    duration: int = InputField(default=5, ge=4, le=15, description="Intended video duration in seconds.")
    ratio: MiniMaxTextRatio = InputField(
        default="16:9", title="Aspect Ratio", description="Intended video aspect ratio."
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        config = context.config.get()
        return StringOutput(
            value=expand_minimax_prompt(
                prompt=self.prompt,
                duration=self.duration,
                ratio=self.ratio,
                api_key=config.external_minimax_api_key,
                base_url=config.external_minimax_base_url,
                is_canceled=context.util.is_canceled,
            )
        )
