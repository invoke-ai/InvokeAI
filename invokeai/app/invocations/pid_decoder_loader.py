from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField, PiDDecoderField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import ModelType


@invocation_output("pid_decoder_output")
class PiDDecoderOutput(BaseInvocationOutput):
    pid_decoder: PiDDecoderField = OutputField(
        description="PiD (Pixel Diffusion Decoder) checkpoint",
        title="PiD Decoder",
    )


@invocation(
    "pid_decoder_loader",
    title="PiD Decoder - FLUX / FLUX.2 / SD3",
    tags=["model", "pid", "decoder"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class PiDDecoderLoaderInvocation(BaseInvocation):
    """Loads a PiD decoder checkpoint, outputting a PiDDecoderField for use
    by the per-backbone PiD decode nodes."""

    pid_decoder_model: ModelIdentifierField = InputField(
        description="PiD decoder checkpoint matching the upstream backbone.",
        title="PiD Decoder",
        ui_model_type=ModelType.PiDDecoder,
    )

    def invoke(self, context: InvocationContext) -> PiDDecoderOutput:
        key = self.pid_decoder_model.key
        if not context.models.exists(key):
            raise Exception(f"Unknown PiD decoder: {key}")
        return PiDDecoderOutput(pid_decoder=PiDDecoderField(decoder=self.pid_decoder_model))
