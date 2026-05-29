from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.model import Gemma2EncoderField, ModelIdentifierField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("gemma2_encoder_output")
class Gemma2EncoderOutput(BaseInvocationOutput):
    gemma2_encoder: Gemma2EncoderField = OutputField(
        description="Gemma-2 text encoder used by PiD decoders",
        title="Gemma-2 Encoder",
    )


@invocation(
    "gemma2_encoder_loader",
    title="Gemma-2 Encoder - PiD",
    tags=["model", "gemma2", "pid"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Gemma2EncoderLoaderInvocation(BaseInvocation):
    """Loads a Gemma-2 causal LM directory and exposes its tokenizer + decoder
    submodels for use by a PiD decode node."""

    gemma2_model: ModelIdentifierField = InputField(
        description="Gemma-2 model used to encode captions for PiD decoders.",
        title="Gemma-2",
        ui_model_base=[BaseModelType.Any],
        ui_model_type=ModelType.Gemma2Encoder,
    )

    def invoke(self, context: InvocationContext) -> Gemma2EncoderOutput:
        key = self.gemma2_model.key
        if not context.models.exists(key):
            raise Exception(f"Unknown Gemma2 model: {key}")

        tokenizer = self.gemma2_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.gemma2_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        return Gemma2EncoderOutput(
            gemma2_encoder=Gemma2EncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
        )
