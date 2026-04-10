from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import (
    ModelIdentifierField,
    Qwen3EncoderField,
    T5EncoderField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.t5_model_identifier import (
    preprocess_t5_encoder_model_identifier,
    preprocess_t5_tokenizer_model_identifier,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("anima_model_loader_output")
class AnimaModelLoaderOutput(BaseInvocationOutput):
    """Anima model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_encoder: Qwen3EncoderField = OutputField(description=FieldDescriptions.qwen3_encoder, title="Qwen3 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    t5_encoder: T5EncoderField = OutputField(description=FieldDescriptions.t5_encoder, title="T5 Encoder")


@invocation(
    "anima_model_loader",
    title="Main Model - Anima",
    tags=["model", "anima"],
    category="model",
    version="1.3.0",
    classification=Classification.Prototype,
)
class AnimaModelLoaderInvocation(BaseInvocation):
    """Loads an Anima model, outputting its submodels.

    Anima uses:
    - Transformer: Cosmos Predict2 DiT + LLM Adapter (from single-file checkpoint)
    - Qwen3 Encoder: Qwen3 0.6B (standalone single-file)
    - VAE: AutoencoderKLQwenImage / Wan 2.1 VAE (standalone single-file or FLUX VAE)
    - T5 Encoder: T5-XXL model (only the tokenizer submodel is used, for LLM Adapter token IDs)
    """

    model: ModelIdentifierField = InputField(
        description="Anima main model (transformer + LLM adapter).",
        input=Input.Direct,
        ui_model_base=BaseModelType.Anima,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: ModelIdentifierField = InputField(
        description="Standalone VAE model. Anima uses a Wan 2.1 / QwenImage VAE (16-channel). "
        "A FLUX VAE can also be used as a compatible fallback.",
        input=Input.Direct,
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    qwen3_encoder_model: ModelIdentifierField = InputField(
        description="Standalone Qwen3 0.6B Encoder model.",
        input=Input.Direct,
        ui_model_type=ModelType.Qwen3Encoder,
        title="Qwen3 Encoder",
    )

    t5_encoder_model: ModelIdentifierField = InputField(
        description="T5-XXL encoder model. The tokenizer submodel is used for Anima text encoding.",
        input=Input.Direct,
        ui_model_type=ModelType.T5Encoder,
        title="T5 Encoder",
    )

    def invoke(self, context: InvocationContext) -> AnimaModelLoaderOutput:
        # Transformer always comes from the main model
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # VAE
        vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})

        # Qwen3 Encoder
        qwen3_tokenizer = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        qwen3_encoder = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        # T5 Encoder (only tokenizer submodel is used by Anima)
        t5_tokenizer = preprocess_t5_tokenizer_model_identifier(self.t5_encoder_model)
        t5_encoder = preprocess_t5_encoder_model_identifier(self.t5_encoder_model)

        return AnimaModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=qwen3_tokenizer, text_encoder=qwen3_encoder),
            vae=VAEField(vae=vae),
            t5_encoder=T5EncoderField(tokenizer=t5_tokenizer, text_encoder=t5_encoder, loras=[]),
        )
