from typing import Optional

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
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("anima_model_loader_output")
class AnimaModelLoaderOutput(BaseInvocationOutput):
    """Anima model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_encoder: Qwen3EncoderField = OutputField(description=FieldDescriptions.qwen3_encoder, title="Qwen3 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "anima_model_loader",
    title="Main Model - Anima",
    tags=["model", "anima"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class AnimaModelLoaderInvocation(BaseInvocation):
    """Loads an Anima model, outputting its submodels.

    Anima uses:
    - Transformer: Cosmos Predict2 DiT + LLM Adapter (from single-file checkpoint)
    - Qwen3 Encoder: Qwen3 0.6B (standalone single-file)
    - VAE: AutoencoderKLQwenImage / Wan 2.1 VAE (standalone single-file or FLUX VAE)
    """

    model: ModelIdentifierField = InputField(
        description="Anima main model (transformer + LLM adapter).",
        input=Input.Direct,
        ui_model_base=BaseModelType.Anima,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE model. Anima uses a Wan 2.1 / QwenImage VAE (16-channel). "
        "If not provided, a FLUX VAE can be used as a fallback.",
        input=Input.Direct,
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    qwen3_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen3 0.6B Encoder model.",
        input=Input.Direct,
        ui_model_type=ModelType.Qwen3Encoder,
        title="Qwen3 Encoder",
    )

    def invoke(self, context: InvocationContext) -> AnimaModelLoaderOutput:
        # Transformer always comes from the main model
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # VAE
        if self.vae_model is not None:
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No VAE source provided. Set 'VAE' to a compatible VAE model "
                "(Wan 2.1 QwenImage VAE or FLUX VAE)."
            )

        # Qwen3 Encoder
        if self.qwen3_encoder_model is not None:
            qwen3_tokenizer = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            qwen3_encoder = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                "No Qwen3 Encoder source provided. Set 'Qwen3 Encoder' to a Qwen3 0.6B model."
            )

        return AnimaModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=qwen3_tokenizer, text_encoder=qwen3_encoder),
            vae=VAEField(vae=vae),
        )
