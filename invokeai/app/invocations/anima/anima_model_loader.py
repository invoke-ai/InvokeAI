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
    Qwen35EncoderField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.architectures import accepted_vae_bases
from invokeai.backend.model_manager.taxonomy import AnimaVariantType, BaseModelType, ModelType, SubModelType


@invocation_output("anima_model_loader_output")
class AnimaModelLoaderOutput(BaseInvocationOutput):
    """Anima model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_encoder: Qwen3EncoderField = OutputField(description=FieldDescriptions.qwen3_encoder, title="Qwen3 Encoder")
    qwen3_5_encoder: Optional[Qwen35EncoderField] = OutputField(
        default=None,
        description=f"{FieldDescriptions.qwen3_5_encoder}. Set only for an Anima-3.8B model.",
        title="Qwen3.5 Encoder",
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "anima_model_loader",
    title="Main Model - Anima",
    tags=["model", "anima"],
    category="model",
    version="1.5.0",
    classification=Classification.Prototype,
)
class AnimaModelLoaderInvocation(BaseInvocation):
    """Loads an Anima model, outputting its submodels.

    Anima uses:
    - Transformer: Cosmos Predict2 DiT + LLM Adapter (from single-file checkpoint)
    - Qwen3 Encoder: Qwen3 0.6B (standalone single-file)
    - VAE: AutoencoderKLQwenImage / Wan 2.1 VAE (standalone single-file)
    - Qwen3.5 Encoder: Qwen3.5 4B, for Anima-3.8B only, whose bundled semantic connector reads it

    The T5-XXL tokenizer needed for LLM Adapter token IDs is bundled in the package,
    so no T5-XXL encoder model needs to be installed.
    """

    model: ModelIdentifierField = InputField(
        description="Anima main model (transformer + LLM adapter).",
        input=Input.Direct,
        ui_model_base=BaseModelType.Anima,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: ModelIdentifierField = InputField(
        description="Standalone VAE model. Anima uses a Wan 2.1 / QwenImage VAE (16-channel).",
        input=Input.Direct,
        ui_model_base=accepted_vae_bases(BaseModelType.Anima),
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    qwen3_encoder_model: ModelIdentifierField = InputField(
        description="Standalone Qwen3 0.6B Encoder model.",
        input=Input.Direct,
        ui_model_type=ModelType.Qwen3Encoder,
        title="Qwen3 Encoder",
    )

    qwen3_5_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen3.5 4B Encoder model. Required by Anima-3.8B, ignored by every other Anima model.",
        input=Input.Direct,
        ui_model_type=ModelType.Qwen35Encoder,
        title="Qwen3.5 Encoder",
    )

    def invoke(self, context: InvocationContext) -> AnimaModelLoaderOutput:
        # Transformer always comes from the main model
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # VAE
        vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})

        # Qwen3 Encoder
        qwen3_tokenizer = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        qwen3_encoder = self.qwen3_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        return AnimaModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_encoder=Qwen3EncoderField(tokenizer=qwen3_tokenizer, text_encoder=qwen3_encoder),
            qwen3_5_encoder=self._qwen3_5_encoder(context),
            vae=VAEField(vae=vae),
        )

    def _qwen3_5_encoder(self, context: InvocationContext) -> Optional[Qwen35EncoderField]:
        """The Qwen3.5 encoder if the main model reads one, else None. Refuses a model that needs one and has none."""
        variant = getattr(context.models.get_config(self.model), "variant", None)
        if variant != AnimaVariantType.Qwen35:
            if self.qwen3_5_encoder_model is not None:
                context.logger.warning(
                    f"{self.model.name} is not conditioned on Qwen3.5; ignoring the selected Qwen3.5 encoder."
                )
            return None
        if self.qwen3_5_encoder_model is None:
            raise ValueError(
                f"{self.model.name} bundles a Qwen3.5 semantic connector and needs a Qwen3.5 4B encoder. "
                "Install one (qwen35_4b.safetensors) and select it as the Qwen3.5 Encoder."
            )
        return Qwen35EncoderField(
            tokenizer=self.qwen3_5_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
            text_encoder=self.qwen3_5_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        )
