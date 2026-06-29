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
    Qwen3VLEncoderField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


@invocation_output("krea2_model_loader_output")
class Krea2ModelLoaderOutput(BaseInvocationOutput):
    """Krea-2 base model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_vl_encoder: Qwen3VLEncoderField = OutputField(
        description=FieldDescriptions.qwen3_vl_encoder, title="Qwen3-VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "krea2_model_loader",
    title="Main Model - Krea-2",
    tags=["model", "krea2", "krea-2"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Krea2ModelLoaderInvocation(BaseInvocation):
    """Loads a Krea-2 model, outputting its submodels.

    By default the VAE (Qwen-Image VAE) and Qwen3-VL text encoder are extracted from the Krea-2
    diffusers pipeline. Standalone overrides may be supplied (e.g. when the transformer is a
    single-file checkpoint that has no bundled VAE / encoder).
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.krea2_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Krea2,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE model. Krea-2 uses the Qwen-Image VAE (16-channel). "
        "If not provided, the VAE is loaded from the Krea-2 (diffusers) model.",
        input=Input.Direct,
        ui_model_base=BaseModelType.QwenImage,
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    qwen3_vl_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen3-VL Encoder model. "
        "If not provided, the encoder is loaded from the Krea-2 (diffusers) model.",
        input=Input.Direct,
        ui_model_type=ModelType.Qwen3VLEncoder,
        title="Qwen3-VL Encoder",
    )

    def invoke(self, context: InvocationContext) -> Krea2ModelLoaderOutput:
        # Transformer always comes from the main model.
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # Determine VAE source.
        if self.vae_model is not None:
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            self._validate_diffusers_format(context, self.model, "Krea-2")
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        # Determine Qwen3-VL Encoder source.
        if self.qwen3_vl_encoder_model is not None:
            tokenizer = self.qwen3_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.qwen3_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            self._validate_diffusers_format(context, self.model, "Krea-2")
            tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        return Krea2ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_vl_encoder=Qwen3VLEncoderField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[]),
            vae=VAEField(vae=vae),
        )

    def _validate_diffusers_format(
        self, context: InvocationContext, model: ModelIdentifierField, model_name: str
    ) -> None:
        """Validate that a model is in Diffusers format (required to extract VAE / encoder submodels)."""
        config = context.models.get_config(model)
        if config.format != ModelFormat.Diffusers:
            raise ValueError(
                f"To extract the VAE and Qwen3-VL encoder, the {model_name} model must be in Diffusers format. "
                f"The selected model '{config.name}' is in {config.format.value} format — provide a standalone "
                "VAE and Qwen3-VL Encoder instead."
            )
