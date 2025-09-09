from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import Input, InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField, Qwen2_5VLField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import (
    CheckpointConfigBase,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("qwen_image_model_loader_output")
class QwenImageModelLoaderOutput(BaseInvocationOutput):
    """Qwen-Image base model loader output"""

    transformer: TransformerField = OutputField(description="Qwen-Image transformer model", title="Transformer")
    qwen2_5_vl: Qwen2_5VLField = OutputField(description="Qwen2.5-VL text encoder for Qwen-Image", title="Text Encoder")
    vae: VAEField = OutputField(description="Qwen-Image VAE", title="VAE")


@invocation(
    "qwen_image_model_loader",
    title="Main Model - Qwen-Image",
    tags=["model", "qwen-image"],
    category="model",
    version="1.0.0",
)
class QwenImageModelLoaderInvocation(BaseInvocation):
    """Loads a Qwen-Image base model, outputting its submodels."""

    model: ModelIdentifierField = InputField(
        description="Qwen-Image main model",
        input=Input.Direct,
        ui_model_base=BaseModelType.QwenImage,
        ui_model_type=ModelType.Main,
    )

    qwen2_5_vl_model: ModelIdentifierField = InputField(
        description="Qwen2.5-VL vision-language model",
        input=Input.Direct,
        title="Qwen2.5-VL Model",
        ui_model_base=BaseModelType.QwenImage,
        # ui_model_type=ModelType.VL
    )

    vae_model: ModelIdentifierField | None = InputField(
        description="VAE model for Qwen-Image",
        title="VAE",
        ui_model_base=BaseModelType.QwenImage,
        ui_model_type=ModelType.VAE,
        default=None,
    )

    def invoke(self, context: InvocationContext) -> QwenImageModelLoaderOutput:
        # Validate that required models exist
        for key in [self.model.key, self.qwen2_5_vl_model.key]:
            if not context.models.exists(key):
                raise ValueError(f"Unknown model: {key}")

        # Validate optional VAE model if provided
        if self.vae_model and not context.models.exists(self.vae_model.key):
            raise ValueError(f"Unknown model: {self.vae_model.key}")

        # Create submodel references
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # Use provided VAE or extract from main model
        if self.vae_model:
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            # Use the VAE bundled with the Qwen-Image model
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        # For Qwen-Image, we use Qwen2.5-VL as the text encoder
        tokenizer = self.qwen2_5_vl_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.qwen2_5_vl_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        # Get transformer config for any model-specific settings
        transformer_config = context.models.get_config(transformer)
        assert isinstance(transformer_config, CheckpointConfigBase)

        return QwenImageModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen2_5_vl=Qwen2_5VLField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[]),
            vae=VAEField(vae=vae),
        )
