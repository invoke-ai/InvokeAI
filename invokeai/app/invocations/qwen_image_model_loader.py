from typing import Literal, Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField, UIType
from invokeai.app.invocations.model import ModelIdentifierField, Qwen2_5VLField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import SubModelType


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
        ui_type=UIType.QwenImageMainModel,
        input=Input.Direct,
    )

    qwen2_5_vl_model: ModelIdentifierField = InputField(
        description="Qwen2.5-VL vision-language model",
        ui_type=UIType.Qwen2_5VLModel,
        input=Input.Direct,
        title="Qwen2.5-VL Model"
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        description="VAE model (uses Qwen-Image's bundled VAE if not specified)",
        ui_type=UIType.VAEModel,
        title="VAE",
        default=None
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

        return QwenImageModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen2_5_vl=Qwen2_5VLField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[]),
            vae=VAEField(vae=vae),
        )