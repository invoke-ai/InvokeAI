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
    QwenVLEncoderField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


@invocation_output("qwen_image_model_loader_output")
class QwenImageModelLoaderOutput(BaseInvocationOutput):
    """Qwen Image model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen_vl_encoder: QwenVLEncoderField = OutputField(
        description=FieldDescriptions.qwen_vl_encoder, title="Qwen VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "qwen_image_model_loader",
    title="Main Model - Qwen Image",
    tags=["model", "qwen_image"],
    category="model",
    version="1.1.0",
    classification=Classification.Prototype,
)
class QwenImageModelLoaderInvocation(BaseInvocation):
    """Loads a Qwen Image model, outputting its submodels.

    The transformer is always loaded from the main model (Diffusers or GGUF).

    For GGUF quantized models, the VAE and Qwen VL encoder must come from a
    separate Diffusers model specified in the "Component Source" field.

    For Diffusers models, all components are extracted from the main model
    automatically. The "Component Source" field is ignored.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.qwen_image_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.QwenImage,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    component_source: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Diffusers Qwen Image model to extract the VAE and Qwen VL encoder from. "
        "Required when using a GGUF quantized transformer. "
        "Ignored when the main model is already in Diffusers format.",
        input=Input.Direct,
        ui_model_base=BaseModelType.QwenImage,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Diffusers,
        title="Component Source (Diffusers)",
    )

    def invoke(self, context: InvocationContext) -> QwenImageModelLoaderOutput:
        main_config = context.models.get_config(self.model)
        main_is_diffusers = main_config.format == ModelFormat.Diffusers

        # Transformer always comes from the main model
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        if main_is_diffusers:
            # Diffusers model: extract all components directly
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
            tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif self.component_source is not None:
            # GGUF/checkpoint transformer: get VAE + encoder from the component source
            source_config = context.models.get_config(self.component_source)
            if source_config.format != ModelFormat.Diffusers:
                raise ValueError(
                    f"The Component Source model must be in Diffusers format. "
                    f"The selected model '{source_config.name}' is in {source_config.format.value} format."
                )
            vae = self.component_source.model_copy(update={"submodel_type": SubModelType.VAE})
            tokenizer = self.component_source.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.component_source.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                "No source for VAE and Qwen VL encoder. "
                "GGUF quantized models only contain the transformer — "
                "please set 'Component Source' to a Diffusers Qwen Image model "
                "to provide the VAE and text encoder."
            )

        return QwenImageModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen_vl_encoder=QwenVLEncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
        )
