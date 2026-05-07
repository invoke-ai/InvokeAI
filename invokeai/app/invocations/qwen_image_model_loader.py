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
    version="1.2.0",
    classification=Classification.Prototype,
)
class QwenImageModelLoaderInvocation(BaseInvocation):
    """Loads a Qwen Image model, outputting its submodels.

    The transformer is always loaded from the main model (Diffusers or GGUF).

    Components can be mixed and matched:
    - VAE: standalone Qwen Image VAE checkpoint, the Component Source (Diffusers),
      or the main model if it's Diffusers.
    - Qwen VL Encoder: standalone Qwen2.5-VL encoder, the Component Source
      (Diffusers), or the main model if it's Diffusers.

    Together, the standalone VAE and standalone encoder allow running a GGUF
    transformer without ever downloading the full ~40 GB Diffusers pipeline.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.qwen_image_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.QwenImage,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen Image VAE model. "
        "If not provided, VAE will be loaded from the Component Source (or from the main model if it is Diffusers).",
        input=Input.Direct,
        ui_model_base=BaseModelType.QwenImage,
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    qwen_vl_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen2.5-VL encoder model. "
        "If not provided, the encoder will be loaded from the Component Source "
        "(or from the main model if it is Diffusers).",
        input=Input.Direct,
        ui_model_type=ModelType.QwenVLEncoder,
        title="Qwen VL Encoder",
    )

    component_source: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Diffusers Qwen Image model to extract VAE and/or Qwen VL encoder from. "
        "Use this if you don't have separate VAE/encoder models. "
        "Ignored for any submodel that is provided separately.",
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

        # Resolve VAE: standalone override > main (if Diffusers) > component source
        if self.vae_model is not None:
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif main_is_diffusers:
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif self.component_source is not None:
            self._validate_component_source_format(context, self.component_source)
            vae = self.component_source.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No source for VAE. Either set 'VAE' to a standalone Qwen Image VAE, "
                "or set 'Component Source' to a Diffusers Qwen Image model."
            )

        # Resolve Qwen VL encoder: standalone override > main (if Diffusers) > component source
        if self.qwen_vl_encoder_model is not None:
            tokenizer = self.qwen_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.qwen_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif main_is_diffusers:
            tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif self.component_source is not None:
            self._validate_component_source_format(context, self.component_source)
            tokenizer = self.component_source.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.component_source.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                "No source for Qwen VL encoder. "
                "Either set 'Qwen VL Encoder' to a standalone Qwen2.5-VL encoder, "
                "or set 'Component Source' to a Diffusers Qwen Image model."
            )

        return QwenImageModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen_vl_encoder=QwenVLEncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
        )

    @staticmethod
    def _validate_component_source_format(context: InvocationContext, model: ModelIdentifierField) -> None:
        source_config = context.models.get_config(model)
        if source_config.format != ModelFormat.Diffusers:
            raise ValueError(
                f"The Component Source model must be in Diffusers format. "
                f"The selected model '{source_config.name}' is in {source_config.format.value} format."
            )
