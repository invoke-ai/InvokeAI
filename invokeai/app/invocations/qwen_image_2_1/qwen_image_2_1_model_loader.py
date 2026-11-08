from typing import Optional

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField, Qwen3VLEncoderField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.architectures import accepted_vae_bases, accepts_vae
from invokeai.backend.model_manager.qwen3_vl_labels import variant_label
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen3VLVariantType,
    SubModelType,
)


@invocation_output("qwen_image_2_1_model_loader_output")
class QwenImage21ModelLoaderOutput(BaseInvocationOutput):
    """Qwen-Image-2.1 model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_vl_encoder: Qwen3VLEncoderField = OutputField(
        description=FieldDescriptions.qwen3_vl_encoder, title="Qwen3-VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "qwen_image_2_1_model_loader",
    title="Main Model - Qwen-Image-2.1",
    tags=["model", "qwen_image_2_1", "qwen-image-2.1"],
    category="model",
    version="1.1.0",
    classification=Classification.Prototype,
)
class QwenImage21ModelLoaderInvocation(BaseInvocation):
    """Loads a Qwen-Image-2.1 model, outputting its submodels.

    A diffusers pipeline brings its own VAE and Qwen3-VL-8B encoder. A single-file or GGUF transformer has
    neither, so it needs a standalone VAE and encoder.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.qwen_image_2_1_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.QwenImage21,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )
    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen-Image-2.1 VAE. If not provided, the VAE is loaded from the diffusers model.",
        input=Input.Direct,
        ui_model_base=accepted_vae_bases(BaseModelType.QwenImage21),
        ui_model_type=ModelType.VAE,
        title="VAE",
    )
    qwen3_vl_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen3-VL 8B encoder. If not provided, the encoder is loaded from the diffusers model.",
        input=Input.Direct,
        # Base `any` excludes MiniMax H3's truncated Qwen3-VL-32B, which shares this model type. The 4B encoder
        # Krea-2 uses is still offered; `invoke` refuses it by its variant.
        ui_model_base=BaseModelType.Any,
        ui_model_type=ModelType.Qwen3VLEncoder,
        title="Qwen3-VL Encoder",
    )
    qwen3_vl_vision_tower_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Qwen3-VL 8B vision tower (the llama.cpp mmproj file). Lets a GGUF encoder read reference images.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Any,
        ui_model_type=ModelType.Qwen3VLVisionTower,
        title="Vision Tower",
    )

    def invoke(self, context: InvocationContext) -> QwenImage21ModelLoaderOutput:
        main_config = context.models.get_config(self.model)
        if main_config.base is not BaseModelType.QwenImage21 or main_config.type is not ModelType.Main:
            raise ValueError(f"Model '{main_config.name}' is not a Qwen-Image-2.1 main model.")

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        if self.vae_model is not None:
            vae_config = context.models.get_config(self.vae_model)
            if vae_config.type is not ModelType.VAE or not accepts_vae(BaseModelType.QwenImage21, vae_config.base):
                raise ValueError(
                    f"VAE '{vae_config.name}' is not a Qwen-Image-2.1 VAE. Qwen-Image-2.1 decodes 64 latent "
                    "channels into RGBA with its own VAE."
                )
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            self._require_diffusers(main_config.name, main_config.format, "VAE")
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        if self.qwen3_vl_encoder_model is not None:
            encoder_config = context.models.get_config(self.qwen3_vl_encoder_model)
            encoder_variant = getattr(encoder_config, "variant", None)
            if (
                encoder_config.type is not ModelType.Qwen3VLEncoder
                or encoder_variant is not Qwen3VLVariantType.Qwen3VL_8B
            ):
                raise ValueError(
                    f"'{encoder_config.name}' is the Qwen3-VL {variant_label(encoder_variant)} encoder. "
                    "Qwen-Image-2.1 conditions on the 8B one (hidden size 4096)."
                )
            tokenizer = self.qwen3_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.qwen3_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            self._require_diffusers(main_config.name, main_config.format, "Qwen3-VL encoder")
            tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        vision_tower = self._vision_tower(context)

        return QwenImage21ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_vl_encoder=Qwen3VLEncoderField(
                tokenizer=tokenizer, text_encoder=text_encoder, vision_tower=vision_tower, loras=[]
            ),
            vae=VAEField(vae=vae),
        )

    def _vision_tower(self, context: InvocationContext) -> Optional[ModelIdentifierField]:
        """The mmproj tower for a GGUF encoder, checked against it; None when none was selected.

        Refused beside any other encoder rather than ignored: those bring a tower of their own, and a stale
        selection would silently decide nothing while looking as if it did.
        """
        if self.qwen3_vl_vision_tower_model is None:
            return None
        tower_config = context.models.get_config(self.qwen3_vl_vision_tower_model)
        tower_variant = getattr(tower_config, "variant", None)
        if tower_config.type is not ModelType.Qwen3VLVisionTower:
            raise ValueError(f"'{tower_config.name}' is not a Qwen3-VL vision tower (the llama.cpp mmproj file).")
        if tower_variant is not Qwen3VLVariantType.Qwen3VL_8B:
            raise ValueError(
                f"'{tower_config.name}' is the Qwen3-VL {variant_label(tower_variant)} vision tower. "
                "Qwen-Image-2.1 reads references with the 8B one (hidden size 4096)."
            )
        encoder_config = (
            context.models.get_config(self.qwen3_vl_encoder_model) if self.qwen3_vl_encoder_model is not None else None
        )
        if encoder_config is None or encoder_config.format is not ModelFormat.GGUFQuantized:
            raise ValueError(
                f"The vision tower '{tower_config.name}' pairs only with a GGUF Qwen3-VL encoder, which has none. "
                "Any other encoder brings its own: clear the vision tower."
            )
        return self.qwen3_vl_vision_tower_model

    @staticmethod
    def _require_diffusers(name: str, model_format: ModelFormat, component: str) -> None:
        if model_format is not ModelFormat.Diffusers:
            raise ValueError(
                f"'{name}' is a {model_format.value} transformer with no {component} of its own. "
                f"Select a standalone {component}, or use the diffusers model."
            )
