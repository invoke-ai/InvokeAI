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
    Qwen35EncoderField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.architectures import accepted_vae_bases, accepts_vae
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen35VariantType,
    SubModelType,
)


@invocation_output("moss_image_model_loader_output")
class MossImageModelLoaderOutput(BaseInvocationOutput):
    """MOSS-Image model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_5_encoder: Qwen35EncoderField = OutputField(
        description=FieldDescriptions.qwen3_5_encoder, title="Qwen3.5 Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "moss_image_model_loader",
    title="Main Model - MOSS-Image",
    tags=["model", "moss_image", "moss-image"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class MossImageModelLoaderInvocation(BaseInvocation):
    """Loads a MOSS-Image model, outputting its submodels.

    The transformer always comes from the main model. The Qwen3.5-9B encoder and the FLUX.2 VAE come from the
    standalone models where selected, and from the MOSS-Image diffusers pipeline otherwise; a single-file transformer
    has neither, so it needs both.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.moss_image_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.MossImage,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE model. MOSS-Image uses the FLUX.2 VAE (32-channel). "
        "If not provided, the VAE is loaded from the MOSS-Image (diffusers) model.",
        input=Input.Direct,
        ui_model_base=accepted_vae_bases(BaseModelType.MossImage),
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    qwen3_5_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen3.5 9B encoder model. "
        "If not provided, the encoder is loaded from the MOSS-Image (diffusers) model.",
        input=Input.Direct,
        ui_model_type=ModelType.Qwen35Encoder,
        title="Qwen3.5 Encoder",
    )

    def invoke(self, context: InvocationContext) -> MossImageModelLoaderOutput:
        main_config = context.models.get_config(self.model)
        if main_config.base is not BaseModelType.MossImage or main_config.type is not ModelType.Main:
            raise ValueError(
                f"Model '{main_config.name}' is not a MOSS-Image main model. Select a MOSS-Image transformer model."
            )
        missing = [
            name
            for name, selected in (("VAE", self.vae_model), ("Qwen3.5 Encoder", self.qwen3_5_encoder_model))
            if selected is None
        ]
        if missing and main_config.format is not ModelFormat.Diffusers:
            raise ValueError(
                f"'{main_config.name}' is a single-file MOSS-Image transformer, which carries no VAE or text "
                f"encoder. Select a standalone {' and '.join(missing)}, or use the MOSS-Image diffusers model."
            )

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        if self.vae_model is not None:
            vae_config = context.models.get_config(self.vae_model)
            if vae_config.type is not ModelType.VAE or not accepts_vae(
                BaseModelType.MossImage, vae_config.base, getattr(vae_config, "latent_channels", None)
            ):
                accepted = " or ".join(base.value for base in accepted_vae_bases(BaseModelType.MossImage))
                raise ValueError(
                    f"VAE '{vae_config.name}' is not compatible with MOSS-Image. Select a VAE of base {accepted}."
                )
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        if self.qwen3_5_encoder_model is not None:
            encoder_config = context.models.get_config(self.qwen3_5_encoder_model)
            # Two Qwen3.5 sizes install under this type, and only the 9B fits: MOSS-Image was trained on its 4096-wide
            # last hidden state. The 4B (Anima-3.8B's) would load and then fail on a shape error naming a tensor.
            if (
                encoder_config.type is not ModelType.Qwen35Encoder
                or getattr(encoder_config, "variant", None) != Qwen35VariantType.Qwen35_9B
            ):
                raise ValueError(
                    f"'{encoder_config.name}' is not a Qwen3.5 9B encoder, which MOSS-Image conditions on (hidden "
                    "size 4096). Select a Qwen3.5 9B encoder."
                )
            tokenizer = self.qwen3_5_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.qwen3_5_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        return MossImageModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_5_encoder=Qwen35EncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
        )
