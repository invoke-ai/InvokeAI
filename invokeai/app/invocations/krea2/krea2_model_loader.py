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
from invokeai.backend.architectures import accepted_vae_bases, accepts_vae
from invokeai.backend.model_manager.qwen3_vl_labels import variant_label as _variant_label
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen3VLVariantType,
    SubModelType,
)


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
        ui_model_base=accepted_vae_bases(BaseModelType.Krea2),
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
        main_config = context.models.get_config(self.model)
        if main_config.base is not BaseModelType.Krea2 or main_config.type is not ModelType.Main:
            raise ValueError(
                f"Model '{main_config.name}' is not a Krea-2 main model. Select a Krea-2 transformer model."
            )

        # Transformer always comes from the main model.
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # Determine VAE source.
        if self.vae_model is not None:
            vae_config = context.models.get_config(self.vae_model)
            if vae_config.type is not ModelType.VAE or not accepts_vae(
                BaseModelType.Krea2, vae_config.base, getattr(vae_config, "latent_channels", None)
            ):
                accepted = " or ".join(base.value for base in accepted_vae_bases(BaseModelType.Krea2))
                raise ValueError(
                    f"VAE '{vae_config.name}' is not compatible with Krea-2. Select a VAE of base {accepted}."
                )
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            self._validate_diffusers_format(context, self.model, "Krea-2")
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        # Determine Qwen3-VL Encoder source.
        if self.qwen3_vl_encoder_model is not None:
            encoder_config = context.models.get_config(self.qwen3_vl_encoder_model)
            if encoder_config.type is not ModelType.Qwen3VLEncoder:
                raise ValueError(f"Encoder '{encoder_config.name}' is not a Qwen3-VL encoder compatible with Krea-2.")
            # Two Qwen3-VL encoders install under this type and they are not interchangeable: Krea-2
            # conditions on the 4B, Ideogram 4 on the 8B. Without this check the 8B loads (~10 GB)
            # and then fails on a 4096-against-2560 shape error naming a tensor, which says nothing
            # about what the user picked.
            encoder_variant = getattr(encoder_config, "variant", None)
            if encoder_variant is not Qwen3VLVariantType.Qwen3VL_4B:
                raise ValueError(
                    f"'{encoder_config.name}' is the Qwen3-VL {_variant_label(encoder_variant)} encoder. "
                    "Krea-2 conditions on the 4B one (hidden size 2560)."
                )
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
