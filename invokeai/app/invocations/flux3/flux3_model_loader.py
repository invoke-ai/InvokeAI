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
from invokeai.backend.model_manager.qwen3_vl_labels import raise_for_wrong_qwen3_vl_variant
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen3VLVariantType,
    SubModelType,
)


@invocation_output("flux3_model_loader_output")
class Flux3ModelLoaderOutput(BaseInvocationOutput):
    """FLUX.3 model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    qwen3_vl_encoder: Qwen3VLEncoderField = OutputField(
        description=FieldDescriptions.qwen3_vl_encoder, title="Qwen3-VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "flux3_model_loader",
    title="Main Model - FLUX.3",
    tags=["model", "flux3", "flux"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux3ModelLoaderInvocation(BaseInvocation):
    """Loads a FLUX.3 Image model (Klein or Klein Base), outputting its submodels.

    A Diffusers model brings its own VAE (the FLUX.2 VAE) and Qwen3-VL 4B text encoder; a single-file or GGUF
    model is only the transformer and needs both picked separately. A standalone pick overrides the bundled
    component.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.flux3_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Flux3,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE (the FLUX.2 VAE, which FLUX.3 shares). Required when Model is a single file; "
        "otherwise an override for the bundled VAE.",
        input=Input.Direct,
        ui_model_base=accepted_vae_bases(BaseModelType.Flux3),
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    qwen3_vl_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen3-VL 4B encoder. Required when Model is a single file; otherwise an override "
        "for the encoder bundled in the Diffusers model.",
        input=Input.Direct,
        # Base `any` excludes MiniMax H3's truncated Qwen3-VL-32B, which shares this model type.
        ui_model_base=BaseModelType.Any,
        ui_model_type=ModelType.Qwen3VLEncoder,
        title="Qwen3-VL Encoder",
    )

    def invoke(self, context: InvocationContext) -> Flux3ModelLoaderOutput:
        main_config = context.models.get_config(self.model)
        if main_config.base is not BaseModelType.Flux3 or main_config.type is not ModelType.Main:
            raise ValueError(f"Model '{main_config.name}' is not a FLUX.3 main model. Select a FLUX.3 transformer.")
        is_diffusers = main_config.format is ModelFormat.Diffusers

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        if self.vae_model is not None:
            vae_config = context.models.get_config(self.vae_model)
            if vae_config.type is not ModelType.VAE or not accepts_vae(
                BaseModelType.Flux3, vae_config.base, getattr(vae_config, "latent_channels", None)
            ):
                accepted = " or ".join(base.value for base in accepted_vae_bases(BaseModelType.Flux3))
                raise ValueError(
                    f"VAE '{vae_config.name}' is not compatible with FLUX.3. Select a VAE of base {accepted}."
                )
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif is_diffusers:
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                f"'{main_config.name}' is a single-file FLUX.3 model and carries no VAE. Select the FLUX.2 VAE."
            )

        if self.qwen3_vl_encoder_model is not None:
            encoder_config = context.models.get_config(self.qwen3_vl_encoder_model)
            if encoder_config.type is not ModelType.Qwen3VLEncoder:
                raise ValueError(f"Encoder '{encoder_config.name}' is not a Qwen3-VL encoder.")
            raise_for_wrong_qwen3_vl_variant(
                encoder_config.name, getattr(encoder_config, "variant", None), Qwen3VLVariantType.Qwen3VL_4B, "FLUX.3"
            )
            tokenizer = self.qwen3_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.qwen3_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif is_diffusers:
            tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                f"'{main_config.name}' is a single-file FLUX.3 model and carries no text encoder. Select the "
                "Qwen3-VL 4B encoder."
            )

        return Flux3ModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            qwen3_vl_encoder=Qwen3VLEncoderField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[]),
            vae=VAEField(vae=vae),
        )
