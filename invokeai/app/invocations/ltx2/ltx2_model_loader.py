"""LTX-2 model loader: one generation is assembled from up to three records."""

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
    LTX2LatentUpsamplerField,
    LTX2TextEncoderField,
    LTX2TransformerField,
    LTX2VocoderField,
    ModelIdentifierField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.configs.main import Main_Diffusers_LTX2_Config
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    LTX2VariantType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@invocation_output("ltx2_model_loader_output")
class LTX2ModelLoaderOutput(BaseInvocationOutput):
    """LTX-2 model loader output."""

    transformer: LTX2TransformerField = OutputField(description=FieldDescriptions.ltx2_model, title="Transformer")
    text_encoder: LTX2TextEncoderField = OutputField(
        description=FieldDescriptions.ltx2_text_encoder, title="Gemma-4 Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="Video VAE")
    audio_vae: VAEField = OutputField(description=FieldDescriptions.ltx2_audio_vae, title="Audio VAE")
    vocoder: LTX2VocoderField = OutputField(description=FieldDescriptions.ltx2_vocoder, title="Vocoder")
    latent_upsampler: LTX2LatentUpsamplerField = OutputField(
        description=FieldDescriptions.ltx2_latent_upsampler, title="Latent Upsampler"
    )

    # Echoed so a graph can record what it ran with. The inputs are all Direct (base-filtered
    # pickers), so nothing upstream could be read instead, and a literal re-typed into a metadata
    # node would go stale the first time a user changed the model.
    model: ModelIdentifierField = OutputField(description="The LTX-2 transformer that was loaded.", title="Model")
    component_source: Optional[ModelIdentifierField] = OutputField(
        default=None, description="The component folder the VAEs, vocoder and connectors came from.", title="Components"
    )
    text_encoder_model: ModelIdentifierField = OutputField(
        description="The Gemma-4 encoder that was loaded.", title="Text Encoder"
    )


@invocation(
    "ltx2_model_loader",
    title="Main Model - LTX-2",
    tags=["model", "ltx", "ltx2", "video"],
    category="model",
    version="1.2.0",
    classification=Classification.Prototype,
)
class LTX2ModelLoaderInvocation(BaseInvocation):
    """Loads an LTX-2 model, outputting its submodels.

    LTX-2 is distributed as separate files, and this node assembles one generation from up to
    three installed records: the 22B transformer (a single-file checkpoint, or a component folder
    that happens to carry one), the component folder that supplies the video VAE, audio VAE,
    vocoder and text connectors, and the Lightricks-tuned Gemma-4 text encoder.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.ltx2_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.LTX2,
        ui_model_type=ModelType.Main,
        title="Model",
    )
    component_source: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The LTX-2 component folder supplying the video VAE, audio VAE, vocoder and text "
        "connectors. Required unless the Model field is itself a folder that carries them.",
        input=Input.Direct,
        ui_model_base=BaseModelType.LTX2,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Diffusers,
        title="Components",
    )
    text_encoder_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.ltx2_text_encoder,
        input=Input.Direct,
        ui_model_base=BaseModelType.LTX2,
        ui_model_type=ModelType.Gemma4Encoder,
        title="Text Encoder",
    )

    def invoke(self, context: InvocationContext) -> LTX2ModelLoaderOutput:
        transformer_config = self._require_main(context, self.model, "Model")

        components = self.component_source or self.model
        components_config = (
            self._require_main(context, self.component_source, "Components")
            if self.component_source is not None
            else transformer_config
        )
        if not isinstance(components_config, Main_Diffusers_LTX2_Config):
            raise ValueError(
                f"'{components_config.name}' is a single-file transformer, so it carries none of the components "
                "this node fans out. Select an installed LTX-2 component folder (e.g. 'LTX-2.5 Components') in "
                "the Components field."
            )
        if isinstance(transformer_config, Main_Diffusers_LTX2_Config) and transformer_config.components_only:
            raise ValueError(
                f"'{transformer_config.name}' is a components-only LTX-2 install: it carries no transformer. "
                "Select a single-file LTX-2 transformer (dev or distilled) in the Model field and keep this "
                "folder in the Components field."
            )

        text_encoder_config = context.models.get_config(self._require_installed(context, self.text_encoder_model))
        if (
            text_encoder_config.base is not BaseModelType.LTX2
            or text_encoder_config.type is not ModelType.Gemma4Encoder
        ):
            raise ValueError(
                f"'{text_encoder_config.name}' is not an LTX-2 Gemma-4 text encoder (resolved to "
                f"type={getattr(text_encoder_config.type, 'value', text_encoder_config.type)})."
            )

        # The variant tells the denoise node which schedule the checkpoint was trained for; a
        # distilled transformer sampled on the dev schedule produces noise.
        variant = getattr(transformer_config, "variant", None)
        return LTX2ModelLoaderOutput(
            transformer=LTX2TransformerField(
                transformer=self.model.model_copy(update={"submodel_type": SubModelType.Transformer}),
                variant=variant.value if isinstance(variant, LTX2VariantType) else None,
            ),
            text_encoder=LTX2TextEncoderField(
                tokenizer=self.text_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
                text_encoder=self.text_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
                connectors=components.model_copy(update={"submodel_type": SubModelType.Connectors}),
            ),
            vae=VAEField(vae=components.model_copy(update={"submodel_type": SubModelType.VAE})),
            audio_vae=VAEField(vae=components.model_copy(update={"submodel_type": SubModelType.AudioVAE})),
            vocoder=LTX2VocoderField(vocoder=components.model_copy(update={"submodel_type": SubModelType.Vocoder})),
            latent_upsampler=LTX2LatentUpsamplerField(
                latent_upsampler=components.model_copy(update={"submodel_type": SubModelType.LatentUpsampler})
            ),
            model=self.model,
            component_source=self.component_source,
            text_encoder_model=self.text_encoder_model,
        )

    @staticmethod
    def _require_installed(context: InvocationContext, identifier: ModelIdentifierField) -> str:
        if not context.models.exists(identifier.key):
            raise ValueError(f"Unknown model: {identifier.key}")
        return identifier.key

    def _require_main(self, context: InvocationContext, identifier: ModelIdentifierField, field: str):
        """Resolve a field's model config and refuse anything that is not an LTX-2 main model.

        The pickers filter on these, but a hand-authored workflow or a client that ignores the hints
        can still send anything, and the failure would otherwise surface minutes later inside a
        loader.
        """
        config = context.models.get_config(self._require_installed(context, identifier))
        if config.base is not BaseModelType.LTX2 or config.type is not ModelType.Main:
            raise ValueError(
                f"The {field} field needs an LTX-2 main model; '{config.name}' resolved to "
                f"type={getattr(config.type, 'value', config.type)}, "
                f"base={getattr(config.base, 'value', config.base)}."
            )
        return config
