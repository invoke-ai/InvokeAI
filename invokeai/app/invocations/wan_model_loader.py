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
    VAEField,
    WanT5EncoderField,
    WanTransformerField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


@invocation_output("wan_model_loader_output")
class WanModelLoaderOutput(BaseInvocationOutput):
    """Wan 2.2 model loader output."""

    transformer: WanTransformerField = OutputField(
        description="Wan transformer (one or two experts depending on the variant)",
        title="Transformer",
    )
    wan_t5_encoder: WanT5EncoderField = OutputField(
        description=FieldDescriptions.wan_t5_encoder,
        title="UMT5-XXL Encoder",
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "wan_model_loader",
    title="Main Model - Wan 2.2",
    tags=["model", "wan"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class WanModelLoaderInvocation(BaseInvocation):
    """Loads a Wan 2.2 model, outputting its submodels.

    Components can be mixed and matched, mirroring the Qwen Image loader pattern:

    - Transformer(s) always come from the main model. For A14B that's both
      ``transformer/`` (high-noise) and ``transformer_2/`` (low-noise); for
      TI2V-5B it's the single ``transformer/``.
    - VAE: standalone Wan VAE > main (if Diffusers) > Component Source (Diffusers).
    - UMT5-XXL encoder: standalone Wan T5 encoder > main (if Diffusers) >
      Component Source (Diffusers).

    The Component Source slot lets users supply a Diffusers Wan main model purely
    for VAE / encoder extraction when the actual transformer is in a single-file
    format (GGUF in Phase 4). Together, the standalone VAE + standalone encoder
    let a GGUF transformer run without a full ~30 GB Diffusers install.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.wan_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Wan,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Wan VAE model. If not set, the VAE is loaded from the main model "
        "(when in Diffusers format) or from the Component Source.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Wan,
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    wan_t5_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Wan UMT5-XXL encoder. If not set, the encoder is loaded from the main "
        "model (when in Diffusers format) or from the Component Source.",
        input=Input.Direct,
        ui_model_type=ModelType.WanT5Encoder,
        title="Wan T5 Encoder",
    )

    component_source: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Diffusers Wan main model to extract VAE and/or encoder from. "
        "Use this if you don't have separate VAE/encoder models. "
        "Ignored for any submodel that is provided separately.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Wan,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Diffusers,
        title="Component Source (Diffusers)",
    )

    def invoke(self, context: InvocationContext) -> WanModelLoaderOutput:
        main_config = context.models.get_config(self.model)
        main_is_diffusers = main_config.format == ModelFormat.Diffusers

        # Primary transformer: the high-noise expert for A14B, or the only
        # transformer for TI2V-5B.
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

        # Dual-expert (A14B) wiring. The probe records ``has_dual_expert`` and
        # the recorded ``boundary_ratio`` from model_index.json on the config.
        transformer_low_noise = None
        boundary_ratio = 0.875  # Sensible Wan A14B default; overridden by model config when present.
        if getattr(main_config, "has_dual_expert", False):
            transformer_low_noise = self.model.model_copy(update={"submodel_type": SubModelType.Transformer2})
            recorded = getattr(main_config, "boundary_ratio", None)
            if recorded is not None:
                boundary_ratio = float(recorded)

        # VAE: standalone override > main (if Diffusers) > component source.
        if self.vae_model is not None:
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif main_is_diffusers:
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif self.component_source is not None:
            self._validate_component_source_format(context, self.component_source)
            vae = self.component_source.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No source for VAE. Either set 'VAE' to a standalone Wan VAE, "
                "or set 'Component Source' to a Diffusers Wan main model."
            )

        # Tokenizer + text encoder: standalone override > main (if Diffusers) > component source.
        if self.wan_t5_encoder_model is not None:
            tokenizer = self.wan_t5_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.wan_t5_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif main_is_diffusers:
            tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        elif self.component_source is not None:
            self._validate_component_source_format(context, self.component_source)
            tokenizer = self.component_source.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = self.component_source.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            raise ValueError(
                "No source for Wan T5 encoder. "
                "Either set 'Wan T5 Encoder' to a standalone UMT5-XXL encoder, "
                "or set 'Component Source' to a Diffusers Wan main model."
            )

        return WanModelLoaderOutput(
            transformer=WanTransformerField(
                transformer=transformer,
                transformer_low_noise=transformer_low_noise,
                boundary_ratio=boundary_ratio,
            ),
            wan_t5_encoder=WanT5EncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
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
