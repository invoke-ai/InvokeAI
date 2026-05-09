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

    Diffusers-format only for now; the transformer(s), VAE, and UMT5-XXL encoder
    are pulled from the main model's submodel folders.

    For Wan 2.2 A14B (dual-expert MoE) the loader emits both ``transformer`` (the
    high-noise expert at ``transformer/``) and ``transformer_low_noise`` (the
    low-noise expert at ``transformer_2/``), along with the model's recorded
    ``boundary_ratio`` for the denoise loop's expert swap.

    The standalone VAE picker is forward-compatibility wiring for Phase 3 (where
    it becomes required for GGUF transformers).
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
        description="Standalone Wan VAE model. If not set, the VAE is loaded from the main "
        "model (when in Diffusers format).",
        input=Input.Direct,
        ui_model_base=BaseModelType.Wan,
        ui_model_type=ModelType.VAE,
        title="VAE",
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

        # VAE: standalone override > main (if Diffusers).
        if self.vae_model is not None:
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif main_is_diffusers:
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No source for VAE. Either set 'VAE' to a standalone Wan VAE model, "
                "or use a Diffusers Wan main model."
            )

        # Tokenizer + text encoder: only from the main model in Phase 1.
        # Phase 3 will add a standalone WanT5Encoder picker so GGUF mains can run
        # without a Diffusers Wan checkpoint installed.
        if not main_is_diffusers:
            raise ValueError(
                "Only Diffusers-format Wan models are supported in this build. "
                "Standalone Wan T5 encoders will be supported in a future release."
            )
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        return WanModelLoaderOutput(
            transformer=WanTransformerField(
                transformer=transformer,
                transformer_low_noise=transformer_low_noise,
                boundary_ratio=boundary_ratio,
            ),
            wan_t5_encoder=WanT5EncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
        )
