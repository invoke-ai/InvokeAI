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

    - Transformer(s):
        * Diffusers main: emits ``transformer/`` and (for A14B) ``transformer_2/``
          from the same model record.
        * GGUF main: emits the single GGUF as the primary transformer; for A14B
          the second-expert GGUF must be wired to ``Transformer (Low Noise)``.
    - VAE: standalone Wan VAE > main (if Diffusers) > Component Source (Diffusers).
    - UMT5-XXL encoder: standalone Wan T5 encoder > main (if Diffusers) >
      Component Source (Diffusers).

    The Component Source slot lets users supply a Diffusers Wan main model purely
    for VAE / encoder extraction when the actual transformer is in a single-file
    format. Together, the standalone VAE + standalone encoder let a GGUF
    transformer run without a full ~30 GB Diffusers install.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.wan_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.Wan,
        ui_model_type=ModelType.Main,
        title="Transformer",
    )

    transformer_low_noise_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Optional second GGUF transformer for the A14B low-noise expert. "
        "Only relevant when the main model is a single-file GGUF and the variant is A14B; "
        "ignored when the main is a Diffusers A14B (both experts are pulled from "
        "transformer/ and transformer_2/ already) or when the variant is TI2V-5B.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Wan,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.GGUFQuantized,
        title="Transformer (Low Noise)",
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
        main_format = main_config.format
        main_is_diffusers = main_format == ModelFormat.Diffusers
        main_is_gguf = main_format == ModelFormat.GGUFQuantized

        # Resolve transformer + dual-expert wiring + boundary_ratio.
        #
        # Diffusers main: transformer/ is the primary, transformer_2/ is the
        # low-noise expert (A14B only). boundary_ratio comes from the probed
        # model_index.json.
        #
        # GGUF main: the file itself is one expert (high or low). For A14B,
        # the user wires the other expert to transformer_low_noise_model.
        # We swap so the *high*-noise expert is always the primary if needed.
        # boundary_ratio falls back to 0.875 unless a Diffusers component_source
        # provides a recorded value.
        boundary_ratio = 0.875
        transformer_low_noise: Optional[ModelIdentifierField] = None

        if main_is_diffusers:
            transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
            if getattr(main_config, "has_dual_expert", False):
                transformer_low_noise = self.model.model_copy(
                    update={"submodel_type": SubModelType.Transformer2}
                )
                recorded = getattr(main_config, "boundary_ratio", None)
                if recorded is not None:
                    boundary_ratio = float(recorded)
        elif main_is_gguf:
            primary_expert = getattr(main_config, "expert", "none")
            primary_id = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

            if self.transformer_low_noise_model is not None:
                low_config = context.models.get_config(self.transformer_low_noise_model)
                if low_config.format != ModelFormat.GGUFQuantized:
                    raise ValueError(
                        f"'Transformer (Low Noise)' must be a GGUF-format Wan model. "
                        f"'{low_config.name}' is in {low_config.format.value} format."
                    )
                low_id = self.transformer_low_noise_model.model_copy(
                    update={"submodel_type": SubModelType.Transformer}
                )
                low_expert = getattr(low_config, "expert", "none")

                # Make sure 'transformer' is the high-noise expert and
                # 'transformer_low_noise' is the low-noise expert. If the user
                # accidentally swapped them, swap back.
                if primary_expert == "low" and low_expert == "high":
                    transformer = low_id
                    transformer_low_noise = primary_id
                else:
                    transformer = primary_id
                    transformer_low_noise = low_id
            else:
                transformer = primary_id
                # A14B without a paired low-noise GGUF will produce degraded
                # quality (only the high-noise expert runs). Warn but don't
                # abort — TI2V-5B GGUFs are single-expert and totally fine.
                if (
                    getattr(main_config, "variant", None)
                    and main_config.variant.value == "t2v_a14b"
                ):
                    context.logger.warning(
                        "A14B GGUF main was provided without a paired 'Transformer (Low Noise)'. "
                        "Only the high-noise expert will run; image quality will be reduced."
                    )

            # Borrow the boundary_ratio recorded on the optional Diffusers
            # component_source, when one is wired.
            if self.component_source is not None:
                src_cfg = context.models.get_config(self.component_source)
                src_boundary = getattr(src_cfg, "boundary_ratio", None)
                if src_boundary is not None:
                    boundary_ratio = float(src_boundary)
        else:
            raise ValueError(
                f"Unsupported main model format for Wan: {main_format.value}. "
                "Use a Diffusers folder or a GGUF single-file checkpoint."
            )

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
