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
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType, WanVariantType


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
    version="1.0.1",
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
        self._validate_main_config(main_config, "Wan main")
        main_format = main_config.format
        main_is_diffusers = main_format == ModelFormat.Diffusers
        main_is_gguf = main_format == ModelFormat.GGUFQuantized
        main_variant = getattr(main_config, "variant", None)
        if main_is_gguf and self.component_source is not None:
            self._validate_component_source_format(context, self.component_source)

        # Resolve transformer + dual-expert wiring + boundary_ratio.
        #
        # Diffusers main: transformer/ is the primary, transformer_2/ is the
        # low-noise expert (A14B only). boundary_ratio comes from the probed
        # model_index.json.
        #
        # GGUF main: the file itself is one expert. For A14B, the user wires
        # the other expert to transformer_low_noise_model. The explicit slots
        # define the roles; filename-derived expert tags are advisory only.
        # boundary_ratio falls back to 0.875 unless a Diffusers component_source
        # provides a recorded value.
        boundary_ratio = 0.9 if main_variant == WanVariantType.I2V_A14B else 0.875
        transformer_low_noise: Optional[ModelIdentifierField] = None

        if main_is_diffusers:
            transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
            if getattr(main_config, "has_dual_expert", False):
                transformer_low_noise = self.model.model_copy(update={"submodel_type": SubModelType.Transformer2})
                recorded = getattr(main_config, "boundary_ratio", None)
                if recorded is not None:
                    boundary_ratio = float(recorded)
        elif main_is_gguf:
            primary_expert = getattr(main_config, "expert", "none")
            primary_id = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})

            if self.transformer_low_noise_model is not None and main_variant == WanVariantType.TI2V_5B:
                # The field's own docs promise this input is ignored for the
                # single-expert TI2V-5B — e.g. a leftover wire from an A14B session.
                context.logger.warning("'Transformer (Low Noise)' is ignored for the single-expert TI2V-5B variant.")

            if self.transformer_low_noise_model is not None and main_variant != WanVariantType.TI2V_5B:
                if self.transformer_low_noise_model.key == self.model.key:
                    raise ValueError(
                        "The same model is wired to both 'Transformer' and 'Transformer (Low Noise)'. "
                        "A Wan A14B expert pair needs two different GGUF models."
                    )
                low_config = context.models.get_config(self.transformer_low_noise_model)
                self._validate_main_config(low_config, "Transformer (Low Noise)")
                if low_config.format != ModelFormat.GGUFQuantized:
                    raise ValueError(
                        f"'Transformer (Low Noise)' must be a GGUF-format Wan model. "
                        f"'{low_config.name}' is in {low_config.format.value} format."
                    )
                low_id = self.transformer_low_noise_model.model_copy(update={"submodel_type": SubModelType.Transformer})
                low_expert = getattr(low_config, "expert", "none")

                if getattr(low_config, "variant", None) != main_variant:
                    raise ValueError("The high-noise and low-noise GGUF models must use the same Wan variant.")

                # Expert tags come from filenames and may be absent or wrong on
                # community models. Explicit graph wiring is authoritative:
                # Transformer is high-noise and Transformer (Low Noise) is
                # low-noise. Tags can only trigger an advisory warning.
                if primary_expert == "none" and low_expert == "none":
                    context.logger.warning(
                        "Neither Wan A14B GGUF filename identifies its expert, so 'Transformer' is assumed to "
                        "be the high-noise expert and 'Transformer (Low Noise)' the low-noise expert. If the "
                        "output looks wrong, swap the two models."
                    )

                elif primary_expert == "low" or low_expert == "high" or primary_expert == low_expert:
                    context.logger.warning(
                        "The Wan A14B GGUF filename tags disagree with the explicit transformer wiring. "
                        "The wiring is authoritative: 'Transformer' runs as the high-noise expert and "
                        "'Transformer (Low Noise)' runs as the low-noise expert."
                    )
                transformer = primary_id
                transformer_low_noise = low_id
            else:
                transformer = primary_id
                # A14B without a paired low-noise GGUF will produce degraded
                # quality (only one expert runs). Warn but don't abort — a
                # single wired transformer is explicit intent just like a pair
                # is, and the tag is only a filename guess, so an untagged file
                # must not be fatal here when the paired path accepts it.
                # TI2V-5B GGUFs are single-expert and totally fine.
                if main_variant in (WanVariantType.T2V_A14B, WanVariantType.I2V_A14B):
                    message = (
                        "An A14B GGUF is wired to 'Transformer' without a paired 'Transformer (Low Noise)'. "
                        "Only this one expert will run; image quality will be reduced."
                    )
                    if primary_expert == "low":
                        message += (
                            " Its filename tags it as the low-noise expert; when running a single expert, "
                            "the high-noise one is usually the better choice."
                        )
                    context.logger.warning(message)

            # Borrow the boundary_ratio recorded on the optional Diffusers
            # component_source, when one is wired.
            if self.component_source is not None:
                src_cfg = context.models.get_config(self.component_source)
                src_boundary = getattr(src_cfg, "boundary_ratio", None)
                if (
                    src_cfg.format == ModelFormat.Diffusers
                    and getattr(src_cfg, "variant", None) == main_variant
                    and src_boundary is not None
                ):
                    boundary_ratio = float(src_boundary)
        else:
            raise ValueError(
                f"Unsupported main model format for Wan: {main_format.value}. "
                "Use a Diffusers folder or a GGUF single-file checkpoint."
            )

        # VAE: standalone override > main (if Diffusers) > component source.
        if self.vae_model is not None:
            self._validate_standalone_vae(context, self.vae_model, main_variant)
            vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif main_is_diffusers:
            vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        elif self.component_source is not None:
            self._validate_component_source_vae(context, self.component_source, main_variant)
            vae = self.component_source.model_copy(update={"submodel_type": SubModelType.VAE})
        else:
            raise ValueError(
                "No source for VAE. Either set 'VAE' to a standalone Wan VAE, "
                "or set 'Component Source' to a Diffusers Wan main model."
            )

        # Tokenizer + text encoder: standalone override > main (if Diffusers) > component source.
        if self.wan_t5_encoder_model is not None:
            t5_config = context.models.get_config(self.wan_t5_encoder_model)
            if t5_config.type != ModelType.WanT5Encoder or t5_config.format != ModelFormat.WanT5Encoder:
                raise ValueError("The Wan T5 Encoder must resolve to a standalone Wan T5 encoder model.")
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
    def _validate_main_config(config: object, label: str) -> None:
        if getattr(config, "base", None) != BaseModelType.Wan or getattr(config, "type", None) != ModelType.Main:
            raise ValueError(f"The {label} model must resolve to a Wan main model.")

    @staticmethod
    def _validate_component_source_format(context: InvocationContext, model: ModelIdentifierField) -> None:
        source_config = context.models.get_config(model)
        if source_config.base != BaseModelType.Wan or source_config.type != ModelType.Main:
            raise ValueError("The Component Source model must resolve to a Wan main model.")
        if source_config.format != ModelFormat.Diffusers:
            raise ValueError(
                f"The Component Source model must be in Diffusers format. "
                f"The selected model '{source_config.name}' is in {source_config.format.value} format."
            )

    @staticmethod
    def _validate_component_source_vae(
        context: InvocationContext, model: ModelIdentifierField, main_variant: WanVariantType
    ) -> None:
        WanModelLoaderInvocation._validate_component_source_format(context, model)
        source_config = context.models.get_config(model)
        source_variant = getattr(source_config, "variant", None)
        main_is_ti2v = main_variant == WanVariantType.TI2V_5B
        source_is_ti2v = source_variant == WanVariantType.TI2V_5B
        if main_is_ti2v != source_is_ti2v:
            raise ValueError(
                "The Component Source VAE is incompatible with the selected transformer. "
                "TI2V-5B requires the 48-channel Wan 2.2 VAE; A14B models require the 16-channel Wan 2.1 VAE."
            )

    @staticmethod
    def _validate_standalone_vae(
        context: InvocationContext, model: ModelIdentifierField, main_variant: WanVariantType
    ) -> None:
        vae_config = context.models.get_config(model)
        if vae_config.base != BaseModelType.Wan or vae_config.type != ModelType.VAE:
            raise ValueError("The VAE must resolve to a standalone Wan VAE model.")
        expected_channels = 48 if main_variant == WanVariantType.TI2V_5B else 16
        if vae_config.latent_channels != expected_channels:
            raise ValueError(
                "The standalone VAE is incompatible with the selected transformer. "
                "TI2V-5B requires the 48-channel Wan 2.2 VAE; A14B models require the 16-channel Wan 2.1 VAE."
            )
