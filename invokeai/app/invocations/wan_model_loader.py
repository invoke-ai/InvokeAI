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

# Transformer-only Wan formats: one file holds exactly one expert, so the A14B MoE
# pair has to be wired up by hand and the VAE / T5 encoder come from elsewhere.
_SINGLE_FILE_FORMATS = frozenset({ModelFormat.GGUFQuantized, ModelFormat.Checkpoint})


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
    # Not bumped for the single-file-checkpoint support: no stored node data changed,
    # only the live template's model-picker filter widened. Bumping would flag every
    # saved Wan workflow as needing an update for no benefit.
    version="1.0.1",
    classification=Classification.Prototype,
)
class WanModelLoaderInvocation(BaseInvocation):
    """Loads a Wan 2.2 model, outputting its submodels.

    Components can be mixed and matched, mirroring the Qwen Image loader pattern:

    - Transformer(s):
        * Diffusers main: emits ``transformer/`` and (for A14B) ``transformer_2/``
          from the same model record.
        * Single-file main (GGUF or safetensors checkpoint): emits the file as the
          primary transformer; for A14B the second-expert file must be wired to
          ``Transformer (Low Noise)``.
    - VAE: standalone Wan VAE > main (if Diffusers) > Component Source (Diffusers).
    - UMT5-XXL encoder: standalone Wan T5 encoder > main (if Diffusers) >
      Component Source (Diffusers).

    The Component Source slot lets users supply a Diffusers Wan main model purely
    for VAE / encoder extraction when the actual transformer is in a single-file
    format. Together, the standalone VAE + standalone encoder let a single-file
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
        description="Optional second single-file transformer for the A14B low-noise expert. "
        "Only relevant when the main model is a single-file GGUF or safetensors checkpoint and "
        "the variant is A14B; ignored when the main is a Diffusers A14B (both experts are pulled "
        "from transformer/ and transformer_2/ already) or when the variant is TI2V-5B.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Wan,
        ui_model_type=ModelType.Main,
        ui_model_format=[ModelFormat.GGUFQuantized, ModelFormat.Checkpoint],
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
        main_is_single_file = main_format in _SINGLE_FILE_FORMATS
        main_variant = getattr(main_config, "variant", None)
        if main_is_single_file and self.component_source is not None:
            self._validate_component_source_format(context, self.component_source)

        # Resolve transformer + dual-expert wiring + boundary_ratio.
        #
        # Diffusers main: transformer/ is the primary, transformer_2/ is the
        # low-noise expert (A14B only). boundary_ratio comes from the probed
        # model_index.json.
        #
        # Single-file main (GGUF or safetensors checkpoint): the file itself is one
        # expert (high or low). For A14B, the user wires the other expert to
        # transformer_low_noise_model. We swap so the *high*-noise expert is always
        # the primary if needed. boundary_ratio falls back to 0.875 unless a
        # Diffusers component_source provides a recorded value.
        boundary_ratio = 0.9 if main_variant == WanVariantType.I2V_A14B else 0.875
        transformer_low_noise: Optional[ModelIdentifierField] = None

        if main_is_diffusers:
            transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
            if getattr(main_config, "has_dual_expert", False):
                transformer_low_noise = self.model.model_copy(update={"submodel_type": SubModelType.Transformer2})
                recorded = getattr(main_config, "boundary_ratio", None)
                if recorded is not None:
                    boundary_ratio = float(recorded)
        elif main_is_single_file:
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
                        "A Wan A14B expert pair needs two different single-file models."
                    )
                low_config = context.models.get_config(self.transformer_low_noise_model)
                self._validate_main_config(low_config, "Transformer (Low Noise)")
                # The two experts don't have to share a format — both single-file
                # loaders produce a plain WanTransformer3DModel, so a GGUF high-noise
                # expert pairs fine with a safetensors low-noise one.
                if low_config.format not in _SINGLE_FILE_FORMATS:
                    raise ValueError(
                        f"'Transformer (Low Noise)' must be a single-file Wan model (GGUF or checkpoint). "
                        f"'{low_config.name}' is in {low_config.format.value} format."
                    )
                low_id = self.transformer_low_noise_model.model_copy(update={"submodel_type": SubModelType.Transformer})
                low_expert = getattr(low_config, "expert", "none")

                if getattr(low_config, "variant", None) != main_variant:
                    low_variant = getattr(low_config, "variant", None)
                    raise ValueError(
                        "The high-noise and low-noise models must use the same Wan variant, but "
                        f"'{main_config.name}' is {main_variant.value} and '{low_config.name}' is "
                        f"{getattr(low_variant, 'value', low_variant)}."
                    )

                # The expert tag is a filename heuristic, so 'none' (untagged) is common on
                # community finetunes. The wiring itself is explicit user intent — main slot
                # = high, low-noise slot = low — so an untagged file is taken at its wired
                # position (or inferred as the complement of its tagged partner). Only a
                # genuine conflict, both files claiming the *same* expert, is an error.
                if primary_expert == low_expert != "none":
                    raise ValueError(
                        f"Both selected models are tagged as the {primary_expert}-noise expert "
                        f"('{main_config.name}' and '{low_config.name}'). A Wan A14B expert pair "
                        "must contain one high and one low expert."
                    )
                if primary_expert == "none" and low_expert == "none":
                    context.logger.warning(
                        "Neither Wan A14B filename identifies its expert, so 'Transformer' is assumed to "
                        "be the high-noise expert and 'Transformer (Low Noise)' the low-noise expert. If the "
                        "output looks wrong, swap the two models."
                    )

                # Make sure 'transformer' is the high-noise expert and
                # 'transformer_low_noise' is the low-noise expert. If the user
                # accidentally swapped them, swap back.
                if primary_expert == "low" or low_expert == "high":
                    transformer = low_id
                    transformer_low_noise = primary_id
                    # The swap overrides the wiring on the strength of a
                    # filename tag, so say so: a mistagged file is otherwise an
                    # invisible expert inversion.
                    context.logger.warning(
                        f"The wired Wan A14B experts look reversed, so they were swapped: "
                        f"'{low_config.name}' (tagged '{low_expert}') runs as the high-noise expert and "
                        f"'{main_config.name}' (tagged '{primary_expert}') as the low-noise expert. "
                        "The tags come from the filenames — if the output looks wrong, a filename is lying."
                    )
                else:
                    transformer = primary_id
                    transformer_low_noise = low_id
            else:
                transformer = primary_id
                # A14B without a paired low-noise expert will produce degraded quality
                # (only one expert runs). Warn but don't abort — a single wired transformer
                # is explicit intent just like a pair is, and the tag is only a filename
                # guess, so an untagged file must not be fatal here when the paired path
                # accepts it. TI2V-5B is single-expert and totally fine.
                if main_variant in (WanVariantType.T2V_A14B, WanVariantType.I2V_A14B):
                    message = (
                        "An A14B single-file main is wired to 'Transformer' without a paired "
                        "'Transformer (Low Noise)'. Only this one expert will run; quality will be reduced."
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
                "Use a Diffusers folder, a GGUF file, or a single-file safetensors checkpoint."
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
