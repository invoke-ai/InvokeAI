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
    MiniMaxH3TextEncoderField,
    MiniMaxH3TransformerField,
    ModelIdentifierField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    MiniMaxH3VariantType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@invocation_output("minimax_h3_model_loader_output")
class MiniMaxH3ModelLoaderOutput(BaseInvocationOutput):
    """MiniMax H3 model loader output."""

    transformer: MiniMaxH3TransformerField = OutputField(
        description="MiniMax H3 task transformer (FL2VA or Ref2VA)", title="Transformer"
    )
    text_encoder: MiniMaxH3TextEncoderField = OutputField(
        description=FieldDescriptions.minimax_h3_text_encoder, title="Qwen3-VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="Video VAE")
    audio_vae: VAEField = OutputField(description=FieldDescriptions.minimax_h3_audio_vae, title="Audio VAE")
    # The three model identifiers are echoed back out so a graph can record what it ran with.
    # Their inputs are Input.Direct (a base-filtered picker), so nothing upstream can be read
    # instead, and a duplicate literal typed into a metadata node would silently go stale.
    model: ModelIdentifierField = OutputField(description="The MiniMax H3 model that was loaded.", title="Model")
    transformer_model: Optional[ModelIdentifierField] = OutputField(
        default=None,
        description="The single-file transformer override that was used, if any.",
        title="Transformer (single file)",
    )
    text_encoder_model: Optional[ModelIdentifierField] = OutputField(
        default=None,
        description="The single-file text encoder override that was used, if any.",
        title="Text Encoder (single file)",
    )


@invocation(
    "minimax_h3_model_loader",
    title="Main Model - MiniMax H3",
    tags=["model", "minimax", "video"],
    category="model",
    version="1.4.0",
    classification=Classification.Prototype,
)
class MiniMaxH3ModelLoaderInvocation(BaseInvocation):
    """Loads a MiniMax H3 model, outputting its submodels.

    All six submodels (transformer, text encoder, tokenizer, processor, video VAE, audio VAE)
    come from the one diffusers-layout install. Optionally, a single-file transformer checkpoint
    (e.g. the pruned int8 repack) replaces the folder's transformer, and/or a single-file
    truncated Qwen3-VL encoder (e.g. the int8 repack) replaces the folder's text encoder, while
    everything else keeps coming from the folder install.

    A Ref2VA diffusers folder's ``transformer_ref/`` weights are NOT folder-loadable (the
    submodel map has no entry for that folder), so Ref2VA generation always selects a
    single-file Ref2VA transformer here; identification marks such folders components-only so
    the UI requires it up front.
    """

    model: ModelIdentifierField = InputField(
        description=FieldDescriptions.minimax_h3_model,
        input=Input.Direct,
        ui_model_base=BaseModelType.MiniMaxH3,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Diffusers,
        title="Model",
    )
    transformer_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Optional single-file MiniMax H3 transformer (e.g. pruned int8) used in place "
        "of the main model's transformer. Text encoder and VAEs still come from the main model.",
        input=Input.Direct,
        ui_model_base=BaseModelType.MiniMaxH3,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Checkpoint,
        title="Transformer (single file)",
    )
    text_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Optional single-file MiniMax H3 Qwen3-VL text encoder (e.g. the truncated int8 "
        "repack) used in place of the main model's text encoder. The tokenizer and processor still "
        "come from the main model.",
        input=Input.Direct,
        ui_model_base=BaseModelType.MiniMaxH3,
        ui_model_type=ModelType.Qwen3VLEncoder,
        ui_model_format=ModelFormat.Checkpoint,
        title="Text Encoder (single file)",
    )

    def invoke(self, context: InvocationContext) -> MiniMaxH3ModelLoaderOutput:
        if not context.models.exists(self.model.key):
            raise ValueError(f"Unknown model: {self.model.key}")

        # Fail fast, and with a usable message, when a single-file main (e.g. the pruned int8
        # transformer repack) lands in the Model field — a checkpoint main carries none of the
        # folder submodels this node fans out, so letting it through would surface minutes
        # later as an opaque loader stack trace. The picker filters on ui_model_format, but
        # hand-authored workflows and clients that ignore the hint can still send one.
        main_config = context.models.get_config(self.model.key)
        if main_config.base is not BaseModelType.MiniMaxH3 or main_config.type is not ModelType.Main:
            raise ValueError(
                f"Model '{self.model.key}' is not a MiniMax H3 main model (resolved to "
                f"type={getattr(main_config.type, 'value', main_config.type)}, "
                f"base={getattr(main_config.base, 'value', main_config.base)})."
            )
        if main_config.format is not ModelFormat.Diffusers:
            raise ValueError(
                f"'{main_config.name}' is a single-file checkpoint and cannot be the Model input — that "
                "field needs the diffusers-folder install (e.g. 'MiniMax H3 Components'), which supplies "
                "the tokenizer, processor and VAEs. Single-file transformer/text-encoder repacks go in "
                "this node's 'Transformer (single file)' / 'Text Encoder (single file)' fields instead."
            )

        if self.transformer_model is not None:
            if not context.models.exists(self.transformer_model.key):
                raise ValueError(f"Unknown transformer model: {self.transformer_model.key}")
            transformer_config = context.models.get_config(self.transformer_model.key)
            transformer = self.transformer_model.model_copy(update={"submodel_type": SubModelType.Transformer})
        else:
            transformer_config = main_config
            transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        # Stamp the transformer's task variant onto the field: the denoise node uses it to
        # reject a task/conditioning mismatch (references on FL2VA weights, or Ref2VA weights
        # without references) instead of silently producing degraded output.
        variant = getattr(transformer_config, "variant", None)
        transformer_variant = variant.value if isinstance(variant, MiniMaxH3VariantType) else None
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        processor = self.model.model_copy(update={"submodel_type": SubModelType.Processor})
        if self.text_encoder_model is not None:
            if not context.models.exists(self.text_encoder_model.key):
                raise ValueError(f"Unknown text encoder model: {self.text_encoder_model.key}")
            text_encoder = self.text_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        audio_vae = self.model.model_copy(update={"submodel_type": SubModelType.AudioVAE})

        return MiniMaxH3ModelLoaderOutput(
            transformer=MiniMaxH3TransformerField(transformer=transformer, variant=transformer_variant),
            text_encoder=MiniMaxH3TextEncoderField(tokenizer=tokenizer, processor=processor, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
            audio_vae=VAEField(vae=audio_vae),
            model=self.model,
            transformer_model=self.transformer_model,
            text_encoder_model=self.text_encoder_model,
        )
