from typing import Any, Optional

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
    Qwen3EncoderField,
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


@invocation_output("ideogram4_model_loader_output")
class Ideogram4ModelLoaderOutput(BaseInvocationOutput):
    """Ideogram 4 model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    unconditional_transformer: TransformerField | None = OutputField(
        default=None,
        description="The unconditional branch, when it is a separate single-file model. None for a "
        "diffusers pipeline, which carries both branches in one submodel.",
        title="Transformer (Unconditional)",
    )
    qwen3_encoder: Qwen3EncoderField = OutputField(
        description=FieldDescriptions.qwen3_encoder, title="Qwen3-VL Encoder"
    )
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "ideogram4_model_loader",
    title="Main Model - Ideogram 4",
    tags=["model", "ideogram4"],
    category="model",
    version="1.1.0",
    classification=Classification.Prototype,
)
class Ideogram4ModelLoaderInvocation(BaseInvocation):
    """Loads an Ideogram 4 model, outputting its submodels.

    Two on-disk layouts are supported:

    * A diffusers pipeline bundles everything - both transformer branches, the Qwen3-VL encoder
      + tokenizer, and the VAE - and needs nothing else selected.
    * Comfy-Org's single files hold ONE branch each, so the conditional file goes in `Model`, the
      unconditional one in `Transformer (Unconditional)`, and the Qwen3-VL 8B encoder and the
      32-channel VAE are selected here as well.

    Both branches run at every denoising step, so the pair is not optional.
    """

    model: ModelIdentifierField = InputField(
        description="The Ideogram 4 model to load. A diffusers pipeline provides every submodel; a "
        "single-file checkpoint is the conditional transformer alone and needs the components below.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Ideogram4,
        ui_model_type=ModelType.Main,
        title="Model",
    )

    unconditional_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="The unconditional branch as a single-file checkpoint. Required when Model is a "
        "single file; ignored for a diffusers pipeline, which bundles both branches.",
        input=Input.Direct,
        ui_model_base=BaseModelType.Ideogram4,
        ui_model_type=ModelType.Main,
        ui_model_format=ModelFormat.Checkpoint,
        title="Transformer (Unconditional)",
    )

    qwen3_vl_encoder_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone Qwen3-VL 8B encoder. Required when Model is a single-file checkpoint; "
        "otherwise an override for the encoder bundled in the diffusers pipeline.",
        input=Input.Direct,
        # Base `any` excludes MiniMax H3's truncated Qwen3-VL-32B, which shares this model type and
        # could otherwise be picked here only to be refused on the way in.
        ui_model_base=BaseModelType.Any,
        ui_model_type=ModelType.Qwen3VLEncoder,
        title="Qwen3-VL Encoder",
    )

    vae_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Standalone VAE (the 32-channel one Ideogram 4 shares with FLUX.2). Required when "
        "Model is a single-file checkpoint; otherwise an override for the bundled VAE.",
        input=Input.Direct,
        ui_model_base=accepted_vae_bases(BaseModelType.Ideogram4),
        ui_model_type=ModelType.VAE,
        title="VAE",
    )

    def invoke(self, context: InvocationContext) -> Ideogram4ModelLoaderOutput:
        main_config = context.models.get_config(self.model)
        if main_config.base is not BaseModelType.Ideogram4 or main_config.type is not ModelType.Main:
            raise ValueError(
                f"Model '{main_config.name}' is not an Ideogram 4 main model. Select an Ideogram 4 transformer."
            )
        is_single_file = main_config.format is ModelFormat.Checkpoint

        unconditional_transformer: TransformerField | None = None
        if is_single_file:
            self._raise_for_wrong_branch(main_config, expected="conditional", slot="Model")
            unconditional_transformer = TransformerField(transformer=self._resolve_unconditional(context), loras=[])
        elif self.unconditional_model is not None:
            # The field's own docs promise this input is ignored for a bundled pipeline - e.g. a
            # leftover wire from a single-file session.
            context.logger.warning(
                "'Transformer (Unconditional)' is ignored for a diffusers Ideogram 4 pipeline, which "
                "carries both branches."
            )

        tokenizer, text_encoder = self._resolve_encoder(context, required=is_single_file)
        vae = self._resolve_vae(context, required=is_single_file)

        return Ideogram4ModelLoaderOutput(
            transformer=TransformerField(
                transformer=self.model.model_copy(update={"submodel_type": SubModelType.Transformer}), loras=[]
            ),
            unconditional_transformer=unconditional_transformer,
            qwen3_encoder=Qwen3EncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
        )

    def _resolve_unconditional(self, context: InvocationContext) -> ModelIdentifierField:
        if self.unconditional_model is None:
            raise ValueError(
                "A single-file Ideogram 4 transformer is one of two branches, so "
                "'Transformer (Unconditional)' must be selected as well. Install "
                "'ideogram4_unconditional_*.safetensors' from the same release, or select the diffusers "
                "pipeline, which carries both."
            )
        if self.unconditional_model.key == self.model.key:
            raise ValueError(
                "The same model is wired to both 'Model' and 'Transformer (Unconditional)'. Ideogram 4 "
                "guides one branch against the other, so two different files are required."
            )
        config = context.models.get_config(self.unconditional_model)
        if config.base is not BaseModelType.Ideogram4 or config.type is not ModelType.Main:
            raise ValueError(f"'{config.name}' is not an Ideogram 4 main model and cannot be the unconditional branch.")
        if config.format is not ModelFormat.Checkpoint:
            raise ValueError(
                f"'Transformer (Unconditional)' must be a single-file Ideogram 4 checkpoint. '{config.name}' "
                f"is in {config.format.value} format, which bundles both branches already."
            )
        self._raise_for_wrong_branch(config, expected="unconditional", slot="Transformer (Unconditional)")
        return self.unconditional_model.model_copy(update={"submodel_type": SubModelType.Transformer})

    @staticmethod
    def _raise_for_wrong_branch(config: Any, *, expected: str, slot: str) -> None:
        """Reject a swapped pair.

        The branch is read from the file's own `model_type` metadata, so unlike Wan's
        filename-derived expert tags it is not a guess and the wiring does not get to override it:
        the two branches are otherwise identical, and swapping them produces coherent images that
        are simply not the ones the prompt asked for - with nothing in the log to say so.
        """
        branch = getattr(config, "branch", None)
        if branch != expected:
            raise ValueError(
                f"'{config.name}' is the {branch} branch of Ideogram 4 and cannot be used as '{slot}', "
                f"which needs the {expected} one. If the two are wired the wrong way round, swap them. "
                "A file whose metadata was stripped is classified by its name, so if this one was "
                f"renamed, rename it to contain '{expected}' (or not) and re-install it."
            )

    def _resolve_encoder(
        self, context: InvocationContext, *, required: bool
    ) -> tuple[ModelIdentifierField, ModelIdentifierField]:
        if self.qwen3_vl_encoder_model is None:
            if required:
                raise ValueError(
                    "A single-file Ideogram 4 transformer carries no text encoder. Select the standalone "
                    "Qwen3-VL 8B encoder on this node, or use the diffusers pipeline, which bundles one."
                )
            return (
                self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
                self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
            )

        config = context.models.get_config(self.qwen3_vl_encoder_model)
        if config.type is not ModelType.Qwen3VLEncoder:
            raise ValueError(f"'{config.name}' is not a Qwen3-VL encoder.")
        # Both Qwen3-VL encoders install under the same type, and Krea-2's is the 4B. Ideogram 4
        # taps 13 layers of the 8B for a 53248-wide feature vector; the 4B would produce 33280 and
        # fail as a shape mismatch inside the first denoising step.
        variant = getattr(config, "variant", None)
        if variant is not Qwen3VLVariantType.Qwen3VL_8B:
            raise ValueError(
                f"'{config.name}' is the Qwen3-VL {_variant_label(variant)} encoder. Ideogram 4 conditions "
                "on the 8B one (hidden size 4096)."
            )
        return (
            self.qwen3_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
            self.qwen3_vl_encoder_model.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
        )

    def _resolve_vae(self, context: InvocationContext, *, required: bool) -> ModelIdentifierField:
        if self.vae_model is None:
            if required:
                raise ValueError(
                    "A single-file Ideogram 4 transformer carries no VAE. Select the standalone 32-channel "
                    "VAE on this node, or use the diffusers pipeline, which bundles one."
                )
            return self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        config = context.models.get_config(self.vae_model)
        if config.type is not ModelType.VAE or not accepts_vae(
            BaseModelType.Ideogram4, config.base, getattr(config, "latent_channels", None)
        ):
            accepted = " or ".join(base.value for base in accepted_vae_bases(BaseModelType.Ideogram4))
            raise ValueError(f"VAE '{config.name}' is not compatible with Ideogram 4. Select a VAE of base {accepted}.")
        return self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})
