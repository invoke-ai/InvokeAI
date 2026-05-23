from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.invocations.model import (
    Mistral3EncoderField,
    ModelIdentifierField,
    PromptEnhancerField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("ernie_image_model_loader_output")
class ErnieImageModelLoaderOutput(BaseInvocationOutput):
    """ERNIE-Image model loader output."""

    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    text_encoder: Mistral3EncoderField = OutputField(description="Mistral3 text encoder", title="Text Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")
    prompt_enhancer: PromptEnhancerField | None = OutputField(
        default=None,
        description="Optional prompt-enhancer (Ministral3ForCausalLM)",
        title="Prompt Enhancer",
    )


@invocation(
    "ernie_image_model_loader",
    title="Main Model - ERNIE-Image",
    tags=["model", "ernie-image"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ErnieImageModelLoaderInvocation(BaseInvocation):
    """Loads an ERNIE-Image diffusers pipeline and emits its submodels.

    The transformer, VAE, text encoder, and (optional) prompt enhancer are all expected to
    live inside the same diffusers pipeline directory under their conventional subfolders
    (`transformer/`, `vae/`, `text_encoder/`, `tokenizer/`, `pe/`, `pe_tokenizer/`).
    """

    model: ModelIdentifierField = InputField(
        description="ERNIE-Image diffusers pipeline (provides transformer, VAE, text encoder, and optional prompt enhancer)",
        input=Input.Direct,
        ui_model_base=BaseModelType.ErnieImage,
        ui_model_type=ModelType.Main,
        title="ERNIE-Image Model",
    )

    use_prompt_enhancer: bool = InputField(
        default=True,
        description="If true and the pipeline ships with a prompt-enhancer submodel, expose it on the output.",
        title="Use Prompt Enhancer",
    )

    def invoke(self, context: InvocationContext) -> ErnieImageModelLoaderOutput:
        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})

        prompt_enhancer: PromptEnhancerField | None = None
        if self.use_prompt_enhancer and self._pipeline_has_prompt_enhancer(context):
            pe_tok = self.model.model_copy(update={"submodel_type": SubModelType.PromptEnhancerTokenizer})
            pe_lm = self.model.model_copy(update={"submodel_type": SubModelType.PromptEnhancer})
            prompt_enhancer = PromptEnhancerField(tokenizer=pe_tok, text_encoder=pe_lm)

        return ErnieImageModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            text_encoder=Mistral3EncoderField(tokenizer=tokenizer, text_encoder=text_encoder),
            vae=VAEField(vae=vae),
            prompt_enhancer=prompt_enhancer,
        )

    def _pipeline_has_prompt_enhancer(self, context: InvocationContext) -> bool:
        """Check whether the pipeline directory ships a prompt-enhancer."""
        from pathlib import Path

        config = context.models.get_config(self.model)
        pe_dir = Path(config.path) / "pe"
        return pe_dir.is_dir()
