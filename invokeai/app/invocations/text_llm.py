import torch
from transformers import AutoTokenizer

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, InputField, UIComponent
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import ModelType
from invokeai.backend.text_llm_pipeline import DEFAULT_SYSTEM_PROMPT, TextLLMPipeline
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "text_llm",
    title="Text LLM",
    tags=["llm", "text", "prompt"],
    category="llm",
    version="1.0.0",
    classification=Classification.Beta,
)
class TextLLMInvocation(BaseInvocation):
    """Run a text language model to generate or expand text (e.g. for prompt expansion)."""

    prompt: str = InputField(
        default="",
        description="Input text prompt.",
        ui_component=UIComponent.Textarea,
    )
    system_prompt: str = InputField(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt that guides the model's behavior.",
        ui_component=UIComponent.Textarea,
    )
    text_llm_model: ModelIdentifierField = InputField(
        title="Text LLM Model",
        description=FieldDescriptions.text_llm_model,
        ui_model_type=ModelType.TextLLM,
    )
    max_tokens: int = InputField(
        default=300,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate.",
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> StringOutput:
        model_config = context.models.get_config(self.text_llm_model)

        with context.models.load(self.text_llm_model).model_on_device() as (_, model):
            model_abs_path = context.models.get_absolute_path(model_config)
            tokenizer = AutoTokenizer.from_pretrained(model_abs_path, local_files_only=True)

            pipeline = TextLLMPipeline(model, tokenizer)
            model_device = next(model.parameters()).device
            output = pipeline.run(
                prompt=self.prompt,
                system_prompt=self.system_prompt,
                max_new_tokens=self.max_tokens,
                device=model_device,
                dtype=TorchDevice.choose_torch_dtype(),
            )

        return StringOutput(value=output)
