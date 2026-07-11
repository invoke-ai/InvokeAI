import torch
from transformers import AutoTokenizer

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, InputField, SystemPromptField, UIComponent
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.taxonomy import ModelType
from invokeai.backend.text_llm_pipeline import DEFAULT_SYSTEM_PROMPT, TextLLMPipeline
from invokeai.backend.util.devices import TorchDevice


def _run_text_llm(
    context: InvocationContext,
    text_llm_model: ModelIdentifierField,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
) -> str:
    """Shared LLM invocation body used by every text-LLM node in this module."""
    model_config = context.models.get_config(text_llm_model)

    with context.models.load(text_llm_model).model_on_device() as (_, model):
        model_abs_path = context.models.get_absolute_path(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_abs_path, local_files_only=True)

        pipeline = TextLLMPipeline(model, tokenizer)
        model_device = next(model.parameters()).device
        return pipeline.run(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            device=model_device,
            dtype=TorchDevice.choose_torch_dtype(),
        )


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
        output = _run_text_llm(
            context=context,
            text_llm_model=self.text_llm_model,
            prompt=self.prompt,
            system_prompt=self.system_prompt,
            max_tokens=self.max_tokens,
        )
        return StringOutput(value=output)


@invocation(
    "text_llm_with_preset",
    title="Text LLM (with System Prompt Preset)",
    tags=["llm", "text", "prompt", "preset", "template"],
    category="llm",
    version="1.0.0",
    classification=Classification.Beta,
)
class TextLLMWithPresetInvocation(BaseInvocation):
    """Run a text language model using a saved system prompt from the System Prompts library.

    Behaves identically to the Text LLM node, but the system prompt is selected from a
    DB-backed preset instead of being typed inline. Useful when you maintain a curated
    library of expansion strategies and want to reuse them across workflows.
    """

    prompt: str = InputField(
        default="",
        description="Input text prompt.",
        ui_component=UIComponent.Textarea,
    )
    system_prompt: SystemPromptField = InputField(
        description="The saved system prompt to use as the LLM's instruction.",
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
        record = context._services.system_prompt_records.get(self.system_prompt.system_prompt_id)
        output = _run_text_llm(
            context=context,
            text_llm_model=self.text_llm_model,
            prompt=self.prompt,
            system_prompt=record.content,
            max_tokens=self.max_tokens,
        )
        return StringOutput(value=output)
