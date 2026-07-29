import torch
from transformers import AutoTokenizer

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import FieldDescriptions, InputField, SystemPromptField, UIComponent
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.primitives import StringOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    SYSTEM_PROMPT_DEFAULT_USER_ID,
    SystemPromptNotFoundError,
    SystemPromptRecordDTO,
)
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

    Note: the field stores the preset's id, not its text. A workflow exported from one install
    only resolves on another if that install has a prompt with the same id -- true for the
    seeded defaults (fixed UUIDs), not for user-created prompts. `StylePresetField` has the
    same limitation.
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

    def _resolve_system_prompt(self, context: InvocationContext) -> SystemPromptRecordDTO:
        """Resolve the referenced preset, enforcing the same access rules as the REST API.

        The record store is unscoped, so without this check a user could read another user's
        private prompt by enqueueing a graph that references its id -- the content becomes the
        LLM's system message and is recoverable from the node's output. Mirrors the ownership
        check in `call_saved_workflow.CallSavedWorkflowInvocation.validate_selected_workflow`.
        """
        system_prompt_id = self.system_prompt.system_prompt_id
        try:
            record = context._services.system_prompt_records.get(system_prompt_id)
        except SystemPromptNotFoundError as e:
            raise ValueError(f"The selected system prompt '{system_prompt_id}' could not be found.") from e

        config = context._services.configuration
        if config.multiuser:
            queue_user_id = context._data.queue_item.user_id
            user = context._services.users.get(queue_user_id)
            is_admin = bool(user and user.is_admin)
            is_owner = record.user_id == queue_user_id
            is_default = record.user_id == SYSTEM_PROMPT_DEFAULT_USER_ID
            if not (is_default or is_owner or record.is_public or is_admin):
                raise ValueError(f"The selected system prompt '{system_prompt_id}' is not accessible to this user.")

        return record

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> StringOutput:
        record = self._resolve_system_prompt(context)
        output = _run_text_llm(
            context=context,
            text_llm_model=self.text_llm_model,
            prompt=self.prompt,
            system_prompt=record.content,
            max_tokens=self.max_tokens,
        )
        return StringOutput(value=output)
