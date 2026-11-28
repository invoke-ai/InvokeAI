import re
from collections.abc import Iterable, Mapping
from typing import Optional

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import (
    PROMPT_TEMPLATE_FIELD_NAME_PATTERN,
    Input,
    InputField,
    OutputField,
    PromptTemplateValueField,
    StylePresetField,
    UIComponent,
    check_prompt_template_value_name,
)
from invokeai.app.services.shared.invocation_context import InvocationContext

# The optional `$` marks a dynamic-prompt variable (`${name}`), which is not a placeholder.
_TEMPLATE_TOKEN_RE = re.compile(rf"\$?\{{({PROMPT_TEMPLATE_FIELD_NAME_PATTERN})\}}")
_PROMPT_NAME = "prompt"


def collect_prompt_template_values(values: Iterable[PromptTemplateValueField]) -> dict[str, str]:
    """The values by lowercased name. Two values for one name have no order to choose by, so they are an error."""
    collected: dict[str, str] = {}
    for value in values:
        name = value.name.lower()
        if name in collected:
            raise ValueError(f"The prompt template was given more than one value for '{name}'.")
        collected[name] = value.value
    return collected


def render_prompt_template(template: str, prompt: str, values: Mapping[str, str]) -> str:
    """Fills a template's `{prompt}` placeholders with the prompt and its `{name}` placeholders with `values`, keyed
    by lowercased name. A placeholder without a value leaves nothing behind.

    Everything is placed in one pass, so a prompt or value that itself contains `{prompt}` or `{name}` stays as
    given. Placeholder names follow `core/promptTemplates.ts` in the web frontend's generation feature; `{prompt}`
    differs on purpose: here every copy is replaced and a template without one does not get the prompt appended.
    """

    def replace(match: re.Match[str]) -> str:
        token, name = match.group(0), match.group(1)
        is_variable = token.startswith("$")
        if name == _PROMPT_NAME:
            return f"${prompt}" if is_variable else prompt
        if is_variable or name.lower() == _PROMPT_NAME:
            return token
        return values.get(name.lower(), "")

    return _TEMPLATE_TOKEN_RE.sub(replace, template)


@invocation_output("prompt_template_output")
class PromptTemplateOutput(BaseInvocationOutput):
    """Output for the Prompt Template node"""

    positive_prompt: str = OutputField(description="The positive prompt with the template applied")
    negative_prompt: str = OutputField(description="The negative prompt with the template applied")


@invocation(
    "prompt_template",
    title="Prompt Template",
    tags=["prompt", "template", "style", "preset"],
    category="prompt",
    version="1.1.0",
)
class PromptTemplateInvocation(BaseInvocation):
    """Applies a Style Preset template to positive and negative prompts.

    Select a Style Preset and provide positive/negative prompts. The node replaces
    {prompt} placeholders in the template with your input prompts, and the template's
    other {name} placeholders with the connected Prompt Template Values.
    """

    style_preset: StylePresetField = InputField(
        description="The Style Preset to use as a template",
    )
    positive_prompt: str = InputField(
        default="",
        description="The positive prompt to insert into the template's {prompt} placeholder",
        ui_component=UIComponent.Textarea,
    )
    negative_prompt: str = InputField(
        default="",
        description="The negative prompt to insert into the template's {prompt} placeholder",
        ui_component=UIComponent.Textarea,
    )
    values: Optional[PromptTemplateValueField | list[PromptTemplateValueField]] = InputField(
        default=None,
        description="Text for the template's other {name} placeholders. May be a single value or a collection. "
        "A placeholder without a value is removed.",
        title="Values",
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> PromptTemplateOutput:
        style_preset = context._services.style_preset_records.get(self.style_preset.style_preset_id)
        if self.values is None:
            given: list[PromptTemplateValueField] = []
        else:
            given = self.values if isinstance(self.values, list) else [self.values]
        values = collect_prompt_template_values(given)
        templates = style_preset.preset_data

        return PromptTemplateOutput(
            positive_prompt=render_prompt_template(templates.positive_prompt, self.positive_prompt, values),
            negative_prompt=render_prompt_template(templates.negative_prompt, self.negative_prompt, values),
        )


@invocation_output("prompt_template_value_output")
class PromptTemplateValueOutput(BaseInvocationOutput):
    """Output for the Prompt Template Value node"""

    value: PromptTemplateValueField = OutputField(
        description="The placeholder's name and text, for a Prompt Template's Values", title="Value"
    )


@invocation(
    "prompt_template_value",
    title="Prompt Template Value",
    tags=["prompt", "template", "style", "preset"],
    category="prompt",
    version="1.0.0",
)
class PromptTemplateValueInvocation(BaseInvocation):
    """Sets the text for one of a prompt template's {name} placeholders.

    Connect it, or a collection of them, to a Prompt Template's Values.
    """

    name: str = InputField(
        description="The placeholder's name without braces: letters A-Z, numbers and underscores, starting with a "
        "letter. Upper and lower case are the same name.",
    )
    value: str = InputField(default="", description="The text that replaces the placeholder")

    def invoke(self, context: InvocationContext) -> PromptTemplateValueOutput:
        # Checked here so a bad name fails with the rule alone, not pydantic's report of it.
        name = check_prompt_template_value_name(self.name)
        return PromptTemplateValueOutput(value=PromptTemplateValueField(name=name, value=self.value))
