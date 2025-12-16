from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import InputField, OutputField, UIComponent
from invokeai.app.services.shared.invocation_context import InvocationContext


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
    version="1.0.0",
)
class PromptTemplateInvocation(BaseInvocation):
    """Applies a Style Preset template to positive and negative prompts.

    Select a Style Preset and provide positive/negative prompts. The node replaces
    {prompt} placeholders in the template with your input prompts.
    """

    style_preset_id: str = InputField(
        description="The ID of the Style Preset to use as a template",
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

    def invoke(self, context: InvocationContext) -> PromptTemplateOutput:
        # Fetch the style preset from the database
        style_preset = context._services.style_preset_records.get(self.style_preset_id)

        # Get the template prompts
        positive_template = style_preset.preset_data.positive_prompt
        negative_template = style_preset.preset_data.negative_prompt

        # Replace {prompt} placeholder with the input prompts
        rendered_positive = positive_template.replace("{prompt}", self.positive_prompt)
        rendered_negative = negative_template.replace("{prompt}", self.negative_prompt)

        return PromptTemplateOutput(
            positive_prompt=rendered_positive,
            negative_prompt=rendered_negative,
        )
