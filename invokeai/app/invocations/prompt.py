
from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator

from invokeai.app.invocations.primitives import StringCollectionOutput

from .baseinvocation import BaseInvocation, InputField, InvocationContext, UIComponent, invocation


@invocation("dynamic_prompt", title="Dynamic Prompt", tags=["prompt", "collection"], category="prompt")
class DynamicPromptInvocation(BaseInvocation):
    """Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator"""

    prompt: str = InputField(description="The prompt to parse with dynamicprompts", ui_component=UIComponent.Textarea)
    max_prompts: int = InputField(default=1, description="The number of prompts to generate")
    combinatorial: bool = InputField(default=False, description="Whether to use the combinatorial generator")

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        if self.combinatorial:
            generator = CombinatorialPromptGenerator()
            prompts = generator.generate(self.prompt, max_prompts=self.max_prompts)
        else:
            generator = RandomPromptGenerator()
            prompts = generator.generate(self.prompt, num_images=self.max_prompts)

        return StringCollectionOutput(collection=prompts)