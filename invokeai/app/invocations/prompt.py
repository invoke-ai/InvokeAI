import os
from typing import Literal, Optional

from pydantic.fields import Field
from pyparsing import ParseException

from .baseinvocation import BaseInvocation, BaseInvocationOutput, InvocationContext
from dynamicprompts import RandomPromptGenerator, CombinatorialPromptGenerator

class PromptOutput(BaseInvocationOutput):
    """Base class for invocations that output a prompt"""
    #fmt: off
    type: Literal["prompt"] = "prompt"

    prompt: str = Field(default=None, description="The output prompt")
    #fmt: on

    class Config:
        schema_extra = {
            'required': [
                'type',
                'prompt',
            ]
        }


class PromptListOutput(BaseInvocationOutput):
    """Base class for invocations that output a list of prompts"""

    # fmt: off
    type: Literal["prompt_list"] = "prompt_list"

    prompts: list[str] = Field(description="The output prompts")
    count: int = Field(description="The size of the prompts list")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "prompts", "count"]}


class DynamicPromptInvocation(BaseInvocation):
    """Parses a prompt using adieyal/dynamicprompts' random or combinatorial generator"""

    type: Literal["dynamic_prompt"] = "dynamic_prompt"
    prompt: str = Field(
        default=None, description="The prompt to parse with dynamicprompts"
    )
    max_prompts: int = Field(default=1, description="The number of prompts to generate")
    combinatorial: bool = Field(
        default=False, description="Whether to use the combinatorial generator"
    )
    # wildcard_path: Optional[str] = Field(default=None, description="Wildcard path")

    def invoke(self, context: InvocationContext) -> PromptListOutput:
        # if self.wildcard_path is not None:
        #     try:
        #         os.stat(self.wildcard_path)
        #     except FileNotFoundError:
        #         context.services.logger.warn(f"Invalid wildcard path ({self.wildcard_path}), ignoring")
        try:
            if self.combinatorial:
                generator = CombinatorialPromptGenerator()
                prompts = generator.generate(self.prompt, max_prompts=self.max_prompts)
            else:
                generator = RandomPromptGenerator()
                prompts = generator.generate(self.prompt, num_images=self.max_prompts)
        except ParseException as e:
            warning = f"Invalid dynamic prompt: {e}"
            context.services.logger.warn(warning)
            graph_execution_state = context.services.graph_execution_manager.get(
                context.graph_execution_state_id
            )
            source_node_id = graph_execution_state.prepared_source_mapping[self.id]
            context.services.events.emit_invocation_warning(
                warning=warning,
                graph_execution_state_id=context.graph_execution_state_id,
                node=self.dict(),
                source_node_id=source_node_id,
            )
            return PromptListOutput(prompts=[self.prompt], count=1)

        return PromptListOutput(prompts=prompts, count=len(prompts))
