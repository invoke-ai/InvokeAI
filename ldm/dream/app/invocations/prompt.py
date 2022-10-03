from typing import Literal
from pydantic.fields import Field
from .baseinvocation import BaseInvocationOutput

class PromptOutput(BaseInvocationOutput):
    """Base class for invocations that output a prompt"""
    type: Literal['prompt'] = 'prompt'

    prompt: str = Field(default=None, description="The output prompt")
