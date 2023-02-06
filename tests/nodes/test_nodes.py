from typing import Literal
from ldm.invoke.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from ldm.invoke.app.invocations.image import ImageField
from ldm.invoke.app.services.invocation_services import InvocationServices
from pydantic import Field
import pytest

# Define test invocations before importing anything that uses invocations
class ListPassThroughInvocationOutput(BaseInvocationOutput):
    type: Literal['test_list_output'] = 'test_list_output'

    collection: list[ImageField] = Field(default_factory=list)

class ListPassThroughInvocation(BaseInvocation):
    type: Literal['test_list'] = 'test_list'

    collection: list[ImageField] = Field(default_factory=list)

    def invoke(self, services: InvocationServices, session_id: str) -> ListPassThroughInvocationOutput:
        return ListPassThroughInvocationOutput(collection = self.collection)

class PromptTestInvocationOutput(BaseInvocationOutput):
    type: Literal['test_prompt_output'] = 'test_prompt_output'

    prompt: str = Field(default = "")

class PromptTestInvocation(BaseInvocation):
    type: Literal['test_prompt'] = 'test_prompt'

    prompt: str = Field(default = "")

    def invoke(self, services: InvocationServices, session_id: str) -> PromptTestInvocationOutput:
        return PromptTestInvocationOutput(prompt = self.prompt)

class ImageTestInvocationOutput(BaseInvocationOutput):
    type: Literal['test_image_output'] = 'test_image_output'

    image: ImageField = Field()

class ImageTestInvocation(BaseInvocation):
    type: Literal['test_image'] = 'test_image'

    prompt: str = Field(default = "")

    def invoke(self, services: InvocationServices, session_id: str) -> PromptTestInvocationOutput:
        return ImageTestInvocationOutput(image=ImageField(image_name=self.id))

class PromptCollectionTestInvocationOutput(BaseInvocationOutput):
    type: Literal['test_prompt_collection_output'] = 'test_prompt_collection_output'
    collection: list[str] = Field(default_factory=list)

class PromptCollectionTestInvocation(BaseInvocation):
    type: Literal['test_prompt_collection'] = 'test_prompt_collection'
    collection: list[str] = Field()

    def invoke(self, services: InvocationServices, session_id: str) -> PromptCollectionTestInvocationOutput:
        return PromptCollectionTestInvocationOutput(collection=self.collection.copy())
