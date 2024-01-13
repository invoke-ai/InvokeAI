from typing import Any, Callable, Union

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.image import ImageField


# Define test invocations before importing anything that uses invocations
@invocation_output("test_list_output")
class ListPassThroughInvocationOutput(BaseInvocationOutput):
    collection: list[ImageField] = OutputField(default=[])


@invocation("test_list", version="1.0.0")
class ListPassThroughInvocation(BaseInvocation):
    collection: list[ImageField] = InputField(default=[])

    def invoke(self, context) -> ListPassThroughInvocationOutput:
        return ListPassThroughInvocationOutput(collection=self.collection)


@invocation_output("test_prompt_output")
class PromptTestInvocationOutput(BaseInvocationOutput):
    prompt: str = OutputField(default="")


@invocation("test_prompt", version="1.0.0")
class PromptTestInvocation(BaseInvocation):
    prompt: str = InputField(default="")

    def invoke(self, context) -> PromptTestInvocationOutput:
        return PromptTestInvocationOutput(prompt=self.prompt)


@invocation("test_error", version="1.0.0")
class ErrorInvocation(BaseInvocation):
    def invoke(self, context) -> PromptTestInvocationOutput:
        raise Exception("This invocation is supposed to fail")


@invocation_output("test_image_output")
class ImageTestInvocationOutput(BaseInvocationOutput):
    image: ImageField = OutputField()


@invocation("test_text_to_image", version="1.0.0")
class TextToImageTestInvocation(BaseInvocation):
    prompt: str = InputField(default="")
    prompt2: str = InputField(default="")

    def invoke(self, context) -> ImageTestInvocationOutput:
        return ImageTestInvocationOutput(image=ImageField(image_name=self.id))


@invocation("test_image_to_image", version="1.0.0")
class ImageToImageTestInvocation(BaseInvocation):
    prompt: str = InputField(default="")
    image: Union[ImageField, None] = InputField(default=None)

    def invoke(self, context) -> ImageTestInvocationOutput:
        return ImageTestInvocationOutput(image=ImageField(image_name=self.id))


@invocation_output("test_prompt_collection_output")
class PromptCollectionTestInvocationOutput(BaseInvocationOutput):
    collection: list[str] = OutputField(default=[])


@invocation("test_prompt_collection", version="1.0.0")
class PromptCollectionTestInvocation(BaseInvocation):
    collection: list[str] = InputField()

    def invoke(self, context) -> PromptCollectionTestInvocationOutput:
        return PromptCollectionTestInvocationOutput(collection=self.collection.copy())


@invocation_output("test_any_output")
class AnyTypeTestInvocationOutput(BaseInvocationOutput):
    value: Any = OutputField()


@invocation("test_any", version="1.0.0")
class AnyTypeTestInvocation(BaseInvocation):
    value: Any = InputField(default=None)

    def invoke(self, context) -> AnyTypeTestInvocationOutput:
        return AnyTypeTestInvocationOutput(value=self.value)


@invocation("test_polymorphic", version="1.0.0")
class PolymorphicStringTestInvocation(BaseInvocation):
    value: Union[str, list[str]] = InputField(default="")

    def invoke(self, context) -> PromptCollectionTestInvocationOutput:
        if isinstance(self.value, str):
            return PromptCollectionTestInvocationOutput(collection=[self.value])
        return PromptCollectionTestInvocationOutput(collection=self.value)


# Importing these must happen after test invocations are defined or they won't register
from invokeai.app.services.events.events_base import EventServiceBase  # noqa: E402
from invokeai.app.services.shared.graph import Edge, EdgeConnection  # noqa: E402


def create_edge(from_id: str, from_field: str, to_id: str, to_field: str) -> Edge:
    return Edge(
        source=EdgeConnection(node_id=from_id, field=from_field),
        destination=EdgeConnection(node_id=to_id, field=to_field),
    )


class TestEvent:
    event_name: str
    payload: Any
    __test__ = False  # not a pytest test case

    def __init__(self, event_name: str, payload: Any):
        self.event_name = event_name
        self.payload = payload


class TestEventService(EventServiceBase):
    events: list
    __test__ = False  # not a pytest test case

    def __init__(self):
        super().__init__()
        self.events = []

    def dispatch(self, event_name: str, payload: Any) -> None:
        pass


def wait_until(condition: Callable[[], bool], timeout: int = 10, interval: float = 0.1) -> None:
    import time

    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition():
            return
        time.sleep(interval)
    raise TimeoutError("Condition not met")
