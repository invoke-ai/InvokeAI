from typing import Any, Callable, Union
from unittest.mock import MagicMock

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.image import ImageField
from invokeai.app.services.events.events_common import EventBase
from invokeai.app.services.session_processor.session_processor_common import ProgressImage
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.shared.invocation_context import InvocationContext


# Define test invocations before importing anything that uses invocations
@invocation_output("test_list_output")
class ListPassThroughInvocationOutput(BaseInvocationOutput):
    collection: list[ImageField] = OutputField(default=[])


@invocation("test_list", version="1.0.0")
class ListPassThroughInvocation(BaseInvocation):
    collection: list[ImageField] = InputField(default=[])

    def invoke(self, context: InvocationContext) -> ListPassThroughInvocationOutput:
        return ListPassThroughInvocationOutput(collection=self.collection)


@invocation_output("test_prompt_output")
class PromptTestInvocationOutput(BaseInvocationOutput):
    prompt: str = OutputField(default="")


@invocation("test_prompt", version="1.0.0")
class PromptTestInvocation(BaseInvocation):
    prompt: str = InputField(default="")

    def invoke(self, context: InvocationContext) -> PromptTestInvocationOutput:
        return PromptTestInvocationOutput(prompt=self.prompt)


@invocation("test_error", version="1.0.0")
class ErrorInvocation(BaseInvocation):
    def invoke(self, context: InvocationContext) -> PromptTestInvocationOutput:
        raise Exception("This invocation is supposed to fail")


@invocation_output("test_image_output")
class ImageTestInvocationOutput(BaseInvocationOutput):
    image: ImageField = OutputField()


@invocation("test_text_to_image", version="1.0.0")
class TextToImageTestInvocation(BaseInvocation):
    prompt: str = InputField(default="")
    prompt2: str = InputField(default="")

    def invoke(self, context: InvocationContext) -> ImageTestInvocationOutput:
        return ImageTestInvocationOutput(image=ImageField(image_name=self.id))


@invocation("test_image_to_image", version="1.0.0")
class ImageToImageTestInvocation(BaseInvocation):
    prompt: str = InputField(default="")
    image: Union[ImageField, None] = InputField(default=None)

    def invoke(self, context: InvocationContext) -> ImageTestInvocationOutput:
        return ImageTestInvocationOutput(image=ImageField(image_name=self.id))


@invocation_output("test_prompt_collection_output")
class PromptCollectionTestInvocationOutput(BaseInvocationOutput):
    collection: list[str] = OutputField(default=[])


@invocation("test_prompt_collection", version="1.0.0")
class PromptCollectionTestInvocation(BaseInvocation):
    collection: list[str] = InputField()

    def invoke(self, context: InvocationContext) -> PromptCollectionTestInvocationOutput:
        return PromptCollectionTestInvocationOutput(collection=self.collection.copy())


@invocation_output("test_any_output")
class AnyTypeTestInvocationOutput(BaseInvocationOutput):
    value: Any = OutputField()


@invocation("test_any", version="1.0.0")
class AnyTypeTestInvocation(BaseInvocation):
    value: Any = InputField(default=None)

    def invoke(self, context: InvocationContext) -> AnyTypeTestInvocationOutput:
        return AnyTypeTestInvocationOutput(value=self.value)


@invocation("test_polymorphic", version="1.0.0")
class PolymorphicStringTestInvocation(BaseInvocation):
    value: Union[str, list[str]] = InputField(default="")

    def invoke(self, context: InvocationContext) -> PromptCollectionTestInvocationOutput:
        if isinstance(self.value, str):
            return PromptCollectionTestInvocationOutput(collection=[self.value])
        return PromptCollectionTestInvocationOutput(collection=self.value)


# Importing these must happen after test invocations are defined or they won't register
from invokeai.app.services.events.events_base import EventServiceBase  # noqa: E402
from invokeai.app.services.shared.graph import Edge, EdgeConnection, GraphExecutionState  # noqa: E402


def create_edge(from_id: str, from_field: str, to_id: str, to_field: str) -> Edge:
    return Edge(
        source=EdgeConnection(node_id=from_id, field=from_field),
        destination=EdgeConnection(node_id=to_id, field=to_field),
    )


class TestEvent(EventBase):
    __test__ = False  # not a pytest test case

    __event_name__ = "test_event"


class TestEventService(EventServiceBase):
    __test__ = False  # not a pytest test case

    def __init__(self):
        super().__init__()
        self.events: list[EventBase] = []

    def dispatch(self, event: EventBase) -> None:
        self.events.append(event)
        pass

    def emit_invocation_progress(
        self,
        queue_item: "SessionQueueItem",
        invocation: "BaseInvocation",
        message: str,
        percentage: float | None = None,
        image: "ProgressImage | None" = None,
    ) -> None:
        pass


def wait_until(condition: Callable[[], bool], timeout: int = 10, interval: float = 0.1) -> None:
    import time

    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition():
            return
        time.sleep(interval)
    raise TimeoutError("Condition not met")


def run_session_with_mock_context(session: GraphExecutionState):
    """Run the session with a mock context to simulate invocation execution.

    The graph may only contain invocations that operate on primitive types. Images, models, or any other types that
    require a real context cannot be used in this mock execution.
    """
    mock_context = MagicMock(spec=InvocationContext)
    invocation = session.next()
    while invocation is not None:
        output = invocation.invoke(mock_context)
        session.complete(invocation.id, output)
        invocation = session.next()


def get_single_output_from_session(session: GraphExecutionState, node_id: str) -> BaseInvocationOutput:
    assert len(session.source_prepared_mapping[node_id]) == 1, (
        "Expected exactly one prepared node for the given node_id"
    )
    prepared_node_id = session.source_prepared_mapping[node_id].pop()
    output = session.results[prepared_node_id]
    assert isinstance(output, BaseInvocationOutput), "Expected output to be of type BaseInvocationOutput"
    return output
