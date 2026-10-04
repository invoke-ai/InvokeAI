from pydantic import BaseModel, Field

from invokeai.app.services.events.events_common import InvocationProgressEvent
from invokeai.app.services.session_processor.session_processor_common import ProgressImage


class ProgressPreviewDTO(BaseModel):
    """The latest preview frame of a running queue item, as served by `GET /queue/{queue_id}/previews`.

    The fields of the `invocation_progress` socket event a preview consumer reads, without the
    serialized invocation. A separate model rather than the event itself: exposing the event as a
    response model would give it a second OpenAPI schema variant and change the generated client
    types for every socket consumer.
    """

    queue_id: str = Field(description="The ID of the queue")
    item_id: int = Field(description="The ID of the queue item")
    session_id: str = Field(description="The ID of the session (aka graph execution state)")
    invocation_source_id: str = Field(description="The ID of the prepared invocation's source node")
    parent_item_id: int | None = Field(
        default=None, description="The parent queue item id when this item is a called-workflow child"
    )
    root_item_id: int | None = Field(
        default=None, description="The root queue item id for this called-workflow chain, if any"
    )
    workflow_call_parent_source_id: str | None = Field(
        default=None,
        description="The visible parent Call Saved Workflow source node for a called-workflow child event",
    )
    revision: int | None = Field(description="Monotonic per queue item and session; see InvocationProgressEvent")
    message: str = Field(description="A message to display")
    percentage: float | None = Field(description="The percentage of the progress, or null if indeterminate")
    image: ProgressImage | None = Field(description="The latest denoising preview")

    @classmethod
    def from_event(cls, event: InvocationProgressEvent) -> "ProgressPreviewDTO":
        return cls(
            queue_id=event.queue_id,
            item_id=event.item_id,
            session_id=event.session_id,
            invocation_source_id=event.invocation_source_id,
            parent_item_id=event.parent_item_id,
            root_item_id=event.root_item_id,
            workflow_call_parent_source_id=event.workflow_call_parent_source_id,
            revision=event.revision,
            message=event.message,
            percentage=event.percentage,
            image=event.image,
        )
