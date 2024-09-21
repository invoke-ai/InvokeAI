from typing import TYPE_CHECKING, Any, ClassVar, Coroutine, Generic, Optional, Protocol, TypeAlias, TypeVar

from fastapi_events.handlers.local import local_handler
from fastapi_events.registry.payload_schema import registry as payload_schema
from pydantic import BaseModel, ConfigDict, Field

from invokeai.app.services.session_processor.session_processor_common import ProgressImage
from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    BatchStatus,
    EnqueueBatchResult,
    SessionQueueItem,
    SessionQueueStatus,
)
from invokeai.app.services.shared.graph import AnyInvocation, AnyInvocationOutput
from invokeai.app.util.misc import get_timestamp
from invokeai.backend.model_manager.config import AnyModelConfig, SubModelType

if TYPE_CHECKING:
    from invokeai.app.services.download.download_base import DownloadJob
    from invokeai.app.services.model_install.model_install_common import ModelInstallJob


class EventBase(BaseModel):
    """Base class for all events. All events must inherit from this class.

    Events must define a class attribute `__event_name__` to identify the event.

    All other attributes should be defined as normal for a pydantic model.

    A timestamp is automatically added to the event when it is created.
    """

    __event_name__: ClassVar[str]
    timestamp: int = Field(description="The timestamp of the event", default_factory=get_timestamp)

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    @classmethod
    def get_events(cls) -> set[type["EventBase"]]:
        """Get a set of all event models."""

        event_subclasses: set[type["EventBase"]] = set()
        for subclass in cls.__subclasses__():
            # We only want to include subclasses that are event models, not intermediary classes
            if hasattr(subclass, "__event_name__"):
                event_subclasses.add(subclass)
            event_subclasses.update(subclass.get_events())

        return event_subclasses


TEvent = TypeVar("TEvent", bound=EventBase, contravariant=True)

FastAPIEvent: TypeAlias = tuple[str, TEvent]
"""
A tuple representing a `fastapi-events` event, with the event name and payload.
Provide a generic type to `TEvent` to specify the payload type.
"""


class FastAPIEventFunc(Protocol, Generic[TEvent]):
    def __call__(self, event: FastAPIEvent[TEvent]) -> Optional[Coroutine[Any, Any, None]]: ...


def register_events(events: set[type[TEvent]] | type[TEvent], func: FastAPIEventFunc[TEvent]) -> None:
    """Register a function to handle specific events.

    :param events: An event or set of events to handle
    :param func: The function to handle the events
    """
    events = events if isinstance(events, set) else {events}
    for event in events:
        assert hasattr(event, "__event_name__")
        local_handler.register(event_name=event.__event_name__, _func=func)  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]


class QueueEventBase(EventBase):
    """Base class for queue events"""

    queue_id: str = Field(description="The ID of the queue")


class QueueItemEventBase(QueueEventBase):
    """Base class for queue item events"""

    item_id: int = Field(description="The ID of the queue item")
    batch_id: str = Field(description="The ID of the queue batch")
    origin: str | None = Field(default=None, description="The origin of the queue item")
    destination: str | None = Field(default=None, description="The destination of the queue item")


class InvocationEventBase(QueueItemEventBase):
    """Base class for invocation events"""

    session_id: str = Field(description="The ID of the session (aka graph execution state)")
    queue_id: str = Field(description="The ID of the queue")
    session_id: str = Field(description="The ID of the session (aka graph execution state)")
    invocation: AnyInvocation = Field(description="The ID of the invocation")
    invocation_source_id: str = Field(description="The ID of the prepared invocation's source node")


@payload_schema.register
class InvocationStartedEvent(InvocationEventBase):
    """Event model for invocation_started"""

    __event_name__ = "invocation_started"

    @classmethod
    def build(cls, queue_item: SessionQueueItem, invocation: AnyInvocation) -> "InvocationStartedEvent":
        return cls(
            queue_id=queue_item.queue_id,
            item_id=queue_item.item_id,
            batch_id=queue_item.batch_id,
            origin=queue_item.origin,
            destination=queue_item.destination,
            session_id=queue_item.session_id,
            invocation=invocation,
            invocation_source_id=queue_item.session.prepared_source_mapping[invocation.id],
        )


@payload_schema.register
class InvocationProgressEvent(InvocationEventBase):
    """Event model for invocation_progress"""

    __event_name__ = "invocation_progress"

    message: str = Field(description="A message to display")
    percentage: float | None = Field(
        default=None, ge=0, le=1, description="The percentage of the progress (omit to indicate indeterminate progress)"
    )
    image: ProgressImage | None = Field(
        default=None, description="An image representing the current state of the progress"
    )

    @classmethod
    def build(
        cls,
        queue_item: SessionQueueItem,
        invocation: AnyInvocation,
        message: str,
        percentage: float | None = None,
        image: ProgressImage | None = None,
    ) -> "InvocationProgressEvent":
        return cls(
            queue_id=queue_item.queue_id,
            item_id=queue_item.item_id,
            batch_id=queue_item.batch_id,
            origin=queue_item.origin,
            destination=queue_item.destination,
            session_id=queue_item.session_id,
            invocation=invocation,
            invocation_source_id=queue_item.session.prepared_source_mapping[invocation.id],
            percentage=percentage,
            image=image,
            message=message,
        )


@payload_schema.register
class InvocationCompleteEvent(InvocationEventBase):
    """Event model for invocation_complete"""

    __event_name__ = "invocation_complete"

    result: AnyInvocationOutput = Field(description="The result of the invocation")

    @classmethod
    def build(
        cls, queue_item: SessionQueueItem, invocation: AnyInvocation, result: AnyInvocationOutput
    ) -> "InvocationCompleteEvent":
        return cls(
            queue_id=queue_item.queue_id,
            item_id=queue_item.item_id,
            batch_id=queue_item.batch_id,
            origin=queue_item.origin,
            destination=queue_item.destination,
            session_id=queue_item.session_id,
            invocation=invocation,
            invocation_source_id=queue_item.session.prepared_source_mapping[invocation.id],
            result=result,
        )


@payload_schema.register
class InvocationErrorEvent(InvocationEventBase):
    """Event model for invocation_error"""

    __event_name__ = "invocation_error"

    error_type: str = Field(description="The error type")
    error_message: str = Field(description="The error message")
    error_traceback: str = Field(description="The error traceback")
    user_id: Optional[str] = Field(default=None, description="The ID of the user who created the invocation")
    project_id: Optional[str] = Field(default=None, description="The ID of the user who created the invocation")

    @classmethod
    def build(
        cls,
        queue_item: SessionQueueItem,
        invocation: AnyInvocation,
        error_type: str,
        error_message: str,
        error_traceback: str,
    ) -> "InvocationErrorEvent":
        return cls(
            queue_id=queue_item.queue_id,
            item_id=queue_item.item_id,
            batch_id=queue_item.batch_id,
            origin=queue_item.origin,
            destination=queue_item.destination,
            session_id=queue_item.session_id,
            invocation=invocation,
            invocation_source_id=queue_item.session.prepared_source_mapping[invocation.id],
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
            user_id=getattr(queue_item, "user_id", None),
            project_id=getattr(queue_item, "project_id", None),
        )


@payload_schema.register
class QueueItemStatusChangedEvent(QueueItemEventBase):
    """Event model for queue_item_status_changed"""

    __event_name__ = "queue_item_status_changed"

    status: QUEUE_ITEM_STATUS = Field(description="The new status of the queue item")
    error_type: Optional[str] = Field(default=None, description="The error type, if any")
    error_message: Optional[str] = Field(default=None, description="The error message, if any")
    error_traceback: Optional[str] = Field(default=None, description="The error traceback, if any")
    created_at: Optional[str] = Field(default=None, description="The timestamp when the queue item was created")
    updated_at: Optional[str] = Field(default=None, description="The timestamp when the queue item was last updated")
    started_at: Optional[str] = Field(default=None, description="The timestamp when the queue item was started")
    completed_at: Optional[str] = Field(default=None, description="The timestamp when the queue item was completed")
    batch_status: BatchStatus = Field(description="The status of the batch")
    queue_status: SessionQueueStatus = Field(description="The status of the queue")
    session_id: str = Field(description="The ID of the session (aka graph execution state)")

    @classmethod
    def build(
        cls, queue_item: SessionQueueItem, batch_status: BatchStatus, queue_status: SessionQueueStatus
    ) -> "QueueItemStatusChangedEvent":
        return cls(
            queue_id=queue_item.queue_id,
            item_id=queue_item.item_id,
            batch_id=queue_item.batch_id,
            origin=queue_item.origin,
            destination=queue_item.destination,
            session_id=queue_item.session_id,
            status=queue_item.status,
            error_type=queue_item.error_type,
            error_message=queue_item.error_message,
            error_traceback=queue_item.error_traceback,
            created_at=str(queue_item.created_at) if queue_item.created_at else None,
            updated_at=str(queue_item.updated_at) if queue_item.updated_at else None,
            started_at=str(queue_item.started_at) if queue_item.started_at else None,
            completed_at=str(queue_item.completed_at) if queue_item.completed_at else None,
            batch_status=batch_status,
            queue_status=queue_status,
        )


@payload_schema.register
class BatchEnqueuedEvent(QueueEventBase):
    """Event model for batch_enqueued"""

    __event_name__ = "batch_enqueued"

    batch_id: str = Field(description="The ID of the batch")
    enqueued: int = Field(description="The number of invocations enqueued")
    requested: int = Field(
        description="The number of invocations initially requested to be enqueued (may be less than enqueued if queue was full)"
    )
    priority: int = Field(description="The priority of the batch")
    origin: str | None = Field(default=None, description="The origin of the batch")

    @classmethod
    def build(cls, enqueue_result: EnqueueBatchResult) -> "BatchEnqueuedEvent":
        return cls(
            queue_id=enqueue_result.queue_id,
            batch_id=enqueue_result.batch.batch_id,
            origin=enqueue_result.batch.origin,
            enqueued=enqueue_result.enqueued,
            requested=enqueue_result.requested,
            priority=enqueue_result.priority,
        )


@payload_schema.register
class QueueClearedEvent(QueueEventBase):
    """Event model for queue_cleared"""

    __event_name__ = "queue_cleared"

    @classmethod
    def build(cls, queue_id: str) -> "QueueClearedEvent":
        return cls(queue_id=queue_id)


class DownloadEventBase(EventBase):
    """Base class for events associated with a download"""

    source: str = Field(description="The source of the download")


@payload_schema.register
class DownloadStartedEvent(DownloadEventBase):
    """Event model for download_started"""

    __event_name__ = "download_started"

    download_path: str = Field(description="The local path where the download is saved")

    @classmethod
    def build(cls, job: "DownloadJob") -> "DownloadStartedEvent":
        assert job.download_path
        return cls(source=str(job.source), download_path=job.download_path.as_posix())


@payload_schema.register
class DownloadProgressEvent(DownloadEventBase):
    """Event model for download_progress"""

    __event_name__ = "download_progress"

    download_path: str = Field(description="The local path where the download is saved")
    current_bytes: int = Field(description="The number of bytes downloaded so far")
    total_bytes: int = Field(description="The total number of bytes to be downloaded")

    @classmethod
    def build(cls, job: "DownloadJob") -> "DownloadProgressEvent":
        assert job.download_path
        return cls(
            source=str(job.source),
            download_path=job.download_path.as_posix(),
            current_bytes=job.bytes,
            total_bytes=job.total_bytes,
        )


@payload_schema.register
class DownloadCompleteEvent(DownloadEventBase):
    """Event model for download_complete"""

    __event_name__ = "download_complete"

    download_path: str = Field(description="The local path where the download is saved")
    total_bytes: int = Field(description="The total number of bytes downloaded")

    @classmethod
    def build(cls, job: "DownloadJob") -> "DownloadCompleteEvent":
        assert job.download_path
        return cls(source=str(job.source), download_path=job.download_path.as_posix(), total_bytes=job.total_bytes)


@payload_schema.register
class DownloadCancelledEvent(DownloadEventBase):
    """Event model for download_cancelled"""

    __event_name__ = "download_cancelled"

    @classmethod
    def build(cls, job: "DownloadJob") -> "DownloadCancelledEvent":
        return cls(source=str(job.source))


@payload_schema.register
class DownloadErrorEvent(DownloadEventBase):
    """Event model for download_error"""

    __event_name__ = "download_error"

    error_type: str = Field(description="The type of error")
    error: str = Field(description="The error message")

    @classmethod
    def build(cls, job: "DownloadJob") -> "DownloadErrorEvent":
        assert job.error_type
        assert job.error
        return cls(source=str(job.source), error_type=job.error_type, error=job.error)


class ModelEventBase(EventBase):
    """Base class for events associated with a model"""


@payload_schema.register
class ModelLoadStartedEvent(ModelEventBase):
    """Event model for model_load_started"""

    __event_name__ = "model_load_started"

    config: AnyModelConfig = Field(description="The model's config")
    submodel_type: Optional[SubModelType] = Field(default=None, description="The submodel type, if any")

    @classmethod
    def build(cls, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> "ModelLoadStartedEvent":
        return cls(config=config, submodel_type=submodel_type)


@payload_schema.register
class ModelLoadCompleteEvent(ModelEventBase):
    """Event model for model_load_complete"""

    __event_name__ = "model_load_complete"

    config: AnyModelConfig = Field(description="The model's config")
    submodel_type: Optional[SubModelType] = Field(default=None, description="The submodel type, if any")

    @classmethod
    def build(cls, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> "ModelLoadCompleteEvent":
        return cls(config=config, submodel_type=submodel_type)


@payload_schema.register
class ModelInstallDownloadStartedEvent(ModelEventBase):
    """Event model for model_install_download_started"""

    __event_name__ = "model_install_download_started"

    id: int = Field(description="The ID of the install job")
    source: str = Field(description="Source of the model; local path, repo_id or url")
    local_path: str = Field(description="Where model is downloading to")
    bytes: int = Field(description="Number of bytes downloaded so far")
    total_bytes: int = Field(description="Total size of download, including all files")
    parts: list[dict[str, int | str]] = Field(
        description="Progress of downloading URLs that comprise the model, if any"
    )

    @classmethod
    def build(cls, job: "ModelInstallJob") -> "ModelInstallDownloadStartedEvent":
        parts: list[dict[str, str | int]] = [
            {
                "url": str(x.source),
                "local_path": str(x.download_path),
                "bytes": x.bytes,
                "total_bytes": x.total_bytes,
            }
            for x in job.download_parts
        ]
        return cls(
            id=job.id,
            source=str(job.source),
            local_path=job.local_path.as_posix(),
            parts=parts,
            bytes=job.bytes,
            total_bytes=job.total_bytes,
        )


@payload_schema.register
class ModelInstallDownloadProgressEvent(ModelEventBase):
    """Event model for model_install_download_progress"""

    __event_name__ = "model_install_download_progress"

    id: int = Field(description="The ID of the install job")
    source: str = Field(description="Source of the model; local path, repo_id or url")
    local_path: str = Field(description="Where model is downloading to")
    bytes: int = Field(description="Number of bytes downloaded so far")
    total_bytes: int = Field(description="Total size of download, including all files")
    parts: list[dict[str, int | str]] = Field(
        description="Progress of downloading URLs that comprise the model, if any"
    )

    @classmethod
    def build(cls, job: "ModelInstallJob") -> "ModelInstallDownloadProgressEvent":
        parts: list[dict[str, str | int]] = [
            {
                "url": str(x.source),
                "local_path": str(x.download_path),
                "bytes": x.bytes,
                "total_bytes": x.total_bytes,
            }
            for x in job.download_parts
        ]
        return cls(
            id=job.id,
            source=str(job.source),
            local_path=job.local_path.as_posix(),
            parts=parts,
            bytes=job.bytes,
            total_bytes=job.total_bytes,
        )


@payload_schema.register
class ModelInstallDownloadsCompleteEvent(ModelEventBase):
    """Emitted once when an install job becomes active."""

    __event_name__ = "model_install_downloads_complete"

    id: int = Field(description="The ID of the install job")
    source: str = Field(description="Source of the model; local path, repo_id or url")

    @classmethod
    def build(cls, job: "ModelInstallJob") -> "ModelInstallDownloadsCompleteEvent":
        return cls(id=job.id, source=str(job.source))


@payload_schema.register
class ModelInstallStartedEvent(ModelEventBase):
    """Event model for model_install_started"""

    __event_name__ = "model_install_started"

    id: int = Field(description="The ID of the install job")
    source: str = Field(description="Source of the model; local path, repo_id or url")

    @classmethod
    def build(cls, job: "ModelInstallJob") -> "ModelInstallStartedEvent":
        return cls(id=job.id, source=str(job.source))


@payload_schema.register
class ModelInstallCompleteEvent(ModelEventBase):
    """Event model for model_install_complete"""

    __event_name__ = "model_install_complete"

    id: int = Field(description="The ID of the install job")
    source: str = Field(description="Source of the model; local path, repo_id or url")
    key: str = Field(description="Model config record key")
    total_bytes: Optional[int] = Field(description="Size of the model (may be None for installation of a local path)")

    @classmethod
    def build(cls, job: "ModelInstallJob") -> "ModelInstallCompleteEvent":
        assert job.config_out is not None
        return cls(id=job.id, source=str(job.source), key=(job.config_out.key), total_bytes=job.total_bytes)


@payload_schema.register
class ModelInstallCancelledEvent(ModelEventBase):
    """Event model for model_install_cancelled"""

    __event_name__ = "model_install_cancelled"

    id: int = Field(description="The ID of the install job")
    source: str = Field(description="Source of the model; local path, repo_id or url")

    @classmethod
    def build(cls, job: "ModelInstallJob") -> "ModelInstallCancelledEvent":
        return cls(id=job.id, source=str(job.source))


@payload_schema.register
class ModelInstallErrorEvent(ModelEventBase):
    """Event model for model_install_error"""

    __event_name__ = "model_install_error"

    id: int = Field(description="The ID of the install job")
    source: str = Field(description="Source of the model; local path, repo_id or url")
    error_type: str = Field(description="The name of the exception")
    error: str = Field(description="A text description of the exception")

    @classmethod
    def build(cls, job: "ModelInstallJob") -> "ModelInstallErrorEvent":
        assert job.error_type is not None
        assert job.error is not None
        return cls(id=job.id, source=str(job.source), error_type=job.error_type, error=job.error)


class BulkDownloadEventBase(EventBase):
    """Base class for events associated with a bulk image download"""

    bulk_download_id: str = Field(description="The ID of the bulk image download")
    bulk_download_item_id: str = Field(description="The ID of the bulk image download item")
    bulk_download_item_name: str = Field(description="The name of the bulk image download item")


@payload_schema.register
class BulkDownloadStartedEvent(BulkDownloadEventBase):
    """Event model for bulk_download_started"""

    __event_name__ = "bulk_download_started"

    @classmethod
    def build(
        cls, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str
    ) -> "BulkDownloadStartedEvent":
        return cls(
            bulk_download_id=bulk_download_id,
            bulk_download_item_id=bulk_download_item_id,
            bulk_download_item_name=bulk_download_item_name,
        )


@payload_schema.register
class BulkDownloadCompleteEvent(BulkDownloadEventBase):
    """Event model for bulk_download_complete"""

    __event_name__ = "bulk_download_complete"

    @classmethod
    def build(
        cls, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str
    ) -> "BulkDownloadCompleteEvent":
        return cls(
            bulk_download_id=bulk_download_id,
            bulk_download_item_id=bulk_download_item_id,
            bulk_download_item_name=bulk_download_item_name,
        )


@payload_schema.register
class BulkDownloadErrorEvent(BulkDownloadEventBase):
    """Event model for bulk_download_error"""

    __event_name__ = "bulk_download_error"

    error: str = Field(description="The error message")

    @classmethod
    def build(
        cls, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str, error: str
    ) -> "BulkDownloadErrorEvent":
        return cls(
            bulk_download_id=bulk_download_id,
            bulk_download_item_id=bulk_download_item_id,
            bulk_download_item_name=bulk_download_item_name,
            error=error,
        )
