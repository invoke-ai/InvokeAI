# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)


from typing import TYPE_CHECKING, Optional

from invokeai.app.services.events.events_common import (
    BaseEvent,
    BatchEnqueuedEvent,
    BulkDownloadCompleteEvent,
    BulkDownloadErrorEvent,
    BulkDownloadStartedEvent,
    DownloadCancelledEvent,
    DownloadCompleteEvent,
    DownloadErrorEvent,
    DownloadProgressEvent,
    DownloadStartedEvent,
    InvocationCompleteEvent,
    InvocationDenoiseProgressEvent,
    InvocationErrorEvent,
    InvocationStartedEvent,
    ModelInstallCancelledEvent,
    ModelInstallCompleteEvent,
    ModelInstallDownloadProgressEvent,
    ModelInstallDownloadsCompleteEvent,
    ModelInstallErrorEvent,
    ModelInstallStartedEvent,
    ModelLoadCompleteEvent,
    ModelLoadStartedEvent,
    QueueClearedEvent,
    QueueItemStatusChangedEvent,
    SessionCanceledEvent,
    SessionCompleteEvent,
    SessionStartedEvent,
)

if TYPE_CHECKING:
    from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
    from invokeai.app.services.events.events_common import BaseEvent
    from invokeai.app.services.model_install.model_install_common import ModelInstallJob
    from invokeai.app.services.session_processor.session_processor_common import ProgressImage
    from invokeai.app.services.session_queue.session_queue_common import (
        BatchStatus,
        EnqueueBatchResult,
        SessionQueueItem,
        SessionQueueStatus,
    )
    from invokeai.backend.model_manager.config import AnyModelConfig, SubModelType


class EventServiceBase:
    """Basic event bus, to have an empty stand-in when not needed"""

    def dispatch(self, event: "BaseEvent") -> None:
        pass

    # region: Invocation

    def emit_invocation_started(self, queue_item: "SessionQueueItem", invocation: "BaseInvocation") -> None:
        self.dispatch(InvocationStartedEvent.build(queue_item, invocation))

    def emit_invocation_denoise_progress(
        self,
        queue_item: "SessionQueueItem",
        invocation: "BaseInvocation",
        step: int,
        total_steps: int,
        progress_image: "ProgressImage",
    ) -> None:
        self.dispatch(InvocationDenoiseProgressEvent.build(queue_item, invocation, step, total_steps, progress_image))

    def emit_invocation_complete(
        self, queue_item: "SessionQueueItem", invocation: "BaseInvocation", output: "BaseInvocationOutput"
    ) -> None:
        self.dispatch(InvocationCompleteEvent.build(queue_item, invocation, output))

    def emit_invocation_error(
        self,
        queue_item: "SessionQueueItem",
        invocation: "BaseInvocation",
        error_type: str,
        error_message: str,
        error_traceback: str,
    ) -> None:
        self.dispatch(InvocationErrorEvent.build(queue_item, invocation, error_type, error_message, error_traceback))

    # endregion

    # region Session

    def emit_session_started(self, queue_item: "SessionQueueItem") -> None:
        self.dispatch(SessionStartedEvent.build(queue_item))

    def emit_session_complete(self, queue_item: "SessionQueueItem") -> None:
        self.dispatch(SessionCompleteEvent.build(queue_item))

    def emit_session_canceled(self, queue_item: "SessionQueueItem") -> None:
        self.dispatch(SessionCanceledEvent.build(queue_item))

    # endregion

    # region Queue

    def emit_queue_item_status_changed(
        self, queue_item: "SessionQueueItem", batch_status: "BatchStatus", queue_status: "SessionQueueStatus"
    ) -> None:
        self.dispatch(QueueItemStatusChangedEvent.build(queue_item, batch_status, queue_status))

    def emit_batch_enqueued(self, enqueue_result: "EnqueueBatchResult") -> None:
        self.dispatch(BatchEnqueuedEvent.build(enqueue_result))

    def emit_queue_cleared(self, queue_id: str) -> None:
        self.dispatch(QueueClearedEvent.build(queue_id))

    # endregion

    # region Download

    def emit_download_started(self, source: str, download_path: str) -> None:
        self.dispatch(DownloadStartedEvent.build(source, download_path))

    def emit_download_progress(self, source: str, download_path: str, current_bytes: int, total_bytes: int) -> None:
        self.dispatch(DownloadProgressEvent.build(source, download_path, current_bytes, total_bytes))

    def emit_download_complete(self, source: str, download_path: str, total_bytes: int) -> None:
        self.dispatch(DownloadCompleteEvent.build(source, download_path, total_bytes))

    def emit_download_cancelled(self, source: str) -> None:
        self.dispatch(DownloadCancelledEvent.build(source))

    def emit_download_error(self, source: str, error_type: str, error: str) -> None:
        self.dispatch(DownloadErrorEvent.build(source, error_type, error))

    # endregion

    # region Model loading

    def emit_model_load_started(self, config: "AnyModelConfig", submodel_type: Optional["SubModelType"] = None) -> None:
        self.dispatch(ModelLoadStartedEvent.build(config, submodel_type))

    def emit_model_load_complete(
        self, config: "AnyModelConfig", submodel_type: Optional["SubModelType"] = None
    ) -> None:
        self.dispatch(ModelLoadCompleteEvent.build(config, submodel_type))

    # endregion

    # region Model install

    def emit_model_install_download_progress(self, job: "ModelInstallJob") -> None:
        self.dispatch(ModelInstallDownloadProgressEvent.build(job))

    def emit_model_install_downloads_complete(self, job: "ModelInstallJob") -> None:
        self.dispatch(ModelInstallDownloadsCompleteEvent.build(job))

    def emit_model_install_started(self, job: "ModelInstallJob") -> None:
        self.dispatch(ModelInstallStartedEvent.build(job))

    def emit_model_install_complete(self, job: "ModelInstallJob") -> None:
        self.dispatch(ModelInstallCompleteEvent.build(job))

    def emit_model_install_cancelled(self, job: "ModelInstallJob") -> None:
        self.dispatch(ModelInstallCancelledEvent.build(job))

    def emit_model_install_error(self, job: "ModelInstallJob") -> None:
        self.dispatch(ModelInstallErrorEvent.build(job))

    # endregion

    # region Bulk image download

    def emit_bulk_download_started(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str
    ) -> None:
        self.dispatch(BulkDownloadStartedEvent.build(bulk_download_id, bulk_download_item_id, bulk_download_item_name))

    def emit_bulk_download_complete(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str
    ) -> None:
        self.dispatch(BulkDownloadCompleteEvent.build(bulk_download_id, bulk_download_item_id, bulk_download_item_name))

    def emit_bulk_download_error(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str, error: str
    ) -> None:
        self.dispatch(
            BulkDownloadErrorEvent.build(bulk_download_id, bulk_download_item_id, bulk_download_item_name, error)
        )

    # endregion
