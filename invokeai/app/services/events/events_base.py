# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)


from typing import TYPE_CHECKING, Optional

from invokeai.app.services.events.events_common import (
    BatchEnqueuedEvent,
    BulkDownloadCompleteEvent,
    BulkDownloadErrorEvent,
    BulkDownloadStartedEvent,
    DownloadCancelledEvent,
    DownloadCompleteEvent,
    DownloadErrorEvent,
    DownloadProgressEvent,
    DownloadStartedEvent,
    EventBase,
    ExtraData,
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
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState

if TYPE_CHECKING:
    from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
    from invokeai.app.services.download.download_base import DownloadJob
    from invokeai.app.services.events.events_common import EventBase
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

    def dispatch(self, event: "EventBase") -> None:
        pass

    # region: Invocation

    def emit_invocation_started(
        self, queue_item: "SessionQueueItem", invocation: "BaseInvocation", extra: Optional[ExtraData] = None
    ) -> None:
        """Emitted when an invocation is started"""
        self.dispatch(InvocationStartedEvent.build(queue_item, invocation, extra))

    def emit_invocation_denoise_progress(
        self,
        queue_item: "SessionQueueItem",
        invocation: "BaseInvocation",
        intermediate_state: PipelineIntermediateState,
        progress_image: "ProgressImage",
        extra: Optional[ExtraData] = None,
    ) -> None:
        """Emitted at each step during denoising of an invocation."""
        self.dispatch(
            InvocationDenoiseProgressEvent.build(queue_item, invocation, intermediate_state, progress_image, extra)
        )

    def emit_invocation_complete(
        self,
        queue_item: "SessionQueueItem",
        invocation: "BaseInvocation",
        output: "BaseInvocationOutput",
        extra: Optional[ExtraData] = None,
    ) -> None:
        """Emitted when an invocation is complete"""
        self.dispatch(InvocationCompleteEvent.build(queue_item, invocation, output, extra))

    def emit_invocation_error(
        self,
        queue_item: "SessionQueueItem",
        invocation: "BaseInvocation",
        error_type: str,
        error: str,
        extra: Optional[ExtraData] = None,
    ) -> None:
        """Emitted when an invocation encounters an error"""
        self.dispatch(InvocationErrorEvent.build(queue_item, invocation, error_type, error, extra))

    # endregion

    # region Session

    def emit_session_started(self, queue_item: "SessionQueueItem", extra: Optional[ExtraData] = None) -> None:
        """Emitted when a session has started"""
        self.dispatch(SessionStartedEvent.build(queue_item, extra))

    def emit_session_complete(self, queue_item: "SessionQueueItem", extra: Optional[ExtraData] = None) -> None:
        """Emitted when a session has completed all invocations"""
        self.dispatch(SessionCompleteEvent.build(queue_item, extra))

    def emit_session_canceled(self, queue_item: "SessionQueueItem", extra: Optional[ExtraData] = None) -> None:
        """Emitted when a session is canceled"""
        self.dispatch(SessionCanceledEvent.build(queue_item, extra))

    # endregion

    # region Queue

    def emit_queue_item_status_changed(
        self,
        queue_item: "SessionQueueItem",
        batch_status: "BatchStatus",
        queue_status: "SessionQueueStatus",
        extra: Optional[ExtraData] = None,
    ) -> None:
        """Emitted when a queue item's status changes"""
        self.dispatch(QueueItemStatusChangedEvent.build(queue_item, batch_status, queue_status, extra))

    def emit_batch_enqueued(self, enqueue_result: "EnqueueBatchResult", extra: Optional[ExtraData] = None) -> None:
        """Emitted when a batch is enqueued"""
        self.dispatch(BatchEnqueuedEvent.build(enqueue_result, extra))

    def emit_queue_cleared(self, queue_id: str, extra: Optional[ExtraData] = None) -> None:
        """Emitted when a queue is cleared"""
        self.dispatch(QueueClearedEvent.build(queue_id, extra))

    # endregion

    # region Download

    def emit_download_started(self, job: "DownloadJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted when a download is started"""
        self.dispatch(DownloadStartedEvent.build(job, extra))

    def emit_download_progress(self, job: "DownloadJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted at intervals during a download"""
        self.dispatch(DownloadProgressEvent.build(job, extra))

    def emit_download_complete(self, job: "DownloadJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted when a download is completed"""
        self.dispatch(DownloadCompleteEvent.build(job, extra))

    def emit_download_cancelled(self, job: "DownloadJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted when a download is cancelled"""
        self.dispatch(DownloadCancelledEvent.build(job, extra))

    def emit_download_error(self, job: "DownloadJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted when a download encounters an error"""
        self.dispatch(DownloadErrorEvent.build(job, extra))

    # endregion

    # region Model loading

    def emit_model_load_started(
        self,
        config: "AnyModelConfig",
        submodel_type: Optional["SubModelType"] = None,
        extra: Optional[ExtraData] = None,
    ) -> None:
        """Emitted when a model load is started."""
        self.dispatch(ModelLoadStartedEvent.build(config, submodel_type, extra))

    def emit_model_load_complete(
        self,
        config: "AnyModelConfig",
        submodel_type: Optional["SubModelType"] = None,
        extra: Optional[ExtraData] = None,
    ) -> None:
        """Emitted when a model load is complete."""
        self.dispatch(ModelLoadCompleteEvent.build(config, submodel_type, extra))

    # endregion

    # region Model install

    def emit_model_install_download_progress(self, job: "ModelInstallJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted at intervals while the install job is in progress (remote models only)."""
        self.dispatch(ModelInstallDownloadProgressEvent.build(job, extra))

    def emit_model_install_downloads_complete(self, job: "ModelInstallJob", extra: Optional[ExtraData] = None) -> None:
        self.dispatch(ModelInstallDownloadsCompleteEvent.build(job, extra))

    def emit_model_install_started(self, job: "ModelInstallJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted once when an install job is started (after any download)."""
        self.dispatch(ModelInstallStartedEvent.build(job, extra))

    def emit_model_install_complete(self, job: "ModelInstallJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted when an install job is completed successfully."""
        self.dispatch(ModelInstallCompleteEvent.build(job, extra))

    def emit_model_install_cancelled(self, job: "ModelInstallJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted when an install job is cancelled."""
        self.dispatch(ModelInstallCancelledEvent.build(job, extra))

    def emit_model_install_error(self, job: "ModelInstallJob", extra: Optional[ExtraData] = None) -> None:
        """Emitted when an install job encounters an exception."""
        self.dispatch(ModelInstallErrorEvent.build(job, extra))

    # endregion

    # region Bulk image download

    def emit_bulk_download_started(
        self,
        bulk_download_id: str,
        bulk_download_item_id: str,
        bulk_download_item_name: str,
        extra: Optional[ExtraData] = None,
    ) -> None:
        """Emitted when a bulk image download is started"""
        self.dispatch(
            BulkDownloadStartedEvent.build(bulk_download_id, bulk_download_item_id, bulk_download_item_name, extra)
        )

    def emit_bulk_download_complete(
        self,
        bulk_download_id: str,
        bulk_download_item_id: str,
        bulk_download_item_name: str,
        extra: Optional[ExtraData] = None,
    ) -> None:
        """Emitted when a bulk image download is complete"""
        self.dispatch(
            BulkDownloadCompleteEvent.build(bulk_download_id, bulk_download_item_id, bulk_download_item_name, extra)
        )

    def emit_bulk_download_error(
        self,
        bulk_download_id: str,
        bulk_download_item_id: str,
        bulk_download_item_name: str,
        error: str,
        extra: Optional[ExtraData] = None,
    ) -> None:
        """Emitted when a bulk image download has an error"""
        self.dispatch(
            BulkDownloadErrorEvent.build(bulk_download_id, bulk_download_item_id, bulk_download_item_name, error, extra)
        )

    # endregion
