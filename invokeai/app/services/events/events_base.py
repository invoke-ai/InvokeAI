# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)


from typing import TYPE_CHECKING, Literal, Optional

from invokeai.app.services.events.events_common import (
    BatchEnqueuedEvent,
    BulkDownloadCompleteEvent,
    BulkDownloadErrorEvent,
    BulkDownloadStartedEvent,
    DownloadCancelledEvent,
    DownloadCompleteEvent,
    DownloadErrorEvent,
    DownloadPausedEvent,
    DownloadProgressEvent,
    DownloadStartedEvent,
    EventBase,
    ImageIndexStatusEvent,
    ImageIndexUpdatedEvent,
    ImageMapProjectionReadyEvent,
    ImageUploadedEvent,
    IntermediatesOperationChangedEvent,
    InvocationCompleteEvent,
    InvocationErrorEvent,
    InvocationProgressEvent,
    InvocationStartedEvent,
    LLMTaskCompleteEvent,
    LLMTaskErrorEvent,
    LLMTaskProgressEvent,
    ModelInstallCancelledEvent,
    ModelInstallCompleteEvent,
    ModelInstallDownloadProgressEvent,
    ModelInstallDownloadsCompleteEvent,
    ModelInstallDownloadStartedEvent,
    ModelInstallErrorEvent,
    ModelInstallStartedEvent,
    ModelLoadCompleteEvent,
    ModelLoadStartedEvent,
    QueueClearedEvent,
    QueueItemsCanceledEvent,
    QueueItemsRetriedEvent,
    QueueItemStatusChangedEvent,
    RecallParametersUpdatedEvent,
    UserAccessChangedEvent,
    VideoRecallAction,
    VideoRecallMode,
    VideoRecallRequestedEvent,
    VideoRecallVideo,
    VideoUploadedEvent,
    WorkflowCreatedEvent,
    WorkflowDeletedEvent,
    WorkflowUpdatedEvent,
)

if TYPE_CHECKING:
    from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
    from invokeai.app.services.board_records.board_records_common import BoardRecord
    from invokeai.app.services.download.download_base import DownloadJob
    from invokeai.app.services.images.images_common import ImageDTO
    from invokeai.app.services.intermediates.intermediates_common import IntermediatesOperation
    from invokeai.app.services.model_install.model_install_common import ModelInstallJob
    from invokeai.app.services.session_processor.session_processor_common import ProgressImage
    from invokeai.app.services.session_queue.session_queue_common import (
        BatchStatus,
        EnqueueBatchResult,
        RetryItemsResult,
        SessionQueueItem,
        SessionQueueStatus,
    )
    from invokeai.app.services.videos.videos_common import VideoDTO
    from invokeai.backend.model_manager.configs.factory import AnyModelConfig
    from invokeai.backend.model_manager.taxonomy import SubModelType


class EventServiceBase:
    """Basic event bus, to have an empty stand-in when not needed"""

    def dispatch(self, event: "EventBase") -> None:
        pass

    # region: Invocation

    def emit_invocation_started(self, queue_item: "SessionQueueItem", invocation: "BaseInvocation") -> None:
        """Emitted when an invocation is started"""
        self.dispatch(InvocationStartedEvent.build(queue_item, invocation.get_event_invocation()))

    def emit_invocation_progress(
        self,
        queue_item: "SessionQueueItem",
        invocation: "BaseInvocation",
        message: str,
        percentage: float | None = None,
        image: "ProgressImage | None" = None,
        revision: int | None = None,
    ) -> InvocationProgressEvent:
        """Emitted periodically during an invocation. Returns the event so the caller can retain it."""
        event = InvocationProgressEvent.build(
            queue_item, invocation.get_event_invocation(), message, percentage, image, revision
        )
        self.dispatch(event)
        return event

    def emit_invocation_complete(
        self, queue_item: "SessionQueueItem", invocation: "BaseInvocation", output: "BaseInvocationOutput"
    ) -> None:
        """Emitted when an invocation is complete"""
        self.dispatch(InvocationCompleteEvent.build(queue_item, invocation.get_event_invocation(), output))

    def emit_invocation_error(
        self,
        queue_item: "SessionQueueItem",
        invocation: "BaseInvocation",
        error_type: str,
        error_message: str,
        error_traceback: str,
    ) -> None:
        """Emitted when an invocation encounters an error"""
        self.dispatch(
            InvocationErrorEvent.build(
                queue_item,
                invocation.get_event_invocation(),
                error_type,
                error_message,
                error_traceback,
            )
        )

    # endregion

    # region Queue

    def emit_queue_item_status_changed(
        self, queue_item: "SessionQueueItem", batch_status: "BatchStatus", queue_status: "SessionQueueStatus"
    ) -> None:
        """Emitted when a queue item's status changes"""
        self.dispatch(QueueItemStatusChangedEvent.build(queue_item, batch_status, queue_status))

    def emit_batch_enqueued(self, enqueue_result: "EnqueueBatchResult", user_id: str = "system") -> None:
        """Emitted when a batch is enqueued"""
        self.dispatch(BatchEnqueuedEvent.build(enqueue_result, user_id))

    def emit_queue_items_retried(
        self, retry_result: "RetryItemsResult", user_ids: list[str], retried_item_ids_by_user: dict[str, list[int]]
    ) -> None:
        """Emitted when a list of queue items are retried"""
        self.dispatch(QueueItemsRetriedEvent.build(retry_result, user_ids, retried_item_ids_by_user))

    def emit_queue_items_canceled(self, queue_id: str, canceled_item_ids_by_user: dict[str, list[int]]) -> None:
        """Emitted when queue items are canceled or deleted in bulk without per-item status change events"""
        self.dispatch(QueueItemsCanceledEvent.build(queue_id, canceled_item_ids_by_user))

    def emit_queue_cleared(self, queue_id: str, user_id: str | None = None) -> None:
        """Emitted when a queue is cleared. `user_id` scopes the clear to one user's items; None means all."""
        self.dispatch(QueueClearedEvent.build(queue_id, user_id))

    def emit_recall_parameters_updated(self, queue_id: str, user_id: str, parameters: dict) -> None:
        """Emitted when recall parameters are updated"""
        self.dispatch(RecallParametersUpdatedEvent.build(queue_id, user_id, parameters))

    def emit_video_recall_requested(
        self,
        queue_id: str,
        user_id: str,
        action: VideoRecallAction,
        mode: VideoRecallMode | None = None,
        strict: bool = False,
        parameters: dict | None = None,
        video: VideoRecallVideo | None = None,
    ) -> None:
        """Emitted when an external caller asks for a user's Video panel to be updated"""
        self.dispatch(VideoRecallRequestedEvent.build(queue_id, user_id, action, mode, strict, parameters, video))

    # endregion

    # region Workflow library

    def emit_workflow_created(self, workflow_id: str, user_id: str, is_public: bool) -> None:
        """Emitted when a workflow is created."""
        self.dispatch(WorkflowCreatedEvent.build(workflow_id=workflow_id, user_id=user_id, is_public=is_public))

    def emit_workflow_updated(self, workflow_id: str, user_id: str, old_is_public: bool, new_is_public: bool) -> None:
        """Emitted when a workflow is updated."""
        self.dispatch(
            WorkflowUpdatedEvent.build(
                workflow_id=workflow_id,
                user_id=user_id,
                old_is_public=old_is_public,
                new_is_public=new_is_public,
            )
        )

    def emit_workflow_deleted(self, workflow_id: str, user_id: str, is_public: bool) -> None:
        """Emitted when a workflow is deleted."""
        self.dispatch(WorkflowDeletedEvent.build(workflow_id=workflow_id, user_id=user_id, is_public=is_public))

    # endregion

    # region User accounts

    def emit_user_access_changed(self, user_id: str, is_admin: bool, is_active: bool, token_epoch: int = 0) -> None:
        """Emitted when a user's role, active status, or token epoch changes.

        Server-internal; never sent to clients.
        """
        self.dispatch(
            UserAccessChangedEvent.build(
                user_id=user_id, is_admin=is_admin, is_active=is_active, token_epoch=token_epoch
            )
        )

    # endregion

    # region Download

    def emit_download_started(self, job: "DownloadJob") -> None:
        """Emitted when a download is started"""
        self.dispatch(DownloadStartedEvent.build(job))

    def emit_download_progress(self, job: "DownloadJob") -> None:
        """Emitted at intervals during a download"""
        self.dispatch(DownloadProgressEvent.build(job))

    def emit_download_complete(self, job: "DownloadJob") -> None:
        """Emitted when a download is completed"""
        self.dispatch(DownloadCompleteEvent.build(job))

    def emit_download_cancelled(self, job: "DownloadJob") -> None:
        """Emitted when a download is cancelled"""
        self.dispatch(DownloadCancelledEvent.build(job))

    def emit_download_paused(self, job: "DownloadJob") -> None:
        """Emitted when a download is paused"""
        self.dispatch(DownloadPausedEvent.build(job))

    def emit_download_error(self, job: "DownloadJob") -> None:
        """Emitted when a download encounters an error"""
        self.dispatch(DownloadErrorEvent.build(job))

    # endregion

    # region Model loading

    def emit_model_load_started(
        self, config: "AnyModelConfig", submodel_type: Optional["SubModelType"] = None, user_id: str = "system"
    ) -> None:
        """Emitted when a model load is started."""
        self.dispatch(ModelLoadStartedEvent.build(config, submodel_type, user_id))

    def emit_model_load_complete(
        self, config: "AnyModelConfig", submodel_type: Optional["SubModelType"] = None, user_id: str = "system"
    ) -> None:
        """Emitted when a model load is complete."""
        self.dispatch(ModelLoadCompleteEvent.build(config, submodel_type, user_id))

    # endregion

    # region Model install

    def emit_model_install_download_started(self, job: "ModelInstallJob") -> None:
        """Emitted at intervals while the install job is started (remote models only)."""
        self.dispatch(ModelInstallDownloadStartedEvent.build(job))

    def emit_model_install_download_progress(self, job: "ModelInstallJob") -> None:
        """Emitted at intervals while the install job is in progress (remote models only)."""
        self.dispatch(ModelInstallDownloadProgressEvent.build(job))

    def emit_model_install_downloads_complete(self, job: "ModelInstallJob") -> None:
        self.dispatch(ModelInstallDownloadsCompleteEvent.build(job))

    def emit_model_install_started(self, job: "ModelInstallJob") -> None:
        """Emitted once when an install job is started (after any download)."""
        self.dispatch(ModelInstallStartedEvent.build(job))

    def emit_model_install_complete(self, job: "ModelInstallJob") -> None:
        """Emitted when an install job is completed successfully."""
        self.dispatch(ModelInstallCompleteEvent.build(job))

    def emit_model_install_cancelled(self, job: "ModelInstallJob") -> None:
        """Emitted when an install job is cancelled."""
        self.dispatch(ModelInstallCancelledEvent.build(job))

    def emit_model_install_error(self, job: "ModelInstallJob") -> None:
        """Emitted when an install job encounters an exception."""
        self.dispatch(ModelInstallErrorEvent.build(job))

    # endregion

    # region LLM utility tasks

    def emit_llm_task_progress(
        self,
        task_id: str,
        user_id: str,
        phase: Literal["loading_model", "generating"],
        message: str,
        percentage: float | None = None,
        current_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Emit a progress event for an LLM utility task (expand-prompt, image-to-prompt)."""
        self.dispatch(
            LLMTaskProgressEvent(
                task_id=task_id,
                user_id=user_id,
                phase=phase,
                message=message,
                percentage=percentage,
                current_tokens=current_tokens,
                total_tokens=total_tokens,
            )
        )

    def emit_llm_task_complete(self, task_id: str, user_id: str) -> None:
        """Emit a completion event for an LLM utility task."""
        self.dispatch(LLMTaskCompleteEvent(task_id=task_id, user_id=user_id))

    def emit_llm_task_error(self, task_id: str, user_id: str, error: str) -> None:
        """Emit an error event for an LLM utility task."""
        self.dispatch(LLMTaskErrorEvent(task_id=task_id, user_id=user_id, error=error))

    # endregion

    # region Bulk image download

    def emit_bulk_download_started(
        self,
        bulk_download_id: str,
        bulk_download_item_id: str,
        bulk_download_item_name: str,
        user_id: str = "system",
    ) -> None:
        """Emitted when a bulk image download is started"""
        self.dispatch(
            BulkDownloadStartedEvent.build(bulk_download_id, bulk_download_item_id, bulk_download_item_name, user_id)
        )

    def emit_bulk_download_complete(
        self,
        bulk_download_id: str,
        bulk_download_item_id: str,
        bulk_download_item_name: str,
        user_id: str = "system",
    ) -> None:
        """Emitted when a bulk image download is complete"""
        self.dispatch(
            BulkDownloadCompleteEvent.build(bulk_download_id, bulk_download_item_id, bulk_download_item_name, user_id)
        )

    def emit_bulk_download_error(
        self,
        bulk_download_id: str,
        bulk_download_item_id: str,
        bulk_download_item_name: str,
        error: str,
        user_id: str = "system",
    ) -> None:
        """Emitted when a bulk image download has an error"""
        self.dispatch(
            BulkDownloadErrorEvent.build(
                bulk_download_id, bulk_download_item_id, bulk_download_item_name, error, user_id
            )
        )

    # endregion

    # region Image index

    def emit_image_index_status(self, total: int, embedded: int, pending: int, failed: int = 0) -> None:
        """Emitted when the image embedding index makes progress"""
        self.dispatch(ImageIndexStatusEvent.build(total=total, embedded=embedded, pending=pending, failed=failed))

    def emit_image_index_updated(self, user_id: str) -> None:
        """Emitted to one user when their images were just (re)embedded"""
        self.dispatch(ImageIndexUpdatedEvent.build(user_id=user_id))

    def emit_image_map_projection_ready(self, user_id: str, point_count: int) -> None:
        """Emitted when a user's image map projection has been recomputed"""
        self.dispatch(ImageMapProjectionReadyEvent.build(user_id=user_id, point_count=point_count))

    # endregion

    # region Intermediates

    def emit_intermediates_operation_changed(self, operation: "IntermediatesOperation") -> None:
        """Emitted whenever an intermediates cleanup operation is created, progresses or finishes"""
        self.dispatch(IntermediatesOperationChangedEvent.build(operation))

    # endregion

    # region Media uploads

    def emit_image_uploaded(
        self, image_dto: "ImageDTO", user_id: str, board: Optional["BoardRecord"], shared_user_ids: list[str]
    ) -> None:
        """Emitted when a gallery-visible image arrives through the upload API; `board` is the one it was sent to"""
        # The board attach can fail quietly inside create(); the DTO says where the image actually is.
        board = board if image_dto.board_id is not None else None
        self.dispatch(
            ImageUploadedEvent.build(
                image_name=image_dto.image_name,
                image_category=image_dto.image_category,
                user_id=user_id,
                board_id=image_dto.board_id,
                board_owner_id=board.user_id if board else None,
                board_visibility=board.board_visibility if board else None,
                shared_user_ids=shared_user_ids if board else [],
            )
        )

    def emit_video_uploaded(
        self, video_dto: "VideoDTO", user_id: str, board: Optional["BoardRecord"], shared_user_ids: list[str]
    ) -> None:
        """Emitted when a gallery-visible video arrives through the upload API; `board` is the one it was sent to"""
        board = board if video_dto.board_id is not None else None
        self.dispatch(
            VideoUploadedEvent.build(
                video_name=video_dto.video_name,
                video_category=video_dto.video_category,
                user_id=user_id,
                board_id=video_dto.board_id,
                board_owner_id=board.user_id if board else None,
                board_visibility=board.board_visibility if board else None,
                shared_user_ids=shared_user_ids if board else [],
            )
        )

    # endregion
