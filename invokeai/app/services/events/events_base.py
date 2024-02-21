# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)


from typing import Any, Dict, List, Optional, Union

from invokeai.app.services.session_processor.session_processor_common import ProgressImage
from invokeai.app.services.session_queue.session_queue_common import (
    BatchStatus,
    EnqueueBatchResult,
    SessionQueueItem,
    SessionQueueStatus,
)
from invokeai.app.util.misc import get_timestamp
from invokeai.backend.model_manager import AnyModelConfig


class EventServiceBase:
    queue_event: str = "queue_event"
    bulk_download_event: str = "bulk_download_event"
    download_event: str = "download_event"
    model_event: str = "model_event"

    """Basic event bus, to have an empty stand-in when not needed"""

    def dispatch(self, event_name: str, payload: Any) -> None:
        pass

    def _emit_bulk_download_event(self, event_name: str, payload: dict) -> None:
        """Bulk download events are emitted to a room with queue_id as the room name"""
        payload["timestamp"] = get_timestamp()
        self.dispatch(
            event_name=EventServiceBase.bulk_download_event,
            payload={"event": event_name, "data": payload},
        )

    def __emit_queue_event(self, event_name: str, payload: dict) -> None:
        """Queue events are emitted to a room with queue_id as the room name"""
        payload["timestamp"] = get_timestamp()
        self.dispatch(
            event_name=EventServiceBase.queue_event,
            payload={"event": event_name, "data": payload},
        )

    def __emit_download_event(self, event_name: str, payload: dict) -> None:
        payload["timestamp"] = get_timestamp()
        self.dispatch(
            event_name=EventServiceBase.download_event,
            payload={"event": event_name, "data": payload},
        )

    def __emit_model_event(self, event_name: str, payload: dict) -> None:
        payload["timestamp"] = get_timestamp()
        self.dispatch(
            event_name=EventServiceBase.model_event,
            payload={"event": event_name, "data": payload},
        )

    # Define events here for every event in the system.
    # This will make them easier to integrate until we find a schema generator.
    def emit_generator_progress(
        self,
        queue_id: str,
        queue_item_id: int,
        queue_batch_id: str,
        graph_execution_state_id: str,
        node_id: str,
        source_node_id: str,
        progress_image: Optional[ProgressImage],
        step: int,
        order: int,
        total_steps: int,
    ) -> None:
        """Emitted when there is generation progress"""
        self.__emit_queue_event(
            event_name="generator_progress",
            payload={
                "queue_id": queue_id,
                "queue_item_id": queue_item_id,
                "queue_batch_id": queue_batch_id,
                "graph_execution_state_id": graph_execution_state_id,
                "node_id": node_id,
                "source_node_id": source_node_id,
                "progress_image": progress_image.model_dump() if progress_image is not None else None,
                "step": step,
                "order": order,
                "total_steps": total_steps,
            },
        )

    def emit_invocation_complete(
        self,
        queue_id: str,
        queue_item_id: int,
        queue_batch_id: str,
        graph_execution_state_id: str,
        result: dict,
        node: dict,
        source_node_id: str,
    ) -> None:
        """Emitted when an invocation has completed"""
        self.__emit_queue_event(
            event_name="invocation_complete",
            payload={
                "queue_id": queue_id,
                "queue_item_id": queue_item_id,
                "queue_batch_id": queue_batch_id,
                "graph_execution_state_id": graph_execution_state_id,
                "node": node,
                "source_node_id": source_node_id,
                "result": result,
            },
        )

    def emit_invocation_error(
        self,
        queue_id: str,
        queue_item_id: int,
        queue_batch_id: str,
        graph_execution_state_id: str,
        node: dict,
        source_node_id: str,
        error_type: str,
        error: str,
    ) -> None:
        """Emitted when an invocation has completed"""
        self.__emit_queue_event(
            event_name="invocation_error",
            payload={
                "queue_id": queue_id,
                "queue_item_id": queue_item_id,
                "queue_batch_id": queue_batch_id,
                "graph_execution_state_id": graph_execution_state_id,
                "node": node,
                "source_node_id": source_node_id,
                "error_type": error_type,
                "error": error,
            },
        )

    def emit_invocation_started(
        self,
        queue_id: str,
        queue_item_id: int,
        queue_batch_id: str,
        graph_execution_state_id: str,
        node: dict,
        source_node_id: str,
    ) -> None:
        """Emitted when an invocation has started"""
        self.__emit_queue_event(
            event_name="invocation_started",
            payload={
                "queue_id": queue_id,
                "queue_item_id": queue_item_id,
                "queue_batch_id": queue_batch_id,
                "graph_execution_state_id": graph_execution_state_id,
                "node": node,
                "source_node_id": source_node_id,
            },
        )

    def emit_graph_execution_complete(
        self, queue_id: str, queue_item_id: int, queue_batch_id: str, graph_execution_state_id: str
    ) -> None:
        """Emitted when a session has completed all invocations"""
        self.__emit_queue_event(
            event_name="graph_execution_state_complete",
            payload={
                "queue_id": queue_id,
                "queue_item_id": queue_item_id,
                "queue_batch_id": queue_batch_id,
                "graph_execution_state_id": graph_execution_state_id,
            },
        )

    def emit_model_load_started(
        self,
        queue_id: str,
        queue_item_id: int,
        queue_batch_id: str,
        graph_execution_state_id: str,
        model_config: AnyModelConfig,
    ) -> None:
        """Emitted when a model is requested"""
        self.__emit_queue_event(
            event_name="model_load_started",
            payload={
                "queue_id": queue_id,
                "queue_item_id": queue_item_id,
                "queue_batch_id": queue_batch_id,
                "graph_execution_state_id": graph_execution_state_id,
                "model_config": model_config.model_dump(),
            },
        )

    def emit_model_load_completed(
        self,
        queue_id: str,
        queue_item_id: int,
        queue_batch_id: str,
        graph_execution_state_id: str,
        model_config: AnyModelConfig,
    ) -> None:
        """Emitted when a model is correctly loaded (returns model info)"""
        self.__emit_queue_event(
            event_name="model_load_completed",
            payload={
                "queue_id": queue_id,
                "queue_item_id": queue_item_id,
                "queue_batch_id": queue_batch_id,
                "graph_execution_state_id": graph_execution_state_id,
                "model_config": model_config.model_dump(),
            },
        )

    def emit_session_canceled(
        self,
        queue_id: str,
        queue_item_id: int,
        queue_batch_id: str,
        graph_execution_state_id: str,
    ) -> None:
        """Emitted when a session is canceled"""
        self.__emit_queue_event(
            event_name="session_canceled",
            payload={
                "queue_id": queue_id,
                "queue_item_id": queue_item_id,
                "queue_batch_id": queue_batch_id,
                "graph_execution_state_id": graph_execution_state_id,
            },
        )

    def emit_queue_item_status_changed(
        self,
        session_queue_item: SessionQueueItem,
        batch_status: BatchStatus,
        queue_status: SessionQueueStatus,
    ) -> None:
        """Emitted when a queue item's status changes"""
        self.__emit_queue_event(
            event_name="queue_item_status_changed",
            payload={
                "queue_id": queue_status.queue_id,
                "queue_item": {
                    "queue_id": session_queue_item.queue_id,
                    "item_id": session_queue_item.item_id,
                    "status": session_queue_item.status,
                    "batch_id": session_queue_item.batch_id,
                    "session_id": session_queue_item.session_id,
                    "error": session_queue_item.error,
                    "created_at": str(session_queue_item.created_at) if session_queue_item.created_at else None,
                    "updated_at": str(session_queue_item.updated_at) if session_queue_item.updated_at else None,
                    "started_at": str(session_queue_item.started_at) if session_queue_item.started_at else None,
                    "completed_at": str(session_queue_item.completed_at) if session_queue_item.completed_at else None,
                },
                "batch_status": batch_status.model_dump(),
                "queue_status": queue_status.model_dump(),
            },
        )

    def emit_batch_enqueued(self, enqueue_result: EnqueueBatchResult) -> None:
        """Emitted when a batch is enqueued"""
        self.__emit_queue_event(
            event_name="batch_enqueued",
            payload={
                "queue_id": enqueue_result.queue_id,
                "batch_id": enqueue_result.batch.batch_id,
                "enqueued": enqueue_result.enqueued,
            },
        )

    def emit_queue_cleared(self, queue_id: str) -> None:
        """Emitted when the queue is cleared"""
        self.__emit_queue_event(
            event_name="queue_cleared",
            payload={"queue_id": queue_id},
        )

    def emit_download_started(self, source: str, download_path: str) -> None:
        """
        Emit when a download job is started.

        :param url: The downloaded url
        """
        self.__emit_download_event(
            event_name="download_started",
            payload={"source": source, "download_path": download_path},
        )

    def emit_download_progress(self, source: str, download_path: str, current_bytes: int, total_bytes: int) -> None:
        """
        Emit "download_progress" events at regular intervals during a download job.

        :param source: The downloaded source
        :param download_path: The local downloaded file
        :param current_bytes: Number of bytes downloaded so far
        :param total_bytes: The size of the file being downloaded (if known)
        """
        self.__emit_download_event(
            event_name="download_progress",
            payload={
                "source": source,
                "download_path": download_path,
                "current_bytes": current_bytes,
                "total_bytes": total_bytes,
            },
        )

    def emit_download_complete(self, source: str, download_path: str, total_bytes: int) -> None:
        """
        Emit a "download_complete" event at the end of a successful download.

        :param source: Source URL
        :param download_path: Path to the locally downloaded file
        :param total_bytes: The size of the downloaded file
        """
        self.__emit_download_event(
            event_name="download_complete",
            payload={
                "source": source,
                "download_path": download_path,
                "total_bytes": total_bytes,
            },
        )

    def emit_download_cancelled(self, source: str) -> None:
        """Emit a "download_cancelled" event in the event that the download was cancelled by user."""
        self.__emit_download_event(
            event_name="download_cancelled",
            payload={
                "source": source,
            },
        )

    def emit_download_error(self, source: str, error_type: str, error: str) -> None:
        """
        Emit a "download_error" event when an download job encounters an exception.

        :param source: Source URL
        :param error_type: The name of the exception that raised the error
        :param error: The traceback from this error
        """
        self.__emit_download_event(
            event_name="download_error",
            payload={
                "source": source,
                "error_type": error_type,
                "error": error,
            },
        )

    def emit_model_install_downloading(
        self,
        source: str,
        local_path: str,
        bytes: int,
        total_bytes: int,
        parts: List[Dict[str, Union[str, int]]],
    ) -> None:
        """
        Emit at intervals while the install job is in progress (remote models only).

        :param source: Source of the model
        :param local_path: Where model is downloading to
        :param parts: Progress of downloading URLs that comprise the model, if any.
        :param bytes: Number of bytes downloaded so far.
        :param total_bytes: Total size of download, including all files.
        This emits a Dict with keys "source", "local_path", "bytes" and "total_bytes".
        """
        self.__emit_model_event(
            event_name="model_install_downloading",
            payload={
                "source": source,
                "local_path": local_path,
                "bytes": bytes,
                "total_bytes": total_bytes,
                "parts": parts,
            },
        )

    def emit_model_install_running(self, source: str) -> None:
        """
        Emit once when an install job becomes active.

        :param source: Source of the model; local path, repo_id or url
        """
        self.__emit_model_event(
            event_name="model_install_running",
            payload={"source": source},
        )

    def emit_model_install_completed(self, source: str, key: str, total_bytes: Optional[int] = None) -> None:
        """
        Emit when an install job is completed successfully.

        :param source: Source of the model; local path, repo_id or url
        :param key: Model config record key
        :param total_bytes: Size of the model (may be None for installation of a local path)
        """
        self.__emit_model_event(
            event_name="model_install_completed",
            payload={
                "source": source,
                "total_bytes": total_bytes,
                "key": key,
            },
        )

    def emit_model_install_cancelled(self, source: str) -> None:
        """
        Emit when an install job is cancelled.

        :param source: Source of the model; local path, repo_id or url
        """
        self.__emit_model_event(
            event_name="model_install_cancelled",
            payload={"source": source},
        )

    def emit_model_install_error(
        self,
        source: str,
        error_type: str,
        error: str,
    ) -> None:
        """
        Emit when an install job encounters an exception.

        :param source: Source of the model
        :param error_type: The name of the exception
        :param error: A text description of the exception
        """
        self.__emit_model_event(
            event_name="model_install_error",
            payload={
                "source": source,
                "error_type": error_type,
                "error": error,
            },
        )

    def emit_bulk_download_started(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str
    ) -> None:
        """Emitted when a bulk download starts"""
        self._emit_bulk_download_event(
            event_name="bulk_download_started",
            payload={
                "bulk_download_id": bulk_download_id,
                "bulk_download_item_id": bulk_download_item_id,
                "bulk_download_item_name": bulk_download_item_name,
            },
        )

    def emit_bulk_download_completed(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str
    ) -> None:
        """Emitted when a bulk download completes"""
        self._emit_bulk_download_event(
            event_name="bulk_download_completed",
            payload={
                "bulk_download_id": bulk_download_id,
                "bulk_download_item_id": bulk_download_item_id,
                "bulk_download_item_name": bulk_download_item_name,
            },
        )

    def emit_bulk_download_failed(
        self, bulk_download_id: str, bulk_download_item_id: str, bulk_download_item_name: str, error: str
    ) -> None:
        """Emitted when a bulk download fails"""
        self._emit_bulk_download_event(
            event_name="bulk_download_failed",
            payload={
                "bulk_download_id": bulk_download_id,
                "bulk_download_item_id": bulk_download_item_id,
                "bulk_download_item_name": bulk_download_item_name,
                "error": error,
            },
        )
