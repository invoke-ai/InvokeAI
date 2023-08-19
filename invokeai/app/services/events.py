# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Any, Optional
from invokeai.app.models.image import ProgressImage
from invokeai.app.util.misc import get_timestamp
from invokeai.app.services.model_manager_service import (
    BaseModelType,
    ModelType,
    SubModelType,
    ModelInfo,
)


class EventServiceBase:
    session_event: str = "session_event"

    """Basic event bus, to have an empty stand-in when not needed"""

    def dispatch(self, event_name: str, payload: Any) -> None:
        pass

    def __emit_session_event(self, event_name: str, payload: dict) -> None:
        payload["timestamp"] = get_timestamp()
        self.dispatch(
            event_name=EventServiceBase.session_event,
            payload=dict(event=event_name, data=payload),
        )

    # Define events here for every event in the system.
    # This will make them easier to integrate until we find a schema generator.
    def emit_generator_progress(
        self,
        graph_execution_state_id: str,
        node: dict,
        source_node_id: str,
        progress_image: Optional[ProgressImage],
        step: int,
        total_steps: int,
    ) -> None:
        """Emitted when there is generation progress"""
        self.__emit_session_event(
            event_name="generator_progress",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                node=node,
                source_node_id=source_node_id,
                progress_image=progress_image.dict() if progress_image is not None else None,
                step=step,
                total_steps=total_steps,
            ),
        )

    def emit_invocation_complete(
        self,
        graph_execution_state_id: str,
        result: dict,
        node: dict,
        source_node_id: str,
    ) -> None:
        """Emitted when an invocation has completed"""
        self.__emit_session_event(
            event_name="invocation_complete",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                node=node,
                source_node_id=source_node_id,
                result=result,
            ),
        )

    def emit_invocation_error(
        self,
        graph_execution_state_id: str,
        node: dict,
        source_node_id: str,
        error_type: str,
        error: str,
    ) -> None:
        """Emitted when an invocation has completed"""
        self.__emit_session_event(
            event_name="invocation_error",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                node=node,
                source_node_id=source_node_id,
                error_type=error_type,
                error=error,
            ),
        )

    def emit_invocation_started(self, graph_execution_state_id: str, node: dict, source_node_id: str) -> None:
        """Emitted when an invocation has started"""
        self.__emit_session_event(
            event_name="invocation_started",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                node=node,
                source_node_id=source_node_id,
            ),
        )

    def emit_graph_execution_complete(self, graph_execution_state_id: str) -> None:
        """Emitted when a session has completed all invocations"""
        self.__emit_session_event(
            event_name="graph_execution_state_complete",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
            ),
        )

    def emit_model_load_started(
        self,
        graph_execution_state_id: str,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: SubModelType,
    ) -> None:
        """Emitted when a model is requested"""
        self.__emit_session_event(
            event_name="model_load_started",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
            ),
        )

    def emit_model_load_completed(
        self,
        graph_execution_state_id: str,
        model_name: str,
        base_model: BaseModelType,
        model_type: ModelType,
        submodel: SubModelType,
        model_info: ModelInfo,
    ) -> None:
        """Emitted when a model is correctly loaded (returns model info)"""
        self.__emit_session_event(
            event_name="model_load_completed",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                model_name=model_name,
                base_model=base_model,
                model_type=model_type,
                submodel=submodel,
                hash=model_info.hash,
                location=str(model_info.location),
                precision=str(model_info.precision),
            ),
        )

    def emit_session_retrieval_error(
        self,
        graph_execution_state_id: str,
        error_type: str,
        error: str,
    ) -> None:
        """Emitted when session retrieval fails"""
        self.__emit_session_event(
            event_name="session_retrieval_error",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                error_type=error_type,
                error=error,
            ),
        )

    def emit_invocation_retrieval_error(
        self,
        graph_execution_state_id: str,
        node_id: str,
        error_type: str,
        error: str,
    ) -> None:
        """Emitted when invocation retrieval fails"""
        self.__emit_session_event(
            event_name="invocation_retrieval_error",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                node_id=node_id,
                error_type=error_type,
                error=error,
            ),
        )
