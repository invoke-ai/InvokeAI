# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import Any, Dict


class EventServiceBase:
    session_event: str = "session_event"

    """Basic event bus, to have an empty stand-in when not needed"""

    def dispatch(self, event_name: str, payload: Any) -> None:
        pass

    def __emit_session_event(self, event_name: str, payload: Dict) -> None:
        self.dispatch(
            event_name=EventServiceBase.session_event,
            payload=dict(event=event_name, data=payload),
        )

    # Define events here for every event in the system.
    # This will make them easier to integrate until we find a schema generator.
    def emit_generator_progress(
        self,
        graph_execution_state_id: str,
        invocation_id: str,
        step: int,
        percent: float,
    ) -> None:
        """Emitted when there is generation progress"""
        self.__emit_session_event(
            event_name="generator_progress",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                invocation_id=invocation_id,
                step=step,
                percent=percent,
            ),
        )

    def emit_invocation_complete(
        self, graph_execution_state_id: str, invocation_id: str, result: Dict
    ) -> None:
        """Emitted when an invocation has completed"""
        self.__emit_session_event(
            event_name="invocation_complete",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                invocation_id=invocation_id,
                result=result,
            ),
        )

    def emit_invocation_error(
        self, graph_execution_state_id: str, invocation_id: str, error: str
    ) -> None:
        """Emitted when an invocation has completed"""
        self.__emit_session_event(
            event_name="invocation_error",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                invocation_id=invocation_id,
                error=error,
            ),
        )

    def emit_invocation_started(
        self, graph_execution_state_id: str, invocation_id: str
    ) -> None:
        """Emitted when an invocation has started"""
        self.__emit_session_event(
            event_name="invocation_started",
            payload=dict(
                graph_execution_state_id=graph_execution_state_id,
                invocation_id=invocation_id,
            ),
        )

    def emit_graph_execution_complete(self, graph_execution_state_id: str) -> None:
        """Emitted when a session has completed all invocations"""
        self.__emit_session_event(
            event_name="graph_execution_state_complete",
            payload=dict(graph_execution_state_id=graph_execution_state_id),
        )
