from abc import ABC, abstractmethod
from threading import Event
from typing import Optional, Protocol

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.session_processor.session_processor_common import SessionProcessorStatus
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.util.profiler import Profiler


class SessionRunnerBase(ABC):
    """
    Base class for session runner.
    """

    @abstractmethod
    def start(self, services: InvocationServices, cancel_event: Event, profiler: Optional[Profiler] = None) -> None:
        """Starts the session runner.

        Args:
            services: The invocation services.
            cancel_event: The cancel event.
            profiler: The profiler to use for session profiling via cProfile. Omit to disable profiling. Basic session
                stats will be still be recorded and logged when profiling is disabled.
        """
        pass

    @abstractmethod
    def run(self, queue_item: SessionQueueItem) -> None:
        """Runs a session.

        Args:
            queue_item: The session to run.
        """
        pass

    @abstractmethod
    def run_node(self, invocation: BaseInvocation, queue_item: SessionQueueItem) -> None:
        """Run a single node in the graph.

        Args:
            invocation: The invocation to run.
            queue_item: The session queue item.
        """
        pass


class SessionProcessorBase(ABC):
    """
    Base class for session processor.

    The session processor is responsible for executing sessions. It runs a simple polling loop,
    checking the session queue for new sessions to execute. It must coordinate with the
    invocation queue to ensure only one session is executing at a time.
    """

    @abstractmethod
    def resume(self) -> SessionProcessorStatus:
        """Starts or resumes the session processor"""
        pass

    @abstractmethod
    def pause(self) -> SessionProcessorStatus:
        """Pauses the session processor"""
        pass

    @abstractmethod
    def get_status(self) -> SessionProcessorStatus:
        """Gets the status of the session processor"""
        pass


class OnBeforeRunNode(Protocol):
    def __call__(self, invocation: BaseInvocation, queue_item: SessionQueueItem) -> None:
        """Callback to run before executing a node.

        Args:
            invocation: The invocation that will be executed.
            queue_item: The session queue item.
        """
        ...


class OnAfterRunNode(Protocol):
    def __call__(self, invocation: BaseInvocation, queue_item: SessionQueueItem, output: BaseInvocationOutput) -> None:
        """Callback to run before executing a node.

        Args:
            invocation: The invocation that was executed.
            queue_item: The session queue item.
        """
        ...


class OnNodeError(Protocol):
    def __call__(
        self,
        invocation: BaseInvocation,
        queue_item: SessionQueueItem,
        error_type: str,
        error_message: str,
        error_traceback: str,
    ) -> None:
        """Callback to run when a node has an error.

        Args:
            invocation: The invocation that errored.
            queue_item: The session queue item.
            error_type: The type of error, e.g. "ValueError".
            error_message: The error message, e.g. "Invalid value".
            error_traceback: The stringified error traceback.
        """
        ...


class OnBeforeRunSession(Protocol):
    def __call__(self, queue_item: SessionQueueItem) -> None:
        """Callback to run before executing a session.

        Args:
            queue_item: The session queue item.
        """
        ...


class OnAfterRunSession(Protocol):
    def __call__(self, queue_item: SessionQueueItem) -> None:
        """Callback to run after executing a session.

        Args:
            queue_item: The session queue item.
        """
        ...


class OnNonFatalProcessorError(Protocol):
    def __call__(
        self,
        queue_item: Optional[SessionQueueItem],
        error_type: str,
        error_message: str,
        error_traceback: str,
    ) -> None:
        """Callback to run when a non-fatal error occurs in the processor.

        Args:
            queue_item: The session queue item, if one was being executed when the error occurred.
            error_type: The type of error, e.g. "ValueError".
            error_message: The error message, e.g. "Invalid value".
            error_traceback: The stringified error traceback.
        """
        ...
