from abc import ABC, abstractmethod
from threading import Event

from invokeai.app.services.invocation_services import InvocationServices
from invokeai.app.services.session_processor.session_processor_common import SessionProcessorStatus
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem


class SessionRunnerBase(ABC):
    """
    Base class for session runner.
    """

    @abstractmethod
    def start(self, services: InvocationServices, cancel_event: Event) -> None:
        """Starts the session runner"""
        pass

    @abstractmethod
    def run(self, queue_item: SessionQueueItem) -> None:
        """Runs the session"""
        pass

    @abstractmethod
    def complete(self, queue_item: SessionQueueItem) -> None:
        """Completes the session"""
        pass

    @abstractmethod
    def run_node(self, node_id: str, queue_item: SessionQueueItem) -> None:
        """Runs an already prepared node on the session"""
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
