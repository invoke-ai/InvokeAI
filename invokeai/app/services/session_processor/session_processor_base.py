from abc import ABC, abstractmethod

from invokeai.app.services.session_processor.session_processor_common import SessionProcessorStatus


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
