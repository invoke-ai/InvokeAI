from abc import ABC, abstractmethod
from typing import Optional

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.session_execution.session_execution_common import SessionExecutionStatusResult
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem


class SessionExecutionServiceBase(ABC):
    @abstractmethod
    def start_service(self, invoker: Invoker) -> None:
        """Service startup"""
        pass

    @abstractmethod
    def invoke_next(self) -> None:
        """Invokes the next queue item"""
        pass

    @abstractmethod
    def start(self) -> None:
        """Starts session queue execution"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops session queue execution after the currently executing session finishes"""
        pass

    @abstractmethod
    def cancel(self) -> None:
        """Stops session queue execution, immediately canceling the currently-executing session"""
        pass

    @abstractmethod
    def get_current(self) -> Optional[SessionQueueItem]:
        """Gets the currently-executing queue item"""
        pass

    @abstractmethod
    def get_status(self) -> SessionExecutionStatusResult:
        """Gets the status of the session queue"""
        pass
