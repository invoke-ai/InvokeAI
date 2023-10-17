# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from abc import ABC, abstractmethod
from typing import Optional

from .invocation_queue_common import InvocationQueueItem


class InvocationQueueABC(ABC):
    """Abstract base class for all invocation queues"""

    @abstractmethod
    def get(self) -> InvocationQueueItem:
        pass

    @abstractmethod
    def put(self, item: Optional[InvocationQueueItem]) -> None:
        pass

    @abstractmethod
    def cancel(self, graph_execution_state_id: str) -> None:
        pass

    @abstractmethod
    def is_canceled(self, graph_execution_state_id: str) -> bool:
        pass
