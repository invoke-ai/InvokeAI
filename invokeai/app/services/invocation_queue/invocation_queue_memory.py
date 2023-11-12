# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import time
from queue import Queue
from typing import Optional

from .invocation_queue_base import InvocationQueueABC
from .invocation_queue_common import InvocationQueueItem


class MemoryInvocationQueue(InvocationQueueABC):
    __queue: Queue
    __cancellations: dict[str, float]

    def __init__(self):
        self.__queue = Queue()
        self.__cancellations = {}

    def get(self) -> InvocationQueueItem:
        item = self.__queue.get()

        while (
            isinstance(item, InvocationQueueItem)
            and item.graph_execution_state_id in self.__cancellations
            and self.__cancellations[item.graph_execution_state_id] > item.timestamp
        ):
            item = self.__queue.get()

        # Clear old items
        for graph_execution_state_id in list(self.__cancellations.keys()):
            if self.__cancellations[graph_execution_state_id] < item.timestamp:
                del self.__cancellations[graph_execution_state_id]

        return item

    def put(self, item: Optional[InvocationQueueItem]) -> None:
        self.__queue.put(item)

    def cancel(self, graph_execution_state_id: str) -> None:
        if graph_execution_state_id not in self.__cancellations:
            self.__cancellations[graph_execution_state_id] = time.time()

    def is_canceled(self, graph_execution_state_id: str) -> bool:
        return graph_execution_state_id in self.__cancellations
