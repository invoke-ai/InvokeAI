from abc import ABC
from typing import Optional

from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem


class SessionProcessorABC(ABC):
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def poll_now(self) -> None:
        pass

    def get_current(self) -> Optional[SessionQueueItem]:
        pass

    def clear_current(self) -> None:
        pass
