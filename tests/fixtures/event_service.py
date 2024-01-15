from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from invokeai.app.services.events.events_base import EventServiceBase


class DummyEvent(BaseModel):
    """Dummy Event to use with Dummy Event service."""

    event_name: str
    payload: Dict[str, Any]


# A dummy event service for testing event issuing
class DummyEventService(EventServiceBase):
    """Dummy event service for testing."""

    events: List[DummyEvent]

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    def dispatch(self, event_name: str, payload: Any) -> None:
        """Dispatch an event by appending it to self.events."""
        self.events.append(DummyEvent(event_name=payload["event"], payload=payload["data"]))


@pytest.fixture
def mock_event_service() -> EventServiceBase:
    """Create a dummy event service."""
    return DummyEventService()
