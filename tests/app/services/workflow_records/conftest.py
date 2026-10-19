from unittest import mock

import pytest

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage
from invokeai.backend.util.logging import InvokeAILogger


@pytest.fixture
def workflow_records(database: Database) -> WorkflowRecordsStorage:
    """The storage on the backend under test, started as the app starts it: the bundled workflows are stored."""
    records = WorkflowRecordsStorage(database)
    invoker = mock.Mock()
    invoker.services.logger = InvokeAILogger.get_logger()
    records.start(invoker)
    return records
