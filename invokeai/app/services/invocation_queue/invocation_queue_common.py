# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import time
from typing import Optional

from pydantic import BaseModel, Field

from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID


class InvocationQueueItem(BaseModel):
    graph_execution_state_id: str = Field(description="The ID of the graph execution state")
    invocation_id: str = Field(description="The ID of the node being invoked")
    session_queue_id: str = Field(description="The ID of the session queue from which this invocation queue item came")
    session_queue_item_id: int = Field(
        description="The ID of session queue item from which this invocation queue item came"
    )
    session_queue_batch_id: str = Field(
        description="The ID of the session batch from which this invocation queue item came"
    )
    workflow: Optional[WorkflowWithoutID] = Field(description="The workflow associated with this queue item")
    invoke_all: bool = Field(default=False)
    timestamp: float = Field(default_factory=time.time)
