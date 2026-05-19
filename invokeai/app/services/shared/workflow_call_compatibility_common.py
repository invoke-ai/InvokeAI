from enum import Enum

from pydantic import BaseModel, Field


class WorkflowCallCompatibilityReason(str, Enum):
    Ok = "ok"
    MissingWorkflowReturn = "missing_workflow_return"
    MultipleWorkflowReturn = "multiple_workflow_return"
    UnsupportedNode = "unsupported_node"
    UnsupportedBatchInput = "unsupported_batch_input"
    InvalidGraph = "invalid_graph"
    InvalidInputs = "invalid_inputs"
    Unknown = "unknown"


class WorkflowCallCompatibility(BaseModel):
    is_callable: bool = Field(description="Whether the workflow can currently be executed by call_saved_workflow.")
    reason: WorkflowCallCompatibilityReason = Field(description="Structured compatibility result.")
    message: str | None = Field(default=None, description="Human-readable compatibility detail when unavailable.")
