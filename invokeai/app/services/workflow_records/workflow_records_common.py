from typing import Any

from pydantic import Field, RootModel, TypeAdapter


class WorkflowNotFoundError(Exception):
    """Raised when a workflow is not found"""


class WorkflowField(RootModel):
    """
    Pydantic model for workflows with custom root of type dict[str, Any].
    Workflows are stored without a strict schema.
    """

    root: dict[str, Any] = Field(description="Workflow dict")


type_adapter_WorkflowField = TypeAdapter(WorkflowField)
