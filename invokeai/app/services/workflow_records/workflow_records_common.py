import datetime
from enum import Enum
from typing import Any, Optional, Union

import semver
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator

from invokeai.app.services.shared.workflow_call_compatibility_common import WorkflowCallCompatibility
from invokeai.app.util.metaenum import MetaEnum

__workflow_meta_version__ = semver.Version.parse("1.0.0")

WORKFLOW_LIBRARY_DEFAULT_USER_ID = "system"
"""Default user_id for workflows created in single-user mode or migrated from pre-multiuser databases."""


class ExposedField(BaseModel):
    nodeId: str
    fieldName: str


class WorkflowNotFoundError(Exception):
    """Raised when a workflow is not found"""


class WorkflowImmutableError(Exception):
    """Raised when a write targets a bundled (default) workflow, which the server never lets anyone modify."""

    def __init__(self, workflow_id: str) -> None:
        self.workflow_id = workflow_id
        super().__init__(f"Workflow {workflow_id} is a bundled workflow and cannot be modified")


class WorkflowAccessDeniedError(Exception):
    """Raised when a scoped write targets a workflow another account owns."""

    def __init__(self, workflow_id: str) -> None:
        self.workflow_id = workflow_id
        super().__init__(f"Workflow {workflow_id} is owned by another account")


class WorkflowRevisionConflictError(Exception):
    """Raised when an update carries a stale content revision (another writer saved first)."""

    def __init__(self, workflow_id: str, expected_revision: int, current_revision: int) -> None:
        self.workflow_id = workflow_id
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        super().__init__(
            f"Workflow {workflow_id} is at revision {current_revision}; the update expected revision {expected_revision}"
        )


class WorkflowIdConflictError(Exception):
    """Raised when a client-reserved workflow id already names a record that is not the same request."""

    def __init__(self, workflow_id: str) -> None:
        self.workflow_id = workflow_id
        super().__init__(f"Workflow id {workflow_id} is already in use")


class WorkflowRecordOrderBy(str, Enum, metaclass=MetaEnum):
    """The order by options for workflow records"""

    CreatedAt = "created_at"
    UpdatedAt = "updated_at"
    OpenedAt = "opened_at"
    Name = "name"
    IsPublic = "is_public"


class WorkflowCategory(str, Enum, metaclass=MetaEnum):
    User = "user"
    Default = "default"


class WorkflowMeta(BaseModel):
    version: str = Field(description="The version of the workflow schema.")
    category: WorkflowCategory = Field(description="The category of the workflow (user or default).")

    @field_validator("version")
    def validate_version(cls, version: str):
        try:
            semver.Version.parse(version)
            return version
        except Exception:
            raise ValueError(f"Invalid workflow meta version: {version}")

    def to_semver(self) -> semver.Version:
        return semver.Version.parse(self.version)


class WorkflowWithoutID(BaseModel):
    name: str = Field(description="The name of the workflow.")
    author: str = Field(description="The author of the workflow.")
    description: str = Field(description="The description of the workflow.")
    version: str = Field(description="The version of the workflow.")
    contact: str = Field(description="The contact of the workflow.")
    tags: str = Field(description="The tags of the workflow.")
    notes: str = Field(description="The notes of the workflow.")
    exposedFields: list[ExposedField] = Field(description="The exposed fields of the workflow.")
    meta: WorkflowMeta = Field(description="The meta of the workflow.")
    # TODO(psyche): nodes, edges and form are very loosely typed - they are strictly modeled and checked on the frontend.
    nodes: list[dict[str, JsonValue]] = Field(description="The nodes of the workflow.")
    edges: list[dict[str, JsonValue]] = Field(description="The edges of the workflow.")
    # TODO(psyche): We have a crapload of workflows that have no form, bc it was added after we introduced workflows.
    # This is typed as optional to prevent errors when pulling workflows from the DB. The frontend adds a default form if
    # it is None.
    form: dict[str, JsonValue] | None = Field(default=None, description="The form of the workflow.")

    model_config = ConfigDict(extra="ignore")

    @field_validator("nodes")
    @classmethod
    def validate_workflow_return_node_uniqueness(cls, nodes: list[dict[str, JsonValue]]):
        workflow_return_count = 0

        for node in nodes:
            if not isinstance(node, dict) or node.get("type") != "invocation":
                continue
            data = node.get("data")
            if isinstance(data, dict) and data.get("type") == "workflow_return":
                workflow_return_count += 1

        if workflow_return_count > 1:
            raise ValueError("A workflow may not contain more than one workflow_return node.")

        return nodes


WorkflowWithoutIDValidator = TypeAdapter(WorkflowWithoutID)


class UnsafeWorkflowWithVersion(BaseModel):
    """
    This utility model only requires a workflow to have a valid version string.
    It is used to validate a workflow version without having to validate the entire workflow.
    """

    meta: WorkflowMeta = Field(description="The meta of the workflow.")


UnsafeWorkflowWithVersionValidator = TypeAdapter(UnsafeWorkflowWithVersion)


class Workflow(WorkflowWithoutID):
    id: str = Field(description="The id of the workflow.")


WorkflowValidator = TypeAdapter(Workflow)


class WorkflowRecordDTOBase(BaseModel):
    workflow_id: str = Field(description="The id of the workflow.")
    name: str = Field(description="The name of the workflow.")
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the workflow.")
    updated_at: Union[datetime.datetime, str] = Field(description="The updated timestamp of the workflow.")
    opened_at: Optional[Union[datetime.datetime, str]] = Field(
        default=None, description="The opened timestamp of the workflow."
    )
    last_run_at: Optional[Union[datetime.datetime, str]] = Field(
        default=None, description="The timestamp of the last completed run of this workflow."
    )
    user_id: str = Field(description="The id of the user who owns this workflow.")
    is_public: bool = Field(description="Whether this workflow is shared with all users.")
    revision: int = Field(description="Monotonic content revision; every write of the workflow document increments it.")


class WorkflowRecordDTO(WorkflowRecordDTOBase):
    workflow: Workflow = Field(description="The workflow.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowRecordDTO":
        data["workflow"] = WorkflowValidator.validate_json(data.get("workflow", ""))
        return WorkflowRecordDTOValidator.validate_python(data)


WorkflowRecordDTOValidator = TypeAdapter(WorkflowRecordDTO)


class WorkflowRecordListItemDTO(WorkflowRecordDTOBase):
    description: str = Field(description="The description of the workflow.")
    category: WorkflowCategory = Field(description="The description of the workflow.")
    tags: str = Field(description="The tags of the workflow.")


WorkflowRecordListItemDTOValidator = TypeAdapter(WorkflowRecordListItemDTO)


class WorkflowRecordWithThumbnailDTO(WorkflowRecordDTO):
    thumbnail_url: str | None = Field(default=None, description="The URL of the workflow thumbnail.")
    call_saved_workflow_compatibility: WorkflowCallCompatibility | None = Field(
        default=None, description="Whether this workflow is currently callable by call_saved_workflow."
    )


class WorkflowRecordListItemWithThumbnailDTO(WorkflowRecordListItemDTO):
    thumbnail_url: str | None = Field(default=None, description="The URL of the workflow thumbnail.")
    call_saved_workflow_compatibility: WorkflowCallCompatibility | None = Field(
        default=None, description="Whether this workflow is currently callable by call_saved_workflow."
    )
