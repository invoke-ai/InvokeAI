import datetime
from enum import Enum
from typing import Any, Optional, Union

import semver
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator

from invokeai.app.util.metaenum import MetaEnum

__workflow_meta_version__ = semver.Version.parse("1.0.0")


class ExposedField(BaseModel):
    nodeId: str
    fieldName: str


class WorkflowNotFoundError(Exception):
    """Raised when a workflow is not found"""


class WorkflowRecordOrderBy(str, Enum, metaclass=MetaEnum):
    """The order by options for workflow records"""

    CreatedAt = "created_at"
    UpdatedAt = "updated_at"
    OpenedAt = "opened_at"
    Name = "name"


class WorkflowCategory(str, Enum, metaclass=MetaEnum):
    User = "user"
    Default = "default"
    Project = "project"


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


class ApiFieldIdentifier(BaseModel):
    node_id: str = Field(..., description="The node ID of the node that contains the field.")
    field_name: str = Field(..., description="The name of the field to include in the composed model.")
    field_name_override: str | None = Field(
        default=None,
        description="The name to use for the field in the composed model. If not provided, a unique name will be generated.",
    )
    field_title_override: str | None = Field(
        default=None,
        description="The title to use for the field in the composed model. If not provided, the field name will be used.",
    )
    field_description_override: str | None = Field(
        default=None,
        description="The description to use for the field in the composed model. If not provided, the field description will be used.",
    )


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
    api_fields: list[ApiFieldIdentifier] | None = Field(
        default=None,
        description="The API fields of the workflow. This is a list of node IDs and field names that should be included in the composed model.",
    )

    model_config = ConfigDict(extra="ignore")


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


class WorkflowRecordListItemWithThumbnailDTO(WorkflowRecordListItemDTO):
    thumbnail_url: str | None = Field(default=None, description="The URL of the workflow thumbnail.")
