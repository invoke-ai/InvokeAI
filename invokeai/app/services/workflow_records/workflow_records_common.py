import datetime
from typing import Any, Union

import semver
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, field_validator

from invokeai.app.util.misc import uuid_string

__workflow_meta_version__ = semver.Version.parse("1.0.0")


class ExposedField(BaseModel):
    nodeId: str
    fieldName: str


class WorkflowMeta(BaseModel):
    version: str = Field(description="The version of the workflow schema.")

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
    # TODO: nodes and edges are very loosely typed
    nodes: list[dict[str, JsonValue]] = Field(description="The nodes of the workflow.")
    edges: list[dict[str, JsonValue]] = Field(description="The edges of the workflow.")


WorkflowWithoutIDValidator = TypeAdapter(WorkflowWithoutID)


class Workflow(WorkflowWithoutID):
    id: str = Field(default_factory=uuid_string, description="The id of the workflow.")


WorkflowValidator = TypeAdapter(Workflow)


class WorkflowRecordDTO(BaseModel):
    workflow_id: str = Field(description="The id of the workflow.")
    workflow: Workflow = Field(description="The workflow.")
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the workflow.")
    updated_at: Union[datetime.datetime, str] = Field(description="The updated timestamp of the workflow.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowRecordDTO":
        data["workflow"] = WorkflowValidator.validate_json(data.get("workflow", ""))
        return WorkflowRecordDTOValidator.validate_python(data)


WorkflowRecordDTOValidator = TypeAdapter(WorkflowRecordDTO)


class WorkflowRecordListItemDTO(BaseModel):
    workflow_id: str = Field(description="The id of the workflow.")
    name: str = Field(description="The name of the workflow.")
    description: str = Field(description="The description of the workflow.")
    created_at: Union[datetime.datetime, str] = Field(description="The created timestamp of the workflow.")
    updated_at: Union[datetime.datetime, str] = Field(description="The updated timestamp of the workflow.")


WorkflowRecordListItemDTOValidator = TypeAdapter(WorkflowRecordListItemDTO)


class WorkflowNotFoundError(Exception):
    """Raised when a workflow is not found"""
