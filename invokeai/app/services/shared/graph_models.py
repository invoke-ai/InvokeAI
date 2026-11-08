from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.fields import Field

from invokeai.app.util.misc import uuid_string


class EdgeConnection(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: str = Field(description="The id of the node for this edge connection")
    field: str = Field(description="The field for this connection")

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__)
            and getattr(other, "node_id", None) == self.node_id
            and getattr(other, "field", None) == self.field
        )

    def __hash__(self):
        return hash(f"{self.node_id}.{self.field}")


class Edge(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["default", "loop_linkage"] = Field(
        default="default",
        description="The kind of relationship represented by this edge",
    )
    source: EdgeConnection = Field(description="The connection for the edge's from node and field")
    destination: EdgeConnection = Field(description="The connection for the edge's to node and field")

    def __str__(self):
        return f"{self.source.node_id}.{self.source.field} -> {self.destination.node_id}.{self.destination.field}"


PreparedExecState = Literal["pending", "ready", "executed", "skipped"]
WorkflowCallStatus = Literal["waiting_for_child", "running_child", "completed", "failed"]


class WorkflowCallFrame(BaseModel):
    """Represents one workflow-call frame in a nested call chain."""

    prepared_call_node_id: str = Field(description="The prepared exec node id for the call site.")
    source_call_node_id: str = Field(description="The source graph node id for the call site.")
    workflow_id: str = Field(description="The saved workflow being called.")
    depth: int = Field(description="The 1-based depth of this call frame.", ge=1)


class WorkflowCallExecution(BaseModel):
    """Tracks one parent/child workflow-call relationship and its lifecycle."""

    id: str = Field(description="The workflow-call execution id.", default_factory=uuid_string)
    parent_session_id: str = Field(description="The parent graph execution state id.")
    child_session_id: Optional[str] = Field(default=None, description="The child graph execution state id, if any.")
    prepared_call_node_id: str = Field(description="The prepared exec node id for the parent call site.")
    source_call_node_id: str = Field(description="The source graph node id for the parent call site.")
    workflow_id: str = Field(description="The saved workflow being called.")
    depth: int = Field(description="The 1-based depth of this call frame.", ge=1)
    status: WorkflowCallStatus = Field(description="The current workflow-call lifecycle state.")
    error_message: Optional[str] = Field(default=None, description="Failure reason, if the call failed.")
    child_session_ids: list[str] = Field(default_factory=list, description="All child graph execution state ids.")
    child_item_ids: list[int] = Field(default_factory=list, description="Child queue item ids in enqueue order.")
    expected_child_count: int = Field(default=1, ge=1, description="The number of child executions for this call.")
    completed_child_item_ids: list[int] = Field(
        default_factory=list,
        description="The child queue item ids whose workflow_return outputs have been aggregated.",
    )
    aggregated_values: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="The aggregated workflow_return values accumulated from child executions.",
    )
    child_outputs: dict[int, dict[str, Any]] = Field(
        default_factory=dict,
        description="Workflow return values keyed by child queue item id.",
    )


class WorkflowCallParentRef(BaseModel):
    """Reference from a child execution state back to its parent workflow-call relationship."""

    workflow_call_id: str = Field(description="The workflow-call execution id.")
    parent_session_id: str = Field(description="The parent graph execution state id.")
    prepared_call_node_id: str = Field(description="The prepared exec node id for the parent call site.")
    source_call_node_id: str = Field(description="The source graph node id for the parent call site.")
    workflow_id: str = Field(description="The saved workflow being called.")
    depth: int = Field(description="The 1-based depth of this call frame.", ge=1)


class ExecutionFrame(BaseModel):
    """Stable execution location for a prepared node."""

    frame_id: str = Field(default="", description="Stable frame identifier")
    state_id: str = Field(default="", description="Owning graph execution state id")
    iteration_path: tuple[int, ...] = Field(default_factory=tuple, description="Loop iteration coordinates")
    workflow_call_depth: int = Field(default=0, ge=0, description="Nested workflow-call depth")

    model_config = ConfigDict(extra="allow")


class ExecutionReference(BaseModel):
    """Stable, frame-aware reference to one prepared execution node.

    This intentionally stays independent from invocation-context/effect models. The execution engine can reconcile
    this small compatibility type with a richer protocol later without changing persisted graph state.
    """

    reference_id: str = Field(default="", description="Stable reference identifier")
    state_id: str = Field(default="", description="Owning graph execution state id")
    exec_node_id: str = Field(default="", description="Prepared execution node id")
    source_node_id: str = Field(default="", description="Authoring graph node id")
    frame: ExecutionFrame = Field(default_factory=ExecutionFrame, description="Execution frame")
    effect_count: Optional[int] = Field(default=None, ge=0, description="Expected effect count, if declared")

    model_config = ConfigDict(extra="allow")

    @property
    def id(self) -> str:
        return self.reference_id

    @property
    def node_id(self) -> str:
        return self.exec_node_id


class ExecutionToken(BaseModel):
    """Data token owned by one prepared output port.

    Tokens contain no graph edge or association metadata. In particular, loop-linkage edges never become data
    tokens.
    """

    token_id: str = Field(description="Stable token identifier")
    reference_id: str = Field(description="Owning execution reference id")
    owner_node_id: str = Field(description="Prepared node that produced the token")
    port: str = Field(description="Output port name")
    frame: ExecutionFrame = Field(description="Frame that produced the token")
    value: Any = Field(description="Output port value")
    token_kind: Literal["data", "activation", "stream_end"] = Field(default="data")

    @model_validator(mode="before")
    @classmethod
    def _restore_excluded_null_value(cls, value: Any) -> Any:
        """Allow exclude-none runtime snapshots to hydrate nullable output ports without changing the schema."""

        if isinstance(value, dict) and "value" not in value:
            return {**value, "value": None}
        return value

    sequence: int | None = Field(default=None, ge=0)

    model_config = ConfigDict(extra="allow")


# Compatibility aliases for workers that use shorter protocol names.
ExecutionRef = ExecutionReference
PreparedExecutionRef = ExecutionReference
