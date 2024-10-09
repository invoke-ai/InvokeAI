import datetime
import json
from itertools import chain, product
from typing import Generator, Iterable, Literal, NamedTuple, Optional, TypeAlias, Union, cast

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_core import to_jsonable_python

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.services.shared.graph import Graph, GraphExecutionState, NodeNotFoundError
from invokeai.app.services.workflow_records.workflow_records_common import (
    WorkflowWithoutID,
    WorkflowWithoutIDValidator,
)
from invokeai.app.util.misc import uuid_string

# region Errors


class BatchZippedLengthError(ValueError):
    """Raise when a batch has items of different lengths."""


class BatchItemsTypeError(ValueError):  # this cannot be a TypeError in pydantic v2
    """Raise when a batch has items of different types."""


class BatchDuplicateNodeFieldError(ValueError):
    """Raise when a batch has duplicate node_path and field_name."""


class TooManySessionsError(ValueError):
    """Raise when too many sessions are requested."""


class SessionQueueItemNotFoundError(ValueError):
    """Raise when a queue item is not found."""


# endregion


# region Batch

BatchDataType = Union[
    StrictStr,
    float,
    int,
]


class NodeFieldValue(BaseModel):
    node_path: str = Field(description="The node into which this batch data item will be substituted.")
    field_name: str = Field(description="The field into which this batch data item will be substituted.")
    value: BatchDataType = Field(description="The value to substitute into the node/field.")


class BatchDatum(BaseModel):
    node_path: str = Field(description="The node into which this batch data collection will be substituted.")
    field_name: str = Field(description="The field into which this batch data collection will be substituted.")
    items: list[BatchDataType] = Field(
        default_factory=list, description="The list of items to substitute into the node/field."
    )


BatchDataCollection: TypeAlias = list[list[BatchDatum]]


class Batch(BaseModel):
    batch_id: str = Field(default_factory=uuid_string, description="The ID of the batch")
    origin: str | None = Field(
        default=None,
        description="The origin of this queue item. This data is used by the frontend to determine how to handle results.",
    )
    destination: str | None = Field(
        default=None,
        description="The origin of this queue item. This data is used by the frontend to determine how to handle results",
    )
    data: Optional[BatchDataCollection] = Field(default=None, description="The batch data collection.")
    graph: Graph = Field(description="The graph to initialize the session with")
    workflow: Optional[WorkflowWithoutID] = Field(
        default=None, description="The workflow to initialize the session with"
    )
    runs: int = Field(
        default=1, ge=1, description="Int stating how many times to iterate through all possible batch indices"
    )

    @field_validator("data")
    def validate_lengths(cls, v: Optional[BatchDataCollection]):
        if v is None:
            return v
        for batch_data_list in v:
            first_item_length = len(batch_data_list[0].items) if batch_data_list and batch_data_list[0].items else 0
            for i in batch_data_list:
                if len(i.items) != first_item_length:
                    raise BatchZippedLengthError("Zipped batch items must all have the same length")
        return v

    @field_validator("data")
    def validate_types(cls, v: Optional[BatchDataCollection]):
        if v is None:
            return v
        for batch_data_list in v:
            for datum in batch_data_list:
                # Get the type of the first item in the list
                first_item_type = type(datum.items[0]) if datum.items else None
                for item in datum.items:
                    if type(item) is not first_item_type:
                        raise BatchItemsTypeError("All items in a batch must have the same type")
        return v

    @field_validator("data")
    def validate_unique_field_mappings(cls, v: Optional[BatchDataCollection]):
        if v is None:
            return v
        paths: set[tuple[str, str]] = set()
        for batch_data_list in v:
            for datum in batch_data_list:
                pair = (datum.node_path, datum.field_name)
                if pair in paths:
                    raise BatchDuplicateNodeFieldError("Each batch data must have unique node_id and field_name")
                paths.add(pair)
        return v

    @model_validator(mode="after")
    def validate_batch_nodes_and_edges(cls, values):
        batch_data_collection = cast(Optional[BatchDataCollection], values.data)
        if batch_data_collection is None:
            return values
        graph = cast(Graph, values.graph)
        for batch_data_list in batch_data_collection:
            for batch_data in batch_data_list:
                try:
                    node = cast(BaseInvocation, graph.get_node(batch_data.node_path))
                except NodeNotFoundError:
                    raise NodeNotFoundError(f"Node {batch_data.node_path} not found in graph")
                if batch_data.field_name not in node.model_fields:
                    raise NodeNotFoundError(f"Field {batch_data.field_name} not found in node {batch_data.node_path}")
        return values

    @field_validator("graph")
    def validate_graph(cls, v: Graph):
        v.validate_self()
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "required": [
                "graph",
                "runs",
            ]
        }
    )


# endregion Batch


# region Queue Items

DEFAULT_QUEUE_ID = "default"

QUEUE_ITEM_STATUS = Literal["pending", "in_progress", "completed", "failed", "canceled"]

NodeFieldValueValidator = TypeAdapter(list[NodeFieldValue])


def get_field_values(queue_item_dict: dict) -> Optional[list[NodeFieldValue]]:
    field_values_raw = queue_item_dict.get("field_values", None)
    return NodeFieldValueValidator.validate_json(field_values_raw) if field_values_raw is not None else None


GraphExecutionStateValidator = TypeAdapter(GraphExecutionState)


def get_session(queue_item_dict: dict) -> GraphExecutionState:
    session_raw = queue_item_dict.get("session", "{}")
    session = GraphExecutionStateValidator.validate_json(session_raw, strict=False)
    return session


def get_workflow(queue_item_dict: dict) -> Optional[WorkflowWithoutID]:
    workflow_raw = queue_item_dict.get("workflow", None)
    if workflow_raw is not None:
        workflow = WorkflowWithoutIDValidator.validate_json(workflow_raw, strict=False)
        return workflow
    return None


class SessionQueueItemWithoutGraph(BaseModel):
    """Session queue item without the full graph. Used for serialization."""

    item_id: int = Field(description="The identifier of the session queue item")
    status: QUEUE_ITEM_STATUS = Field(default="pending", description="The status of this queue item")
    priority: int = Field(default=0, description="The priority of this queue item")
    batch_id: str = Field(description="The ID of the batch associated with this queue item")
    origin: str | None = Field(
        default=None,
        description="The origin of this queue item. This data is used by the frontend to determine how to handle results.",
    )
    destination: str | None = Field(
        default=None,
        description="The origin of this queue item. This data is used by the frontend to determine how to handle results",
    )
    session_id: str = Field(
        description="The ID of the session associated with this queue item. The session doesn't exist in graph_executions until the queue item is executed."
    )
    error_type: Optional[str] = Field(default=None, description="The error type if this queue item errored")
    error_message: Optional[str] = Field(default=None, description="The error message if this queue item errored")
    error_traceback: Optional[str] = Field(
        default=None,
        description="The error traceback if this queue item errored",
        validation_alias=AliasChoices("error_traceback", "error"),
    )
    created_at: Union[datetime.datetime, str] = Field(description="When this queue item was created")
    updated_at: Union[datetime.datetime, str] = Field(description="When this queue item was updated")
    started_at: Optional[Union[datetime.datetime, str]] = Field(description="When this queue item was started")
    completed_at: Optional[Union[datetime.datetime, str]] = Field(description="When this queue item was completed")
    queue_id: str = Field(description="The id of the queue with which this item is associated")
    field_values: Optional[list[NodeFieldValue]] = Field(
        default=None, description="The field values that were used for this queue item"
    )

    @classmethod
    def queue_item_dto_from_dict(cls, queue_item_dict: dict) -> "SessionQueueItemDTO":
        # must parse these manually
        queue_item_dict["field_values"] = get_field_values(queue_item_dict)
        return SessionQueueItemDTO(**queue_item_dict)

    model_config = ConfigDict(
        json_schema_extra={
            "required": [
                "item_id",
                "status",
                "batch_id",
                "queue_id",
                "session_id",
                "priority",
                "session_id",
                "created_at",
                "updated_at",
            ]
        }
    )


class SessionQueueItemDTO(SessionQueueItemWithoutGraph):
    pass


class SessionQueueItem(SessionQueueItemWithoutGraph):
    session: GraphExecutionState = Field(description="The fully-populated session to be executed")
    workflow: Optional[WorkflowWithoutID] = Field(
        default=None, description="The workflow associated with this queue item"
    )

    @classmethod
    def queue_item_from_dict(cls, queue_item_dict: dict) -> "SessionQueueItem":
        # must parse these manually
        queue_item_dict["field_values"] = get_field_values(queue_item_dict)
        queue_item_dict["session"] = get_session(queue_item_dict)
        queue_item_dict["workflow"] = get_workflow(queue_item_dict)
        return SessionQueueItem(**queue_item_dict)

    model_config = ConfigDict(
        json_schema_extra={
            "required": [
                "item_id",
                "status",
                "batch_id",
                "queue_id",
                "session_id",
                "session",
                "priority",
                "session_id",
                "created_at",
                "updated_at",
            ]
        }
    )


# endregion Queue Items

# region Query Results


class SessionQueueStatus(BaseModel):
    queue_id: str = Field(..., description="The ID of the queue")
    item_id: Optional[int] = Field(description="The current queue item id")
    batch_id: Optional[str] = Field(description="The current queue item's batch id")
    session_id: Optional[str] = Field(description="The current queue item's session id")
    pending: int = Field(..., description="Number of queue items with status 'pending'")
    in_progress: int = Field(..., description="Number of queue items with status 'in_progress'")
    completed: int = Field(..., description="Number of queue items with status 'complete'")
    failed: int = Field(..., description="Number of queue items with status 'error'")
    canceled: int = Field(..., description="Number of queue items with status 'canceled'")
    total: int = Field(..., description="Total number of queue items")


class SessionQueueCountsByDestination(BaseModel):
    queue_id: str = Field(..., description="The ID of the queue")
    destination: str = Field(..., description="The destination of queue items included in this status")
    pending: int = Field(..., description="Number of queue items with status 'pending' for the destination")
    in_progress: int = Field(..., description="Number of queue items with status 'in_progress' for the destination")
    completed: int = Field(..., description="Number of queue items with status 'complete' for the destination")
    failed: int = Field(..., description="Number of queue items with status 'error' for the destination")
    canceled: int = Field(..., description="Number of queue items with status 'canceled' for the destination")
    total: int = Field(..., description="Total number of queue items for the destination")


class BatchStatus(BaseModel):
    queue_id: str = Field(..., description="The ID of the queue")
    batch_id: str = Field(..., description="The ID of the batch")
    origin: str | None = Field(..., description="The origin of the batch")
    destination: str | None = Field(..., description="The destination of the batch")
    pending: int = Field(..., description="Number of queue items with status 'pending'")
    in_progress: int = Field(..., description="Number of queue items with status 'in_progress'")
    completed: int = Field(..., description="Number of queue items with status 'complete'")
    failed: int = Field(..., description="Number of queue items with status 'error'")
    canceled: int = Field(..., description="Number of queue items with status 'canceled'")
    total: int = Field(..., description="Total number of queue items")


class EnqueueBatchResult(BaseModel):
    queue_id: str = Field(description="The ID of the queue")
    enqueued: int = Field(description="The total number of queue items enqueued")
    requested: int = Field(description="The total number of queue items requested to be enqueued")
    batch: Batch = Field(description="The batch that was enqueued")
    priority: int = Field(description="The priority of the enqueued batch")


class ClearResult(BaseModel):
    """Result of clearing the session queue"""

    deleted: int = Field(..., description="Number of queue items deleted")


class PruneResult(ClearResult):
    """Result of pruning the session queue"""

    pass


class CancelByBatchIDsResult(BaseModel):
    """Result of canceling by list of batch ids"""

    canceled: int = Field(..., description="Number of queue items canceled")


class CancelByDestinationResult(CancelByBatchIDsResult):
    """Result of canceling by a destination"""

    pass


class CancelByQueueIDResult(CancelByBatchIDsResult):
    """Result of canceling by queue id"""

    pass


class IsEmptyResult(BaseModel):
    """Result of checking if the session queue is empty"""

    is_empty: bool = Field(..., description="Whether the session queue is empty")


class IsFullResult(BaseModel):
    """Result of checking if the session queue is full"""

    is_full: bool = Field(..., description="Whether the session queue is full")


# endregion Query Results


# region Util


def populate_graph(graph: Graph, node_field_values: Iterable[NodeFieldValue]) -> Graph:
    """
    Populates the given graph with the given batch data items.
    """
    graph_clone = graph.model_copy(deep=True)
    for item in node_field_values:
        node = graph_clone.get_node(item.node_path)
        if node is None:
            continue
        setattr(node, item.field_name, item.value)
        graph_clone.update_node(item.node_path, node)
    return graph_clone


def create_session_nfv_tuples(
    batch: Batch, maximum: int
) -> Generator[tuple[GraphExecutionState, list[NodeFieldValue], Optional[WorkflowWithoutID]], None, None]:
    """
    Create all graph permutations from the given batch data and graph. Yields tuples
    of the form (graph, batch_data_items) where batch_data_items is the list of BatchDataItems
    that was applied to the graph.
    """

    # TODO: Should this be a class method on Batch?

    data: list[list[tuple[NodeFieldValue]]] = []
    batch_data_collection = batch.data if batch.data is not None else []
    for batch_datum_list in batch_data_collection:
        # each batch_datum_list needs to be convered to NodeFieldValues and then zipped

        node_field_values_to_zip: list[list[NodeFieldValue]] = []
        for batch_datum in batch_datum_list:
            node_field_values = [
                NodeFieldValue(node_path=batch_datum.node_path, field_name=batch_datum.field_name, value=item)
                for item in batch_datum.items
            ]
            node_field_values_to_zip.append(node_field_values)
        data.append(list(zip(*node_field_values_to_zip, strict=True)))  # type: ignore [arg-type]

    # create generator to yield session,nfv tuples
    count = 0
    for _ in range(batch.runs):
        for d in product(*data):
            if count >= maximum:
                return
            flat_node_field_values = list(chain.from_iterable(d))
            graph = populate_graph(batch.graph, flat_node_field_values)
            yield (GraphExecutionState(graph=graph), flat_node_field_values, batch.workflow)
            count += 1


def calc_session_count(batch: Batch) -> int:
    """
    Calculates the number of sessions that would be created by the batch, without incurring
    the overhead of actually generating them. Adapted from `create_sessions().
    """
    # TODO: Should this be a class method on Batch?
    if not batch.data:
        return batch.runs
    data = []
    for batch_datum_list in batch.data:
        to_zip = []
        for batch_datum in batch_datum_list:
            batch_data_items = range(len(batch_datum.items))
            to_zip.append(batch_data_items)
        data.append(list(zip(*to_zip, strict=True)))
    data_product = list(product(*data))
    return len(data_product) * batch.runs


class SessionQueueValueToInsert(NamedTuple):
    """A tuple of values to insert into the session_queue table"""

    # Careful with the ordering of this - it must match the insert statement
    queue_id: str  # queue_id
    session: str  # session json
    session_id: str  # session_id
    batch_id: str  # batch_id
    field_values: Optional[str]  # field_values json
    priority: int  # priority
    workflow: Optional[str]  # workflow json
    origin: str | None
    destination: str | None


ValuesToInsert: TypeAlias = list[SessionQueueValueToInsert]


def prepare_values_to_insert(queue_id: str, batch: Batch, priority: int, max_new_queue_items: int) -> ValuesToInsert:
    values_to_insert: ValuesToInsert = []
    for session, field_values, workflow in create_session_nfv_tuples(batch, max_new_queue_items):
        # sessions must have unique id
        session.id = uuid_string()
        values_to_insert.append(
            SessionQueueValueToInsert(
                queue_id,  # queue_id
                session.model_dump_json(warnings=False, exclude_none=True),  # session (json)
                session.id,  # session_id
                batch.batch_id,  # batch_id
                # must use pydantic_encoder bc field_values is a list of models
                json.dumps(field_values, default=to_jsonable_python) if field_values else None,  # field_values (json)
                priority,  # priority
                json.dumps(workflow, default=to_jsonable_python) if workflow else None,  # workflow (json)
                batch.origin,  # origin
                batch.destination,  # destination
            )
        )
    return values_to_insert


# endregion Util

Batch.model_rebuild(force=True)
SessionQueueItem.model_rebuild(force=True)
