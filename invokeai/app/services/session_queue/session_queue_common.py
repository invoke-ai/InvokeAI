import datetime
import json
from itertools import chain, product
from typing import Iterable, Literal, Optional, TypeAlias, Union, cast

from pydantic import BaseModel, Field, StrictStr, parse_raw_as, root_validator, validator
from pydantic.json import pydantic_encoder

from invokeai.app.invocations.baseinvocation import BaseInvocation
from invokeai.app.services.graph import Graph, GraphExecutionState, NodeNotFoundError
from invokeai.app.util.misc import uuid_string

# region Errors


class BatchZippedLengthError(ValueError):
    """Raise when a batch has items of different lengths."""


class BatchItemsTypeError(TypeError):
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
    data: Optional[BatchDataCollection] = Field(default=None, description="The batch data collection.")
    graph: Graph = Field(description="The graph to initialize the session with")
    runs: int = Field(
        default=1, ge=1, description="Int stating how many times to iterate through all possible batch indices"
    )

    @validator("data")
    def validate_len(cls, v: Optional[BatchDataCollection]):
        if v is None:
            return v
        for batch_data_list in v:
            if any(len(batch_data_list[0].items) != len(i.items) for i in batch_data_list):
                raise BatchZippedLengthError("Zipped batch items must have all have same length")
        return v

    @validator("data")
    def validate_types(cls, v: Optional[BatchDataCollection]):
        if v is None:
            return v
        for batch_data_list in v:
            for datum in batch_data_list:
                for item in datum.items:
                    if not all(isinstance(item, type(i)) for i in datum.items):
                        raise BatchItemsTypeError("All items in a batch must have have same type")
        return v

    @validator("data")
    def validate_unique_field_mappings(cls, v: Optional[BatchDataCollection]):
        if v is None:
            return v
        paths: set[tuple[str, str]] = set()
        count: int = 0
        for batch_data_list in v:
            for datum in batch_data_list:
                paths.add((datum.node_path, datum.field_name))
                count += 1
        if len(paths) != count:
            raise BatchDuplicateNodeFieldError("Each batch data must have unique node_id and field_name")
        return v

    @root_validator(skip_on_failure=True)
    def validate_batch_nodes_and_edges(cls, values):
        batch_data_collection = cast(Optional[BatchDataCollection], values["data"])
        if batch_data_collection is None:
            return values
        graph = cast(Graph, values["graph"])
        for batch_data_list in batch_data_collection:
            for batch_data in batch_data_list:
                try:
                    node = cast(BaseInvocation, graph.get_node(batch_data.node_path))
                except NodeNotFoundError:
                    raise NodeNotFoundError(f"Node {batch_data.node_path} not found in graph")
                if batch_data.field_name not in node.__fields__:
                    raise NodeNotFoundError(f"Field {batch_data.field_name} not found in node {batch_data.node_path}")
        return values

    class Config:
        schema_extra = {
            "required": [
                "graph",
                "runs",
            ]
        }


# endregion Batch


# region Queue Items

DEFAULT_QUEUE_ID = "default"

QUEUE_ITEM_STATUS = Literal["pending", "in_progress", "completed", "failed", "canceled"]


def get_field_values(queue_item_dict: dict) -> Optional[list[NodeFieldValue]]:
    field_values_raw = queue_item_dict.get("field_values", None)
    return parse_raw_as(list[NodeFieldValue], field_values_raw) if field_values_raw is not None else None


def get_session(queue_item_dict: dict) -> GraphExecutionState:
    session_raw = queue_item_dict.get("session", "{}")
    return parse_raw_as(GraphExecutionState, session_raw)


class SessionQueueItemWithoutGraph(BaseModel):
    """Session queue item without the full graph. Used for serialization."""

    item_id: str = Field(description="The unique identifier of the session queue item")
    order_id: int = Field(description="The auto-incrementing ID of the session queue item")
    status: QUEUE_ITEM_STATUS = Field(default="pending", description="The status of this queue item")
    priority: int = Field(default=0, description="The priority of this queue item")
    batch_id: str = Field(description="The ID of the batch associated with this queue item")
    session_id: str = Field(
        description="The ID of the session associated with this queue item. The session doesn't exist in graph_executions until the queue item is executed."
    )
    field_values: Optional[list[NodeFieldValue]] = Field(
        default=None, description="The field values that were used for this queue item"
    )
    queue_id: str = Field(description="The id of the queue with which this item is associated")
    error: Optional[str] = Field(default=None, description="The error message if this queue item errored")
    created_at: Union[datetime.datetime, str] = Field(description="When this queue item was created")
    updated_at: Union[datetime.datetime, str] = Field(description="When this queue item was updated")
    started_at: Optional[Union[datetime.datetime, str]] = Field(description="When this queue item was started")
    completed_at: Optional[Union[datetime.datetime, str]] = Field(description="When this queue item was completed")

    @classmethod
    def from_dict(cls, queue_item_dict: dict) -> "SessionQueueItemDTO":
        # must parse these manually
        queue_item_dict["field_values"] = get_field_values(queue_item_dict)
        return SessionQueueItemDTO(**queue_item_dict)

    class Config:
        schema_extra = {
            "required": [
                "item_id",
                "order_id",
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


class SessionQueueItemDTO(SessionQueueItemWithoutGraph):
    pass


class SessionQueueItem(SessionQueueItemWithoutGraph):
    session: GraphExecutionState = Field(description="The fully-populated session to be executed")

    @classmethod
    def from_dict(cls, queue_item_dict: dict) -> "SessionQueueItem":
        # must parse these manually
        queue_item_dict["field_values"] = get_field_values(queue_item_dict)
        queue_item_dict["session"] = get_session(queue_item_dict)
        return SessionQueueItem(**queue_item_dict)

    class Config:
        schema_extra = {
            "required": [
                "item_id",
                "order_id",
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


# endregion Queue Items

# region Query Results


class SessionQueueStatus(BaseModel):
    queue_id: str = Field(..., description="The ID of the queue")
    item_id: Optional[str] = Field(description="The current queue item id")
    batch_id: Optional[str] = Field(description="The current queue item's batch id")
    session_id: Optional[str] = Field(description="The current queue item's session id")
    pending: int = Field(..., description="Number of queue items with status 'pending'")
    in_progress: int = Field(..., description="Number of queue items with status 'in_progress'")
    completed: int = Field(..., description="Number of queue items with status 'complete'")
    failed: int = Field(..., description="Number of queue items with status 'error'")
    canceled: int = Field(..., description="Number of queue items with status 'canceled'")
    total: int = Field(..., description="Total number of queue items")


class BatchStatus(BaseModel):
    queue_id: str = Field(..., description="The ID of the queue")
    batch_id: str = Field(..., description="The ID of the batch")
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


class EnqueueGraphResult(BaseModel):
    enqueued: int = Field(description="The total number of queue items enqueued")
    requested: int = Field(description="The total number of queue items requested to be enqueued")
    batch: Batch = Field(description="The batch that was enqueued")
    priority: int = Field(description="The priority of the enqueued batch")
    queue_item: SessionQueueItemDTO = Field(description="The queue item that was enqueued")


class ClearResult(BaseModel):
    """Result of clearing the session queue"""

    deleted: int = Field(..., description="Number of queue items deleted")


class PruneResult(ClearResult):
    """Result of pruning the session queue"""

    pass


class CancelByBatchIDsResult(BaseModel):
    """Result of canceling by list of batch ids"""

    canceled: int = Field(..., description="Number of queue items canceled")


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
    graph = graph.copy(deep=True)
    for item in node_field_values:
        node = graph.get_node(item.node_path)
        if node is None:
            continue
        setattr(node, item.field_name, item.value)
        graph.update_node(item.node_path, node)
    return graph


def create_session_nfv_tuples(batch: Batch, maximum: int) -> list[tuple[GraphExecutionState, list[NodeFieldValue]]]:
    """
    Create all graph permutations from the given batch data and graph. Returns a list of tuples
    of the form (graph, batch_data_items) where batch_data_items is the list of BatchDataItems
    that was applied to the graph.
    """
    # TODO: Should this be a class method on Batch?
    data: list[list[tuple[NodeFieldValue]]] = []
    batch_data_collection = batch.data if batch.data is not None else []
    for batch_datum_list in batch_data_collection:
        # each batch_datum_list contains a list of BatchDatums to be convered to NodeFieldValues and then zipped
        node_field_values_to_zip: list[list[NodeFieldValue]] = []
        for batch_datum in batch_datum_list:
            # derive NodeFieldValues from the BatchDatum by mapping over its items
            node_field_values = [
                NodeFieldValue(node_path=batch_datum.node_path, field_name=batch_datum.field_name, value=item)
                for item in batch_datum.items
            ]
            node_field_values_to_zip.append(node_field_values)
        # zip the NodeFieldValues together
        data.append(list(zip(*node_field_values_to_zip)))
    # take the cartesian product of the zipped lists of NodeFieldValue, then flatten them
    node_field_values = [list(chain.from_iterable(d)) for d in product(*data)]
    # create graphs for each permutation
    graphs = [populate_graph(batch.graph, nfv) for nfv in node_field_values]
    sessions_and_node_field_values = []
    # create sessions and NodeFieldValues for permutations * runs, bailing if/when we hit the max
    for _ in range(batch.runs):
        if len(sessions_and_node_field_values) >= maximum:
            break
        for idx, nfv in enumerate(node_field_values):
            if len(sessions_and_node_field_values) >= maximum:
                break
            sessions_and_node_field_values.append((GraphExecutionState(graph=graphs[idx]), nfv))
    return sessions_and_node_field_values


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
        data.append(list(zip(*to_zip)))
    data_product = list(product(*data))
    return len(data_product) * batch.runs


ValuesToInsert: TypeAlias = list[
    tuple[
        str,  # item_id
        str,  # queue_id
        str,  # session json
        str,  # session_id
        str,  # batch_id
        Optional[str],  # field_values json
        int,  # priority
        int,  # order_id
    ]
]
"""(item_id, queue_id, session (json), session_id, batch_id, field_values (json), priority)"""


def prepare_values_to_insert(
    queue_id: str, batch: Batch, priority: int, max_new_queue_items: int, order_id: int
) -> ValuesToInsert:
    values_to_insert: ValuesToInsert = []
    session_and_field_value_tuples = create_session_nfv_tuples(batch, max_new_queue_items)
    for session, field_values in session_and_field_value_tuples:
        # sessions must have unique id
        session.id = uuid_string()
        values_to_insert.append(
            (
                uuid_string(),  # item_id
                queue_id,  # queue_id
                session.json(),  # session (json)
                session.id,  # session_id
                batch.batch_id,  # batch_id
                # must use pydantic_encoder bc field_values is a list of models
                json.dumps(field_values, default=pydantic_encoder) if field_values else None,  # field_values (json)
                priority,  # priority
                order_id,
            )
        )
        order_id += 1
    return values_to_insert


# endregion Util
