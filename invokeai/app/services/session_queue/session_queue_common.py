import datetime
import json
from itertools import chain, product
from typing import Generator, Literal, Optional, TypeAlias, Union, cast

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
from invokeai.app.invocations.fields import ImageField
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

BatchDataType = Union[StrictStr, float, int, ImageField]


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
                if not datum.items:
                    continue

                # Special handling for numbers - they can be mixed
                # TODO(psyche): Update BatchDatum to have a `type` field to specify the type of the items, then we can have strict float and int fields
                if all(isinstance(item, (int, float)) for item in datum.items):
                    continue

                # Get the type of the first item in the list
                first_item_type = type(datum.items[0])
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
                if batch_data.field_name not in type(node).model_fields:
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


class FieldIdentifier(BaseModel):
    kind: Literal["input", "output"] = Field(description="The kind of field")
    node_id: str = Field(description="The ID of the node")
    field_name: str = Field(description="The name of the field")
    user_label: str | None = Field(description="The user label of the field, if any")


class SessionQueueItem(BaseModel):
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
    retried_from_item_id: Optional[int] = Field(
        default=None, description="The item_id of the queue item that this item was retried from"
    )
    is_api_validation_run: bool = Field(
        default=False,
        description="Whether this queue item is an API validation run.",
    )
    published_workflow_id: Optional[str] = Field(
        default=None,
        description="The ID of the published workflow associated with this queue item",
    )
    credits: Optional[float] = Field(default=None, description="The total credits used for this queue item")
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
    item_ids: list[int] = Field(description="The IDs of the queue items that were enqueued")


class RetryItemsResult(BaseModel):
    queue_id: str = Field(description="The ID of the queue")
    retried_item_ids: list[int] = Field(description="The IDs of the queue items that were retried")


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


class DeleteByDestinationResult(BaseModel):
    """Result of deleting by a destination"""

    deleted: int = Field(..., description="Number of queue items deleted")


class DeleteAllExceptCurrentResult(DeleteByDestinationResult):
    """Result of deleting all except current"""

    pass


class CancelByQueueIDResult(CancelByBatchIDsResult):
    """Result of canceling by queue id"""

    pass


class CancelAllExceptCurrentResult(CancelByBatchIDsResult):
    """Result of canceling all except current"""

    pass


class IsEmptyResult(BaseModel):
    """Result of checking if the session queue is empty"""

    is_empty: bool = Field(..., description="Whether the session queue is empty")


class IsFullResult(BaseModel):
    """Result of checking if the session queue is full"""

    is_full: bool = Field(..., description="Whether the session queue is full")


# endregion Query Results


# region Util


def create_session_nfv_tuples(batch: Batch, maximum: int) -> Generator[tuple[str, str, str], None, None]:
    """
    Given a batch and a maximum number of sessions to create, generate a tuple of session_id, session_json, and
    field_values_json for each session.

    The batch has a "source" graph and a data property. The data property is a list of lists of BatchDatum objects.
    Each BatchDatum has a field identifier (e.g. a node id and field name), and a list of values to substitute into
    the field.

    This structure allows us to create a new graph for every possible permutation of BatchDatum objects:
    - Each BatchDatum can be "expanded" into a dict of node-field-value tuples - one for each item in the BatchDatum.
    - Zip each inner list of expanded BatchDatum objects together. Call this a "batch_data_list".
    - Take the cartesian product of all zipped batch_data_lists, resulting in a list of permutations of BatchDatum
    - Take the cartesian product of all zipped batch_data_lists, resulting in a list of lists of BatchDatum objects.
        Each inner list now represents the substitution values for a single permutation (session).
    - For each permutation, substitute the values into the graph

    This function is optimized for performance, as it is used to generate a large number of sessions at once.

    Args:
        batch: The batch to generate sessions from
        maximum: The maximum number of sessions to generate

    Returns:
        A generator that yields tuples of session_id, session_json, and field_values_json for each session. The
        generator will stop early if the maximum number of sessions is reached.
    """

    # TODO: Should this be a class method on Batch?

    data: list[list[tuple[dict]]] = []
    batch_data_collection = batch.data if batch.data is not None else []

    for batch_datum_list in batch_data_collection:
        node_field_values_to_zip: list[list[dict]] = []
        # Expand each BatchDatum into a list of dicts - one for each item in the BatchDatum
        for batch_datum in batch_datum_list:
            node_field_values = [
                # Note: A tuple here is slightly faster than a dict, but we need the object in dict form to be inserted
                # in the session_queue table anyways. So, overall creating NFVs as dicts is faster.
                {"node_path": batch_datum.node_path, "field_name": batch_datum.field_name, "value": item}
                for item in batch_datum.items
            ]
            node_field_values_to_zip.append(node_field_values)
        # Zip the dicts together to create a list of dicts for each permutation
        data.append(list(zip(*node_field_values_to_zip, strict=True)))  # type: ignore [arg-type]

    # We serialize the graph and session once, then mutate the graph dict in place for each session.
    #
    # This sounds scary, but it's actually fine.
    #
    # The batch prep logic injects field values into the same fields for each generated session.
    #
    # For example, after the product operation, we'll end up with a list of node-field-value tuples like this:
    # [
    #   (
    #     {"node_path": "1", "field_name": "a", "value": 1},
    #     {"node_path": "2", "field_name": "b", "value": 2},
    #     {"node_path": "3", "field_name": "c", "value": 3},
    #   ),
    #   (
    #     {"node_path": "1", "field_name": "a", "value": 4},
    #     {"node_path": "2", "field_name": "b", "value": 5},
    #     {"node_path": "3", "field_name": "c", "value": 6},
    #   )
    # ]
    #
    # Note that each tuple has the same length, and each tuple substitutes values in for exactly the same node fields.
    # No matter the complexity of the batch, this property holds true.
    #
    # This means each permutation's substitution can be done in-place on the same graph dict, because it overwrites the
    # previous mutation. We only need to serialize the graph once, and then we can mutate it in place for each session.
    #
    # Previously, we had created new Graph objects for each session, but this was very slow for large (1k+ session
    # batches). We then tried dumping the graph to dict and using deep-copy to create a new dict for each session,
    # but this was also slow.
    #
    # Overall, we achieved a 100x speedup by mutating the graph dict in place for each session over creating new Graph
    # objects for each session.
    #
    # We will also mutate the session dict in place, setting a new ID for each session and setting the mutated graph
    # dict as the session's graph.

    # Dump the batch's graph to a dict once
    graph_as_dict = batch.graph.model_dump(warnings=False, exclude_none=True)

    # We must provide a Graph object when creating the "dummy" session dict, but we don't actually use it. It will be
    # overwritten for each session by the mutated graph_as_dict.
    session_dict = GraphExecutionState(graph=Graph()).model_dump(warnings=False, exclude_none=True)

    # Now we can create a generator that yields the session_id, session_json, and field_values_json for each session.
    count = 0

    # Each batch may have multiple runs, so we need to generate the same number of sessions for each run. The total is
    # still limited by the maximum number of sessions.
    for _ in range(batch.runs):
        for d in product(*data):
            if count >= maximum:
                # We've reached the maximum number of sessions we may generate
                return

            # Flatten the list of lists of dicts into a single list of dicts
            # TODO(psyche): Is the a more efficient way to do this?
            flat_node_field_values = list(chain.from_iterable(d))

            # Need a fresh ID for each session
            session_id = uuid_string()

            # Mutate the session dict in place
            session_dict["id"] = session_id

            # Substitute the values into the graph
            for nfv in flat_node_field_values:
                graph_as_dict["nodes"][nfv["node_path"]][nfv["field_name"]] = nfv["value"]

            # Mutate the session dict in place
            session_dict["graph"] = graph_as_dict

            # Serialize the session and field values
            # Note the use of pydantic's to_jsonable_python to handle serialization of any python object, including sets.
            session_json = json.dumps(session_dict, default=to_jsonable_python)
            field_values_json = json.dumps(flat_node_field_values, default=to_jsonable_python)

            # Yield the session_id, session_json, and field_values_json
            yield (session_id, session_json, field_values_json)

            # Increment the count so we know when to stop
            count += 1


def calc_session_count(batch: Batch) -> int:
    """
    Calculates the number of sessions that would be created by the batch, without incurring the overhead of actually
    creating them, as is done in `create_session_nfv_tuples()`.

    The count is used to communicate to the user how many sessions were _requested_ to be created, as opposed to how
    many were _actually_ created (which may be less due to the maximum number of sessions).
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


ValueToInsertTuple: TypeAlias = tuple[
    str,  # queue_id
    str,  # session (as stringified JSON)
    str,  # session_id
    str,  # batch_id
    str | None,  # field_values (optional, as stringified JSON)
    int,  # priority
    str | None,  # workflow (optional, as stringified JSON)
    str | None,  # origin (optional)
    str | None,  # destination (optional)
    int | None,  # retried_from_item_id (optional, this is always None for new items)
]
"""A type alias for the tuple of values to insert into the session queue table.

**If you change this, be sure to update the `enqueue_batch` and `retry_items_by_id` methods in the session queue service!**
"""


def prepare_values_to_insert(
    queue_id: str, batch: Batch, priority: int, max_new_queue_items: int
) -> list[ValueToInsertTuple]:
    """
    Given a batch, prepare the values to insert into the session queue table. The list of tuples can be used with an
    `executemany` statement to insert multiple rows at once.

    Args:
        queue_id: The ID of the queue to insert the items into
        batch: The batch to prepare the values for
        priority: The priority of the queue items
        max_new_queue_items: The maximum number of queue items to insert

    Returns:
        A list of tuples to insert into the session queue table. Each tuple contains the following values:
        - queue_id
        - session (as stringified JSON)
        - session_id
        - batch_id
        - field_values (optional, as stringified JSON)
        - priority
        - workflow (optional, as stringified JSON)
        - origin (optional)
        - destination (optional)
        - retried_from_item_id (optional, this is always None for new items)
    """

    # A tuple is a fast and memory-efficient way to store the values to insert. Previously, we used a NamedTuple, but
    # measured a ~5% performance improvement by using a normal tuple instead. For very large batches (10k+ items), the
    # this difference becomes noticeable.
    #
    # So, despite the inferior DX with normal tuples, we use one here for performance reasons.

    values_to_insert: list[ValueToInsertTuple] = []

    # pydantic's to_jsonable_python handles serialization of any python object, including sets, which json.dumps does
    # not support by default. Apparently there are sets somewhere in the graph.

    # The same workflow is used for all sessions in the batch - serialize it once
    workflow_json = json.dumps(batch.workflow, default=to_jsonable_python) if batch.workflow else None

    for session_id, session_json, field_values_json in create_session_nfv_tuples(batch, max_new_queue_items):
        values_to_insert.append(
            (
                queue_id,
                session_json,
                session_id,
                batch.batch_id,
                field_values_json,
                priority,
                workflow_json,
                batch.origin,
                batch.destination,
                None,
            )
        )
    return values_to_insert


# endregion Util

Batch.model_rebuild(force=True)
SessionQueueItem.model_rebuild(force=True)
