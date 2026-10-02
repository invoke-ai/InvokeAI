import copy  # noqa: F401
import json
import weakref  # noqa: F401
from functools import wraps  # noqa: F401
from typing import (
    TYPE_CHECKING,  # noqa: F401
    Any,
    Callable,
    Concatenate,  # noqa: F401
    Deque,
    Iterable,
    Literal,
    Optional,
    ParamSpec,  # noqa: F401
    Type,
    TypeVar,  # noqa: F401
    Union,  # noqa: F401
    get_args,  # noqa: F401
    get_origin,  # noqa: F401
)

from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,  # noqa: F401
    GetJsonSchemaHandler,  # noqa: F401
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,  # noqa: F401
)
from pydantic.fields import Field
from pydantic.json_schema import JsonSchemaValue, SkipJsonSchema  # noqa: F401
from pydantic_core import PydanticSerializationError, core_schema  # noqa: F401

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationRegistry,  # noqa: F401
    invocation,  # noqa: F401
    invocation_output,  # noqa: F401
)
from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.collections import CollectionConcatInvocation
from invokeai.app.invocations.fields import Input, InputField, OutputField, OutputScope, UIType  # noqa: F401
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import (
    LOOP_LINKAGE_FIELD,
    ForInvocation,
    ForInvocationOutput,
    ForReturnInvocation,
    ForReturnInvocationOutput,
    LoopState,
)
from invokeai.app.services.shared import graph_if_dependencies, graph_if_runtime
from invokeai.app.services.shared.execution_effects import (
    AddEdgeEffect,
    AwaitEffect,
    CloseStreamEffect,
    ContinuationEffect,
    EmitEffect,
    ExecutionEffect,
    FailEffect,
    RemoveEdgeEffect,
    SetValueEffect,
    SpawnExecutionEffect,
)
from invokeai.app.services.shared.execution_effects import ExecutionRef as EffectExecutionRef
from invokeai.app.services.shared.execution_effects import (
    ExecutionToken as EffectExecutionToken,
)
from invokeai.app.services.shared.execution_engine.child import (
    ChildDependencyRecord,
    ChildDependencyUpdate,
    ChildExecutionCapability,
)
from invokeai.app.services.shared.execution_engine.primitives import (
    ActivationGate,
    ContinuationRecord,
    StreamData,
)
from invokeai.app.services.shared.execution_engine.primitives import (
    ExecutionFrame as EngineExecutionFrame,
)
from invokeai.app.services.shared.execution_engine.runtime import ExecutionEngineRuntime
from invokeai.app.services.shared.execution_engine.scheduler import (
    ActivationDependency,
)
from invokeai.app.services.shared.graph_execution_runtime import (
    _BodyIterateCollectFanIn,
    _DirectIterateCollectFanIn,
    _ExecutionRuntime,
)
from invokeai.app.services.shared.graph_for_planner import _GenericForPlanner
from invokeai.app.services.shared.graph_if_activation import _IfActivationController
from invokeai.app.services.shared.graph_iterate_planner import (
    _attach_direct_execution_edges,
    _can_use_direct_iterate_collect_planner,
    _create_direct_execution_node_copy,
    _get_body_iterate_collect_fan_in,
    _get_direct_iterate_collect_fan_in,
    _get_direct_iterate_collect_nodes,
    _initialize_direct_execution_node,
    _mark_direct_source_empty,
    _prepare_body_iterate_collect_fan_in_unchecked,
    _prepare_direct_iterate_collect,
    _prepare_direct_iterate_collect_fan_in_unchecked,
    _prepare_direct_iterate_collect_unchecked,
)
from invokeai.app.services.shared.graph_materializer import _ExecutionMaterializer
from invokeai.app.services.shared.graph_models import (
    Edge,  # noqa: F401
    EdgeConnection,  # noqa: F401
    ExecutionFrame,  # noqa: F401
    ExecutionRef,  # noqa: F401
    ExecutionReference,  # noqa: F401
    ExecutionToken,  # noqa: F401
    PreparedExecState,  # noqa: F401
    PreparedExecutionRef,  # noqa: F401
    WorkflowCallExecution,  # noqa: F401
    WorkflowCallFrame,  # noqa: F401
    WorkflowCallParentRef,  # noqa: F401
    WorkflowCallStatus,  # noqa: F401
)
from invokeai.app.services.shared.graph_nested_iterate_planner import (
    can_use_nested_iterate_planner,
    get_nested_iterate_sequence_depth,
    prepare_nested_iterate_bodies,
    prepare_nested_iterate_sequence,
)
from invokeai.app.services.shared.graph_runtime_records import (
    _ApplyTransaction,
    _PreparedExecNodeMetadata,
    _PreparedExecRegistry,
)
from invokeai.app.services.shared.graph_scheduler import _ExecutionScheduler, _GenericGraphSchedulerAdapter
from invokeai.app.services.shared.graph_validation import (
    _RESERVED_EFFECT_PORTS,
    COLLECTION_FIELD,
    ITEM_FIELD,
    AnyInvocation,  # noqa: F401
    AnyInvocationOutput,  # noqa: F401
    CollectInvocation,  # noqa: F401
    CollectInvocationOutput,  # noqa: F401
    CyclicalGraphError,  # noqa: F401
    DuplicateNodeIdError,  # noqa: F401
    Graph,
    InvalidEdgeError,  # noqa: F401
    IterateInvocation,  # noqa: F401
    IterateInvocationOutput,  # noqa: F401
    NodeAlreadyExecutedError,  # noqa: F401
    NodeAlreadyInGraphError,  # noqa: F401
    NodeFieldNotFoundError,  # noqa: F401
    NodeIdMismatchError,  # noqa: F401
    NodeInputError,  # noqa: F401
    NodeNotFoundError,  # noqa: F401
    NoneType,  # noqa: F401
    T,  # noqa: F401
    UnknownGraphValidationError,  # noqa: F401
    _EdgeList,  # noqa: F401
    _EdgeListMutationParams,  # noqa: F401
    _EdgeListMutationResult,  # noqa: F401
    _invalidates_edge_indexes,  # noqa: F401
    _LazyNetworkX,  # noqa: F401
    _SupportedNestedForBody,  # noqa: F401
    _SupportedNestedIterateBody,  # noqa: F401
    are_connection_types_compatible,  # noqa: F401
    are_connections_compatible,  # noqa: F401
    copydeep,
    extract_collection_item_types,  # noqa: F401
    get_input_field_type,  # noqa: F401
    get_output_field_scope,  # noqa: F401
    get_output_field_type,  # noqa: F401
    is_any,  # noqa: F401
    is_list_or_contains_list,  # noqa: F401
    is_union_subtype,  # noqa: F401
    loc_to_dot_sep,  # noqa: F401
    nx,
)
from invokeai.app.services.shared.invocation_context import InvocationContext  # noqa: F401
from invokeai.app.util.misc import uuid_string

_JSON_SERIALIZER = TypeAdapter(Any)

_EXECUTION_STATE_RUNTIME_FIELDS = frozenset(
    {"execution_refs", "execution_tokens", "execution_effects", "execution_child_dependencies"}
)
_EXECUTION_STATE_REQUIRED_FIELDS = (
    "id",
    "graph",
    "execution_graph",
    "executed",
    "executed_history",
    "results",
    "errors",
    "workflow_call_stack",
    "workflow_call_history",
    "prepared_source_mapping",
    "source_prepared_mapping",
)


def _hide_execution_state_runtime_fields(schema: JsonSchemaValue) -> None:
    """Keep private execution-ledger fields out of API schemas."""

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for field_name in _EXECUTION_STATE_RUNTIME_FIELDS:
            properties.pop(field_name, None)
        schema["required"] = [field_name for field_name in _EXECUTION_STATE_REQUIRED_FIELDS if field_name in properties]


class _GraphExecutionStateRuntime:
    """Process-local execution state kept behind one pydantic private attribute."""

    __slots__ = (
        "ready_queues",
        "ready_node_ids",
        "active_class",
        "resolved_if_exec_branches",
        "pending_if_exec_nodes",
        "if_branch_sources_cache",
        "if_activation_dependencies_by_source",
        "if_activation_dependencies_by_exec",
        "prepared_if_exec_ids_by_source_and_frame",
        "prepared_if_exec_frames_by_source",
        "prepared_if_exec_indexed_sources",
        "prepared_exec_metadata",
        "prepared_exec_registry",
        "if_activation_controller_instance",
        "execution_materializer",
        "generic_for_planner",
        "execution_scheduler",
        "execution_runtime",
        "for_parent_iteration_paths_cache",
        "all_for_contexts_finalized_cache",
        "prepared_for_index",
        "final_prepared_for_index",
        "prepared_for_index_by_exec",
        "source_graph_flat",
        "execution_graph_flat",
        "completed_source_ids_cache",
        "unexecuted_prepared_counts",
        "for_source_by_return_id",
        "apply_transaction",
        "generic_execution_runtime",
        "generic_graph_scheduler",
        "generic_child_dependencies",
        "pending_lifecycle_execution_nodes",
        "execution_effects_persisted",
        "legacy_execution_snapshot",
        "legacy_snapshot_loaded",
        "fresh_flat_if_activation",
    )

    def __init__(self) -> None:
        self.ready_queues: dict[str, Deque[str]] = {}
        self.ready_node_ids: set[str] = set()
        self.active_class: Optional[str] = None
        self.resolved_if_exec_branches: dict[str, str] = {}
        self.pending_if_exec_nodes: set[str] = set()
        self.if_branch_sources_cache: dict[tuple[str, str], frozenset[str]] = {}
        self.if_activation_dependencies_by_source: dict[
            tuple[str, tuple[int, ...]], tuple[ActivationDependency, ...]
        ] = {}
        self.if_activation_dependencies_by_exec: dict[str, tuple[ActivationDependency, ...]] = {}
        self.prepared_if_exec_ids_by_source_and_frame: dict[tuple[str, tuple[int, ...]], tuple[str, ...]] = {}
        self.prepared_if_exec_frames_by_source: dict[str, tuple[tuple[int, ...], ...]] = {}
        self.prepared_if_exec_indexed_sources: set[str] = set()
        self.prepared_exec_metadata: dict[str, _PreparedExecNodeMetadata] = {}
        self.prepared_exec_registry: Optional[_PreparedExecRegistry] = None
        self.if_activation_controller_instance: Optional[_IfActivationController] = None
        self.execution_materializer: Optional[_ExecutionMaterializer] = None
        self.generic_for_planner: Optional[_GenericForPlanner] = None
        self.execution_scheduler: Optional[_ExecutionScheduler | _GenericGraphSchedulerAdapter] = None
        self.execution_runtime: Optional[_ExecutionRuntime] = None
        self.for_parent_iteration_paths_cache: dict[str, set[tuple[int, ...]]] = {}
        self.all_for_contexts_finalized_cache: dict[str, bool] = {}
        self.prepared_for_index: Optional[dict[tuple[str, tuple[int, ...]], str]] = None
        self.final_prepared_for_index: Optional[dict[tuple[str, tuple[int, ...]], str]] = None
        self.prepared_for_index_by_exec: dict[str, tuple[str, tuple[int, ...], tuple[int, ...]]] = {}
        self.source_graph_flat: Any | None = None
        self.execution_graph_flat: Any | None = None
        self.completed_source_ids_cache: Optional[set[str]] = None
        self.unexecuted_prepared_counts: Optional[dict[str, int]] = None
        self.for_source_by_return_id: Optional[dict[str, str]] = None
        self.apply_transaction: Optional[_ApplyTransaction] = None
        self.generic_execution_runtime: Optional[ExecutionEngineRuntime] = None
        self.generic_graph_scheduler: Optional[_GenericGraphSchedulerAdapter] = None
        self.generic_child_dependencies: dict[str, ChildDependencyRecord] = {}
        self.pending_lifecycle_execution_nodes: set[str] = set()
        self.execution_effects_persisted = False
        self.legacy_execution_snapshot = True
        self.legacy_snapshot_loaded = False
        self.fresh_flat_if_activation: Optional[bool] = None


class GraphExecutionState(BaseModel):
    """Tracks source-graph expansion, execution progress, and runtime results."""

    id: str = Field(description="The id of the execution state", default_factory=uuid_string)
    # TODO: Store a reference to the graph instead of the actual graph?
    graph: Graph = Field(description="The graph being executed")

    # The graph of materialized nodes
    execution_graph: Graph = Field(
        description="The expanded graph of activated and executed nodes",
        default_factory=Graph,
    )

    # Nodes that have been executed
    executed: set[str] = Field(description="The set of node ids that have been executed", default_factory=set)
    executed_history: list[str] = Field(
        description="The list of node ids that have been executed, in order of execution",
        default_factory=list,
    )

    # The results of executed nodes
    results: dict[str, AnyInvocationOutput] = Field(description="The results of node executions", default_factory=dict)

    # Errors raised when executing nodes
    errors: dict[str, str] = Field(description="Errors raised when executing nodes", default_factory=dict)

    workflow_call_stack: list[WorkflowCallFrame] = Field(
        description="The nested workflow call stack inherited by this execution state.",
        default_factory=list,
    )
    workflow_call_history: list[WorkflowCallExecution] = Field(
        description="Completed or failed workflow-call relationships observed by this execution state.",
        default_factory=list,
    )
    workflow_call_parent: Optional[WorkflowCallParentRef] = Field(
        default=None,
        description="Parent workflow-call relationship metadata when this execution state is a child workflow session.",
    )
    waiting_workflow_call: Optional[WorkflowCallFrame] = Field(
        default=None,
        description="The child workflow call this execution state is currently waiting on, if any.",
    )
    waiting_workflow_call_execution: Optional[WorkflowCallExecution] = Field(
        default=None,
        description="The active workflow-call relationship metadata for the current waiting child workflow, if any.",
    )
    waiting_workflow_call_child_session: Optional["GraphExecutionState"] = Field(
        default=None,
        description="The child workflow execution state spawned by the current waiting workflow call, if any.",
    )
    max_workflow_call_depth: int = Field(
        default=4,
        ge=1,
        description="The maximum permitted workflow call depth for nested workflow execution.",
    )

    # Map of prepared/executed nodes to their original nodes
    prepared_source_mapping: dict[str, str] = Field(
        description="The map of prepared nodes to original graph nodes",
        default_factory=dict,
    )

    # Map of original nodes to prepared nodes
    source_prepared_mapping: dict[str, set[str]] = Field(
        description="The map of original graph nodes to prepared nodes",
        default_factory=dict,
    )
    finalized_loop_contexts: set[tuple[str, tuple[int, ...]]] = Field(
        description="The finalized loop source and parent iteration contexts",
        default_factory=set,
    )
    prepared_iteration_paths: dict[str, tuple[int, ...]] = Field(
        description="The iteration coordinates of each prepared execution node",
        default_factory=dict,
    )
    execution_refs: SkipJsonSchema[dict[str, ExecutionReference]] = Field(
        default_factory=dict,
        description="Stable frame-aware references for prepared execution nodes",
        exclude=True,
    )
    execution_tokens: SkipJsonSchema[dict[str, ExecutionToken]] = Field(
        default_factory=dict,
        description="Data tokens produced by prepared execution output ports",
        exclude=True,
    )
    execution_effects: SkipJsonSchema[dict[str, list[Any]]] = Field(
        default_factory=dict,
        description="Effects accepted for each execution reference",
        exclude=True,
    )
    execution_child_dependencies: SkipJsonSchema[dict[str, ChildDependencyRecord]] = Field(
        default_factory=dict,
        description="Durable generic child dependency records",
        exclude=True,
    )
    # All process-local runtime state is kept behind this single private attribute.
    _runtime_state: _GraphExecutionStateRuntime = PrivateAttr(default_factory=_GraphExecutionStateRuntime)
    # Optional priority; others follow in name order
    ready_order: list[str] = Field(default_factory=list)
    indegree: dict[str, int] = Field(default_factory=dict, description="Remaining unmet input count for exec nodes")

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "GraphExecutionState":
        copied = super().model_copy(update=update, deep=deep)
        if not deep:
            copied.__pydantic_private__["_runtime_state"] = copy.copy(self.__pydantic_private__["_runtime_state"])
        if update is not None and "graph" in update:
            copied._invalidate_source_graph_cache()
        return copied

    def __copy__(self) -> "GraphExecutionState":
        copied = super().__copy__()
        copied.__pydantic_private__["_runtime_state"] = copy.copy(self.__pydantic_private__["_runtime_state"])
        return copied

    def _tx_record_once(self, key: tuple[Any, ...], undo: Callable[[], None]) -> None:
        transaction = self.__pydantic_private__["_runtime_state"].apply_transaction
        if transaction is not None:
            transaction.record_once(key, undo)

    def _tx_record(self, undo: Callable[[], None]) -> None:
        transaction = self.__pydantic_private__["_runtime_state"].apply_transaction
        if transaction is not None:
            transaction.record(undo)

    def _tx_set_mapping(self, mapping: dict[Any, Any], key: Any, value: Any) -> None:
        transaction = self.__pydantic_private__["_runtime_state"].apply_transaction
        if transaction is not None:
            marker = ("mapping", id(mapping), key)
            if key in mapping:
                old_value = mapping[key]
                transaction.record_once(marker, lambda: mapping.__setitem__(key, old_value))
            else:
                transaction.record_once(marker, lambda: mapping.pop(key, None))
        mapping[key] = value

    def _tx_pop_mapping(self, mapping: dict[Any, Any], key: Any) -> None:
        if key not in mapping:
            return
        old_value = mapping[key]
        transaction = self.__pydantic_private__["_runtime_state"].apply_transaction
        if transaction is not None:
            transaction.record_once(("mapping", id(mapping), key), lambda: mapping.__setitem__(key, old_value))
        mapping.pop(key, None)

    def _tx_set_attr(self, obj: Any, name: str, value: Any) -> None:
        transaction = self.__pydantic_private__["_runtime_state"].apply_transaction
        if transaction is not None:
            old_value = getattr(obj, name)
            transaction.record_once(("attr", id(obj), name), lambda: setattr(obj, name, old_value))
        setattr(obj, name, value)

    def _tx_add_set(self, values: set[Any], value: Any) -> None:
        transaction = self.__pydantic_private__["_runtime_state"].apply_transaction
        if transaction is not None:
            existed = value in values
            transaction.record_once(
                ("set", id(values), value),
                lambda: None if existed else values.discard(value),
            )
        values.add(value)

    def _tx_discard_set(self, values: set[Any], value: Any) -> None:
        transaction = self.__pydantic_private__["_runtime_state"].apply_transaction
        if transaction is not None:
            existed = value in values
            transaction.record_once(
                ("set", id(values), value),
                lambda: values.add(value) if existed else None,
            )
        values.discard(value)

    def _tx_append_list(self, values: list[Any], value: Any) -> None:
        self._tx_record(lambda: values.pop())
        values.append(value)

    def _tx_add_execution_node(self, node: BaseInvocation) -> None:
        def undo() -> None:
            self.execution_graph.nodes.pop(node.id, None)
            self.execution_graph._invalidate_edge_indexes()

        self._tx_record(undo)
        self.execution_graph.add_node(node)

    def _tx_add_execution_edges(self, edges: list[Edge]) -> None:
        for edge in edges:
            self._tx_record(lambda edge=edge: self.execution_graph.delete_edge(edge))
        self.execution_graph._extend_edges_unchecked(edges)

    def _tx_delete_execution_edge(self, edge: Edge) -> None:
        if edge not in self.execution_graph.edges:
            return
        edge_index = self.execution_graph.edges.index(edge)
        self._tx_record(lambda: self.execution_graph.edges.insert(edge_index, edge))
        self.execution_graph.delete_edge(edge)

    def _tx_queue_append(self, queue: Deque[str], value: str) -> None:
        self._tx_record(lambda: queue.pop())
        queue.append(value)

    def _tx_queue_insert(self, queue: Deque[str], index: int, value: str) -> None:
        self._tx_record(lambda: queue.remove(value))
        queue.insert(index, value)

    def _tx_queue_remove(self, queue: Deque[str], value: str) -> None:
        index = queue.index(value)
        self._tx_record(lambda: queue.insert(index, value))
        queue.remove(value)

    def _reset_apply_derived_caches(self) -> None:
        self._prepared_exec_registry = None
        self._execution_materializer = None
        self._generic_for_planner = None
        self._execution_scheduler = None
        self._generic_graph_scheduler = None
        self._if_activation_controller_instance = None
        self._pending_if_exec_nodes = set()
        self._if_branch_sources_cache = {}
        self._if_activation_dependencies_by_source = {}
        self._if_activation_dependencies_by_exec = {}
        self._prepared_if_exec_ids_by_source_and_frame = {}
        self._prepared_if_exec_frames_by_source = {}
        self._prepared_if_exec_indexed_sources = set()
        self._execution_runtime = None
        self._source_graph_flat = None
        self._execution_graph_flat = None
        self._completed_source_ids_cache = None
        self._unexecuted_prepared_counts = None
        self._for_source_by_return_id = None
        self._for_parent_iteration_paths_cache = {}
        self._all_for_contexts_finalized_cache = {}
        self._prepared_for_index = None
        self._final_prepared_for_index = None
        self._prepared_for_index_by_exec = {}
        self._prepared_if_exec_ids_by_source_and_frame = {}
        self._prepared_if_exec_frames_by_source = {}
        self._prepared_if_exec_indexed_sources = set()
        self._pending_if_exec_nodes = {
            exec_node_id
            for exec_node_id, node in self.execution_graph.nodes.items()
            if exec_node_id not in self.executed
            and isinstance(node, IfInvocation)
            and self._is_pending_if(exec_node_id)
        }

    def _type_key(self, node_obj: BaseInvocation) -> str:
        return node_obj.__class__.__name__

    def _prepared_registry(self) -> _PreparedExecRegistry:
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.prepared_exec_registry is None:
            runtime.prepared_exec_registry = _PreparedExecRegistry(
                prepared_source_mapping=self.prepared_source_mapping,
                source_prepared_mapping=self.source_prepared_mapping,
                prepared_iteration_paths=self.prepared_iteration_paths,
                metadata=runtime.prepared_exec_metadata,
                on_iteration_path_change=self._invalidate_loop_caches_for_exec_node,
                state=self,
            )
        return runtime.prepared_exec_registry

    def _prepared_if_exec_ids(self, source_node_id: str, iteration_path: tuple[int, ...]) -> tuple[str, ...]:
        """Return prepared If executions for one source/frame without rescanning them per dependency check."""
        if source_node_id not in self._prepared_if_exec_indexed_sources:
            frame_paths: list[tuple[int, ...]] = []
            for exec_node_id in self.source_prepared_mapping.get(source_node_id, ()):
                if not isinstance(self.execution_graph.nodes.get(exec_node_id), IfInvocation):
                    continue
                prepared_path = self._get_iteration_path(exec_node_id)
                if prepared_path is None:
                    continue
                key = (source_node_id, prepared_path)
                existing = self._prepared_if_exec_ids_by_source_and_frame.get(key, ())
                self._prepared_if_exec_ids_by_source_and_frame[key] = (*existing, exec_node_id)
                if not existing:
                    frame_paths.append(prepared_path)
            self._prepared_if_exec_frames_by_source[source_node_id] = tuple(frame_paths)
            self._prepared_if_exec_indexed_sources.add(source_node_id)
        return self._prepared_if_exec_ids_by_source_and_frame.get((source_node_id, iteration_path), ())

    def _prepared_if_exec_frames(self, source_node_id: str) -> tuple[tuple[int, ...], ...]:
        """Return indexed If frames for one source."""
        self._prepared_if_exec_ids(source_node_id, ())
        return self._prepared_if_exec_frames_by_source.get(source_node_id, ())

    def _if_activation_controller(self) -> _IfActivationController:
        if self._if_activation_controller_instance is None:
            self._if_activation_controller_instance = _IfActivationController(self)
        return self._if_activation_controller_instance

    def _get_source_graph_flat(self) -> Any:
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.source_graph_flat is None:
            runtime.source_graph_flat = self.graph.nx_graph_flat()
        return runtime.source_graph_flat

    def _get_fresh_if_nodes(self) -> tuple[IfInvocation, ...]:
        return graph_if_dependencies._get_fresh_if_nodes(self)

    def _can_use_fresh_flat_if_activation(self) -> bool:
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.fresh_flat_if_activation is None:
            runtime.fresh_flat_if_activation = graph_if_dependencies._can_use_fresh_flat_if_activation(self)
        return runtime.fresh_flat_if_activation

    def _can_use_fresh_mixed_if_iterate_collect(self) -> bool:
        return graph_if_dependencies._can_use_fresh_mixed_if_iterate_collect(self)

    def _get_fresh_if_branch_sources(self, if_node_id: str, branch_field: str) -> set[str]:
        return graph_if_dependencies._get_fresh_if_branch_sources(self, if_node_id, branch_field)

    def _get_source_activation_dependencies(
        self, source_node_id: str, iteration_path: tuple[int, ...] = ()
    ) -> tuple[ActivationDependency, ...]:
        return graph_if_dependencies._get_source_activation_dependencies(self, source_node_id, iteration_path)

    def _record_activation_dependencies(self, exec_node_id: str) -> tuple[ActivationDependency, ...]:
        return graph_if_runtime._record_activation_dependencies(self, exec_node_id)

    def _is_source_activation_admitted(self, source_node_id: str, iteration_path: tuple[int, ...] = ()) -> bool:
        return graph_if_runtime._is_source_activation_admitted(self, source_node_id, iteration_path)

    def _is_source_inactive(self, source_node_id: str, iteration_path: tuple[int, ...] = ()) -> bool:
        return graph_if_runtime._is_source_inactive(self, source_node_id, iteration_path)

    def _get_execution_graph_flat(self) -> Any:
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.execution_graph_flat is None:
            runtime.execution_graph_flat = self.execution_graph.nx_graph_flat()
        return runtime.execution_graph_flat

    def _add_execution_graph_node(self, node_id: str) -> None:
        execution_graph_flat = self.__pydantic_private__["_runtime_state"].execution_graph_flat
        if execution_graph_flat is not None:
            execution_graph_flat.add_node(node_id)

    def _add_execution_graph_edges(self, edges: Iterable[Edge]) -> None:
        execution_graph_flat = self.__pydantic_private__["_runtime_state"].execution_graph_flat
        if execution_graph_flat is not None:
            execution_graph_flat.add_edges_from(
                (edge.source.node_id, edge.destination.node_id) for edge in edges if edge.type == "default"
            )

    def _mark_source_executed(self, source_node_id: str) -> None:
        self._tx_add_set(self.executed, source_node_id)
        self.__pydantic_private__["_runtime_state"].completed_source_ids_cache = None
        self._tx_add_set(self._get_completed_source_ids_cache(), source_node_id)
        if source_node_id not in self.executed_history:
            self._tx_append_list(self.executed_history, source_node_id)

    def _discard_source_executed(self, source_node_id: str) -> None:
        self._tx_discard_set(self.executed, source_node_id)
        # A source can be discarded while its already-prepared executions are still complete. New executions
        # invalidate the derived completion cache when they are registered.

    def _mark_exec_node_executed(self, exec_node_id: str) -> None:
        """Mark prepared exec node executed and maintain per-source pending count."""
        if exec_node_id in self.executed:
            return
        self._tx_add_set(self.executed, exec_node_id)
        counts = self.__pydantic_private__["_runtime_state"].unexecuted_prepared_counts
        if counts is None:
            return
        source_node_id = self.prepared_source_mapping.get(exec_node_id)
        if source_node_id is None:
            return
        # Empty loop contexts can drop reverse mappings while retaining forward mappings.
        if exec_node_id not in self.source_prepared_mapping.get(source_node_id, ()):
            return
        old_count = counts.get(source_node_id, 0)
        self._tx_record_once(
            ("unexecuted_prepared_count", source_node_id),
            lambda: counts.__setitem__(source_node_id, old_count),
        )
        counts[source_node_id] = max(old_count - 1, 0)

    def _reset_unexecuted_prepared(self, source_node_id: str) -> None:
        """Reset pending count after dropping prepared nodes for a source."""
        if self._unexecuted_prepared_counts is None:
            return
        counts = self._unexecuted_prepared_counts
        old_count = counts.get(source_node_id, 0)
        self._tx_record_once(
            ("unexecuted_prepared_count", source_node_id),
            lambda: counts.__setitem__(source_node_id, old_count),
        )
        counts[source_node_id] = 0

    def _count_unexecuted_prepared(self, source_node_id: str) -> int:
        """Return prepared exec nodes not yet executed or skipped for source."""
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.unexecuted_prepared_counts is None:
            runtime.unexecuted_prepared_counts = {
                mapped_source_id: sum(1 for exec_node_id in prepared_ids if exec_node_id not in self.executed)
                for mapped_source_id, prepared_ids in self.source_prepared_mapping.items()
            }
        return runtime.unexecuted_prepared_counts.get(source_node_id, 0)

    def _get_completed_source_ids_cache(self) -> set[str]:
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.completed_source_ids_cache is None:
            runtime.completed_source_ids_cache = {
                source_node_id
                for source_node_id in self.graph.nodes
                if source_node_id in self.executed
                or (
                    self.source_prepared_mapping.get(source_node_id)
                    and self._count_unexecuted_prepared(source_node_id) == 0
                )
                or self._is_source_inactive(source_node_id)
            }
        return runtime.completed_source_ids_cache

    def _get_for_source_by_return_id(self) -> dict[str, str]:
        if self._for_source_by_return_id is None:
            source_graph = self._get_source_graph_flat()
            self._for_source_by_return_id = {
                body_path_to_return[1]: source_for_id
                for source_for_id, source_for_node in self.graph.nodes.items()
                if isinstance(source_for_node, ForInvocation)
                and (body_path_to_return := self.graph._get_for_body_path_to_return(source_for_id, source_graph))
                is not None
            }
        return self._for_source_by_return_id

    def _invalidate_source_graph_cache(self) -> None:
        runtime = self.__pydantic_private__["_runtime_state"]
        runtime.source_graph_flat = None
        runtime.fresh_flat_if_activation = None
        runtime.for_source_by_return_id = None
        runtime.if_branch_sources_cache = {}
        runtime.if_activation_dependencies_by_source = {}
        runtime.if_activation_dependencies_by_exec = {}
        runtime.if_activation_controller_instance = None

    def _materializer(self) -> _ExecutionMaterializer:
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.execution_materializer is None:
            runtime.execution_materializer = _ExecutionMaterializer(self)
        return runtime.execution_materializer

    def _for_planner(self) -> _GenericForPlanner:
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.generic_for_planner is None:
            runtime.generic_for_planner = _GenericForPlanner(self)
        return runtime.generic_for_planner

    def _create_for_body_iteration(self, source_for_id: str, prepared_for_id: str) -> Optional[str]:
        if self._is_generic_graph_scheduler(self._scheduler()) and self._can_use_generic_for_scheduler():
            return self._for_planner().create_for_body_iteration(
                source_for_id=source_for_id,
                prepared_for_id=prepared_for_id,
            )
        return self._materializer().create_for_body_iteration(
            source_for_id=source_for_id,
            prepared_for_id=prepared_for_id,
        )

    def _get_for_parent(self, exec_node_id: str) -> Optional[str]:
        source_return_id = self._prepared_registry().get_source_node_id(exec_node_id)
        iteration_path = self._get_iteration_path(exec_node_id)
        source_for_id = self._get_for_source_by_return_id().get(source_return_id)
        if source_for_id is not None:
            prepared_for_id = self._get_prepared_for_index().get((source_for_id, iteration_path))
            if prepared_for_id is not None:
                return prepared_for_id

        # The indexed source/path lookup is the normal path. The ancestor fallback only covers legacy execution
        # graphs whose durable linkage was not materialized; keep it explicit so an unexpected miss is diagnosable.
        execution_graph = self._get_execution_graph_flat()
        for ancestor_id in nx.ancestors(execution_graph, exec_node_id):
            source_node = self.execution_graph.get_node(ancestor_id)
            if isinstance(source_node, ForInvocation):
                return ancestor_id

        # An empty nested Iterate has no item execution node, so its synthetic Collect and ForReturn have no
        # execution-graph edge back to their owning For. The same indexed lookup handles this case when the
        # synthetic node has been assigned its durable parent path.
        if source_for_id is not None:
            for prepared_for_id in self._prepared_registry().get_prepared_ids(source_for_id):
                prepared_for_node = self.execution_graph.get_node(prepared_for_id)
                if not isinstance(prepared_for_node, ForInvocation) or prepared_for_node.index < 0:
                    continue
                if self._get_iteration_path(prepared_for_id) == iteration_path:
                    return prepared_for_id
        return None

    def _get_loop_state_for_next_iteration(
        self, for_exec_node_id: str, return_output: ForReturnInvocationOutput
    ) -> LoopState:
        if return_output.state is not None:
            return return_output.state

        for_output = self.results.get(for_exec_node_id)
        if isinstance(for_output, ForInvocationOutput):
            return for_output.state

        return LoopState()

    def _get_ordered_for_return_outputs(
        self, for_exec_node_id: str, source_return_id: str
    ) -> list[ForReturnInvocationOutput]:
        parent_iteration_path = self._get_for_parent_iteration_path(for_exec_node_id)
        prepared_return_ids = self._prepared_registry().get_prepared_ids(source_return_id)
        prepared_return_ids = [
            prepared_return_id
            for prepared_return_id in prepared_return_ids
            if self._get_iteration_path(prepared_return_id)[:-1] == parent_iteration_path
        ]
        prepared_return_ids = sorted(prepared_return_ids, key=self._get_iteration_path)
        return [
            output
            for prepared_return_id in prepared_return_ids
            if isinstance((output := self.results.get(prepared_return_id)), ForReturnInvocationOutput)
        ]

    def _finalize_for_outputs(
        self,
        for_exec_node_id: str,
        source_for_id: str,
        source_return_id: str,
        return_output: ForReturnInvocationOutput,
    ) -> None:
        for_output = self.results.get(for_exec_node_id)
        if not isinstance(for_output, ForInvocationOutput):
            return

        return_outputs = self._get_ordered_for_return_outputs(for_exec_node_id, source_return_id)
        self._tx_set_attr(for_output, "output_collection", [output.output for output in return_outputs])
        self._tx_set_attr(
            for_output,
            "final_state",
            self._get_loop_state_for_next_iteration(for_exec_node_id, return_output),
        )
        self._mark_loop_context_finalized(source_for_id, for_exec_node_id)
        self._refresh_output_tokens(for_exec_node_id)

    def _refresh_output_tokens(self, exec_node_id: str) -> None:
        output = self.results.get(exec_node_id)
        execution_ref = self.execution_refs.get(exec_node_id)
        if output is None or execution_ref is None:
            return
        for token_id, token in self._build_execution_tokens(execution_ref, output).items():
            existing = self.execution_tokens.get(token_id)
            if existing is None:
                self._tx_set_mapping(self.execution_tokens, token_id, token)
            else:
                self._tx_set_attr(existing, "value", token.value)

    def _apply_for_continuation(
        self,
        exec_node_id: str,
        output: BaseInvocationOutput,
        planner: _GenericForPlanner | _ExecutionMaterializer,
    ) -> Optional[str]:
        if not isinstance(output, ForReturnInvocationOutput):
            return None
        if not isinstance(self.execution_graph.get_node(exec_node_id), ForReturnInvocation):
            return None

        for_exec_node_id = self._get_for_parent(exec_node_id)
        if for_exec_node_id is None:
            return None

        for_node = self.execution_graph.get_node(for_exec_node_id)
        if not isinstance(for_node, ForInvocation):
            return None
        for_return_node = self.execution_graph.get_node(exec_node_id)
        assert isinstance(for_return_node, ForReturnInvocation)

        self._complete_for_continuation(
            for_exec_node_id,
            self._for_return_continuation_payload(for_return_node, output),
        )

        registry = self._prepared_registry()
        source_for_id = registry.get_source_node_id(for_exec_node_id)
        source_return_id = registry.get_source_node_id(exec_node_id)

        next_index = for_node.index + 1
        if next_index >= len(for_node.collection) or for_return_node.continue_condition is False:
            self._finalize_for_outputs(for_exec_node_id, source_for_id, source_return_id, output)
            planner.create_nested_for_return(
                inner_for_id=source_for_id,
                prepared_inner_for_id=for_exec_node_id,
            )
            self._tx_set_attr(for_node, "collection", [])
            return for_exec_node_id

        next_state = self._get_loop_state_for_next_iteration(for_exec_node_id, output)
        parent_iteration_path = self._get_for_parent_iteration_path(for_exec_node_id)
        collection = for_node.collection
        self._tx_set_attr(for_node, "collection", [])

        next_for_id = planner.create_for_iteration(
            source_for_id=source_for_id,
            iteration_index=next_index,
            collection=collection,
            state=next_state,
            iteration_path=(*parent_iteration_path, next_index),
        )
        self._discard_source_executed(source_for_id)
        if self._is_generic_graph_scheduler(self._scheduler()) and can_use_nested_iterate_planner(self):
            prepare_nested_iterate_bodies(self)
        else:
            planner.create_for_body_iteration(source_for_id=source_for_id, prepared_for_id=next_for_id)
        return None

    def _apply_generic_for_continuation(self, exec_node_id: str, output: BaseInvocationOutput) -> Optional[str]:
        """Apply graph-state continuation boundary for an admitted fresh For run."""

        return self._apply_for_continuation(exec_node_id, output, self._for_planner())

    def _try_schedule_next_for_iteration(self, exec_node_id: str, output: BaseInvocationOutput) -> Optional[str]:
        """Advance a compatibility-owned loop continuation."""

        return self._apply_for_continuation(exec_node_id, output, self._materializer())

    def _can_use_generic_two_sibling_nested_for_scheduler(
        self, for_nodes: list[ForInvocation], return_nodes: list[ForReturnInvocation], source_graph: Any
    ) -> bool:
        if len(for_nodes) != 3 or len(return_nodes) != 3 or len(self.graph.nodes) != 10 or len(self.graph.edges) != 13:
            return False

        outer_candidates = []
        nested_bodies = {}
        for node in for_nodes:
            nested_body = self.graph._get_supported_for_nested_for_body(node.id, source_graph)
            if nested_body is not None:
                nested_bodies[node.id] = nested_body
                if len(nested_body.inner_for_ids) == 2 and len(nested_body.continuation_nodes) == 1:
                    outer_candidates.append(node)
        if len(outer_candidates) != 1:
            return False

        outer_for = outer_candidates[0]
        nested_body = nested_bodies[outer_for.id]
        child_ids = set(nested_body.inner_for_ids)
        if self.graph._get_input_edges(outer_for.id, COLLECTION_FIELD) or not outer_for.collection:
            return False
        outer_return_id = nested_body.outer_return_id
        child_return_ids = {self.graph._get_linked_for_return_id(child_id) for child_id in child_ids}
        if None in child_return_ids or child_return_ids | {outer_return_id} != {node.id for node in return_nodes}:
            return False

        join_id = next(iter(nested_body.continuation_nodes))
        join = self.graph.get_node(join_id)
        if not isinstance(join, CollectionConcatInvocation):
            return False
        join_inputs = self.graph._get_input_edges(join_id)
        if len(join_inputs) != 2 or {edge.destination.field for edge in join_inputs} != {"first", "second"}:
            return False
        if {edge.source.node_id for edge in join_inputs} != child_ids or any(
            edge.source.field != "output_collection" for edge in join_inputs
        ):
            return False

        outer_return_inputs = self.graph._get_input_edges(outer_return_id, "output")
        if len(outer_return_inputs) != 1 or (
            outer_return_inputs[0].source.node_id != join_id or outer_return_inputs[0].source.field != "collection"
        ):
            return False
        if self.graph._get_input_edges(outer_return_id) != outer_return_inputs:
            return False
        if self.graph._get_linked_for_return_id(outer_for.id) != outer_return_id:
            return False

        control_types = (ForInvocation, ForReturnInvocation, IterateInvocation, CollectInvocation, IfInvocation)
        outer_item_edges = self.graph._get_output_edges(outer_for.id, ITEM_FIELD)
        if len(outer_item_edges) != 2:
            return False
        body_ids: set[str] = set()

        for child_id in child_ids:
            child_collection_edges = self.graph._get_input_edges(child_id, COLLECTION_FIELD)
            if len(child_collection_edges) != 1 or self.graph._get_input_edges(child_id) != child_collection_edges:
                return False
            child_collection_edge = child_collection_edges[0]
            if child_collection_edge.source.node_id != outer_for.id or child_collection_edge.source.field != ITEM_FIELD:
                return False
            if child_collection_edges[0] not in outer_item_edges:
                return False

            child_return_id = self.graph._get_linked_for_return_id(child_id)
            if child_return_id is None:
                return False
            child_body_path = self.graph._get_for_body_path_to_return(child_id, source_graph)
            if child_body_path is None or len(child_body_path[0]) != 2 or child_body_path[1] != child_return_id:
                return False
            child_body_ids = child_body_path[0] - {child_return_id}
            if len(child_body_ids) != 1:
                return False
            body_id = next(iter(child_body_ids))
            body_ids.add(body_id)
            body = self.graph.get_node(body_id)
            if isinstance(body, control_types):
                return False
            body_inputs = self.graph._get_input_edges(body_id)
            body_outputs = self.graph._get_output_edges(body_id)
            if len(body_inputs) != 1 or len(body_outputs) != 1:
                return False
            if body_inputs[0].source.node_id != child_id or body_inputs[0].source.field != ITEM_FIELD:
                return False
            if self.graph._get_output_edges(child_id, ITEM_FIELD) != body_inputs:
                return False
            if body_outputs[0].destination.node_id != child_return_id or body_outputs[0].destination.field != "output":
                return False
            if self.graph._get_input_edges(child_return_id) != body_outputs:
                return False
            if self.graph._get_for_final_output_edges(child_id) != [
                edge for edge in self.graph._get_output_edges(child_id) if edge.source.field == "output_collection"
            ]:
                return False

        outer_final_edges = self.graph._get_for_final_output_edges(outer_for.id)
        if len(outer_final_edges) != 1 or outer_final_edges[0].source.field != "output_collection":
            return False
        after_id = outer_final_edges[0].destination.node_id
        after = self.graph.get_node(after_id)
        if isinstance(after, control_types) or outer_final_edges[0].destination.field != "value":
            return False
        if self.graph._get_input_edges(after_id) != outer_final_edges:
            return False
        expected_nodes = {outer_for.id, *child_ids, outer_return_id, join_id, after_id, *child_return_ids, *body_ids}
        return set(self.graph.nodes) == expected_nodes

    def _can_use_generic_serial_nested_for_scheduler(
        self,
        for_nodes: list[ForInvocation],
        return_nodes: list[ForReturnInvocation],
        source_graph: Any,
        *,
        nested_level: Literal[3, 4] = 3,
    ) -> bool:
        if len(for_nodes) != nested_level or len(return_nodes) != nested_level:
            return False
        if nested_level == 4 and (len(self.graph.nodes) != 10 or len(self.graph.edges) != 13):
            return False

        outer_candidates: list[ForInvocation] = []
        for_node_bodies: dict[str, Any] = {}
        for node in for_nodes:
            nested_body = self.graph._get_supported_for_nested_for_body(node.id, source_graph)
            if nested_body is not None:
                for_node_bodies[node.id] = nested_body
        nested_child_ids = {
            inner_for_id for nested_body in for_node_bodies.values() for inner_for_id in nested_body.inner_for_ids
        }
        outer_candidates = [
            node
            for node in for_nodes
            if node.id not in nested_child_ids
            and (nested_body := for_node_bodies.get(node.id)) is not None
            and len(nested_body.inner_for_ids) == 1
            and not nested_body.continuation_nodes
        ]
        if len(outer_candidates) != 1:
            return False

        outer_for = outer_candidates[0]
        outer_collection_edges = self.graph._get_input_edges(outer_for.id, COLLECTION_FIELD)
        if not outer_collection_edges and not outer_for.collection:
            return False
        if outer_collection_edges:
            if nested_level != 3 or len(outer_collection_edges) != 1 or outer_for.collection:
                return False
            outer_collection_edge = outer_collection_edges[0]
            if outer_collection_edge.destination.field != COLLECTION_FIELD:
                return False
            producer = self.graph.get_node(outer_collection_edge.source.node_id)
            if not isinstance(producer, CollectionConcatInvocation) or (not producer.first and not producer.second):
                return False
            if (
                outer_collection_edge.source.field != "collection"
                or self.graph._get_input_edges(producer.id)
                or self.graph._get_output_edges(producer.id) != outer_collection_edges
                or len(self.graph.nodes) != 9
                or len(self.graph.edges) != 11
            ):
                return False

        chain = [outer_for]
        while len(chain) < nested_level:
            parent_for = chain[-1]
            nested_body = for_node_bodies.get(parent_for.id)
            if nested_body is None or len(nested_body.inner_for_ids) != 1 or nested_body.continuation_nodes:
                return False
            child_for = self.graph.get_node(nested_body.inner_for_ids[0])
            if not isinstance(child_for, ForInvocation):
                return False
            collection_edges = self.graph._get_input_edges(child_for.id, COLLECTION_FIELD)
            if len(collection_edges) != 1 or (
                collection_edges[0].source.node_id != parent_for.id
                or collection_edges[0].source.field != ITEM_FIELD
                or self.graph._get_output_edges(parent_for.id, ITEM_FIELD) != collection_edges
            ):
                return False
            chain.append(child_for)

        if len(chain) != nested_level or {node.id for node in chain} != {node.id for node in for_nodes}:
            return False
        if chain[-1].id in for_node_bodies:
            return False

        linked_return_ids = {self.graph._get_linked_for_return_id(node.id) for node in chain}
        if None in linked_return_ids or linked_return_ids != {node.id for node in return_nodes}:
            return False

        final_output_edges = self.graph._get_for_final_output_edges(outer_for.id)
        if nested_level == 4 and len(final_output_edges) != 1:
            return False
        if nested_level != 4 and len(final_output_edges) > 1:
            return False
        if final_output_edges:
            final_consumer = self.graph.get_node(final_output_edges[0].destination.node_id)
            if isinstance(final_consumer, (ForInvocation, ForReturnInvocation, IterateInvocation, CollectInvocation)):
                return False
        return True

    def _can_use_generic_for_scheduler(self) -> bool:
        """Allow generic routing only for the explicitly supported fresh For shapes."""

        if self._legacy_snapshot_loaded:
            return False
        if any(isinstance(node, (IfInvocation, CallSavedWorkflowInvocation)) for node in self.graph.nodes.values()):
            return False
        for_nodes = [node for node in self.graph.nodes.values() if isinstance(node, ForInvocation)]
        return_nodes = [node for node in self.graph.nodes.values() if isinstance(node, ForReturnInvocation)]
        if any(isinstance(node, (IterateInvocation, CollectInvocation)) for node in self.graph.nodes.values()):
            return self._can_use_generic_nested_iterate_scheduler(for_nodes, return_nodes)
        source_graph = self._get_source_graph_flat()
        if len(for_nodes) == 3 and len(return_nodes) == 3:
            if self._can_use_generic_two_sibling_nested_for_scheduler(for_nodes, return_nodes, source_graph):
                return True
            return self._can_use_generic_serial_nested_for_scheduler(for_nodes, return_nodes, source_graph)
        if len(for_nodes) == 4 and len(return_nodes) == 4:
            return self._can_use_generic_serial_nested_for_scheduler(
                for_nodes,
                return_nodes,
                source_graph,
                nested_level=4,
            )
        if len(for_nodes) == 2 and len(return_nodes) == 2:
            outer_for = next(
                (
                    node
                    for node in for_nodes
                    if (nested_body := self.graph._get_supported_for_nested_for_body(node.id, source_graph)) is not None
                    and len(nested_body.inner_for_ids) == 1
                    and not nested_body.continuation_nodes
                ),
                None,
            )
            if outer_for is None or (
                not self.graph._get_input_edges(outer_for.id, COLLECTION_FIELD) and not outer_for.collection
            ):
                return False
            nested_body = self.graph._get_supported_for_nested_for_body(outer_for.id, source_graph)
            assert nested_body is not None
            outer_collection_edges = self.graph._get_input_edges(outer_for.id, COLLECTION_FIELD)
            if outer_collection_edges:
                if len(outer_collection_edges) != 1 or outer_for.collection:
                    return False
                outer_collection_edge = outer_collection_edges[0]
                if outer_collection_edge.destination.field != COLLECTION_FIELD:
                    return False
                producer = self.graph.get_node(outer_collection_edge.source.node_id)
                if not isinstance(producer, CollectionConcatInvocation) or (not producer.first and not producer.second):
                    return False
                if (
                    outer_collection_edge.source.field != "collection"
                    or self.graph._get_input_edges(producer.id)
                    or self.graph._get_output_edges(producer.id) != outer_collection_edges
                ):
                    return False
                inner_for_id = nested_body.inner_for_ids[0]
                inner_return_id = self.graph._get_linked_for_return_id(inner_for_id)
                final_output_edges = self.graph._get_for_final_output_edges(outer_for.id)
                if inner_return_id is None or len(final_output_edges) != 1:
                    return False
                final_consumer = self.graph.get_node(final_output_edges[0].destination.node_id)
                if (
                    final_output_edges[0].source.field != "output_collection"
                    or final_consumer.id in {outer_for.id, inner_for_id, inner_return_id, nested_body.outer_return_id}
                    or isinstance(
                        final_consumer, (ForInvocation, ForReturnInvocation, IterateInvocation, CollectInvocation)
                    )
                    or self.graph._get_input_edges(final_consumer.id) != final_output_edges
                    or len(self.graph.nodes) != 7
                    or len(self.graph.edges) != 8
                ):
                    return False
            inner_for = self.graph.get_node(nested_body.inner_for_ids[0])
            if not isinstance(inner_for, ForInvocation):
                return False
            inner_collection_edges = self.graph._get_input_edges(inner_for.id, COLLECTION_FIELD)
            if len(inner_collection_edges) != 1 or (
                inner_collection_edges[0].source.node_id != outer_for.id
                or inner_collection_edges[0].source.field != ITEM_FIELD
            ):
                return False
            return True
        if len(for_nodes) != 1 or len(return_nodes) != 1:
            return False
        collection_edges = self.graph._get_input_edges(for_nodes[0].id, COLLECTION_FIELD)
        empty_collection_source: BaseInvocation | None = None
        if collection_edges:
            collection_source = self.graph.get_node(collection_edges[0].source.node_id)
            collection_value = getattr(collection_source, "value", None)
            if (
                collection_edges[0].source.field == "value"
                and isinstance(collection_value, list)
                and not collection_value
            ):
                empty_collection_source = collection_source
        body_path = self.graph._get_for_body_path_to_return(for_nodes[0].id, self._get_source_graph_flat())
        if body_path is None:
            return False
        body_node_ids, return_node_id = body_path
        if empty_collection_source is not None:
            body_node_ids_without_return = body_node_ids - {return_node_id}
            if len(body_node_ids_without_return) != 1:
                return False
            body_node_id = next(iter(body_node_ids_without_return))
            body_node = self.graph.get_node(body_node_id)
            control_node_types = (
                ForInvocation,
                ForReturnInvocation,
                IterateInvocation,
                CollectInvocation,
                IfInvocation,
                CallSavedWorkflowInvocation,
            )
            if (
                len(self.graph.nodes) != 4
                or len(self.graph.edges) != 4
                or set(self.graph.nodes)
                != {empty_collection_source.id, for_nodes[0].id, body_node_id, return_nodes[0].id}
                or isinstance(empty_collection_source, control_node_types)
                or isinstance(body_node, control_node_types)
                or len(collection_edges) != 1
                or self.graph._get_input_edges(empty_collection_source.id)
                or self.graph._get_output_edges(empty_collection_source.id) != collection_edges
                or self.graph._get_input_edges(for_nodes[0].id) != collection_edges
                or collection_edges[0].source.field != "value"
                or collection_edges[0].destination.field != COLLECTION_FIELD
                or self.graph._get_output_edges(for_nodes[0].id, ITEM_FIELD)
                != [edge for edge in self.graph._get_input_edges(body_node_id) if edge.source.field == ITEM_FIELD]
                or len(self.graph._get_input_edges(body_node_id)) != 1
                or self.graph._get_input_edges(body_node_id)[0].source.node_id != for_nodes[0].id
                or self.graph._get_input_edges(body_node_id)[0].source.field != ITEM_FIELD
                or self.graph._get_output_edges(body_node_id, "value")
                != self.graph._get_input_edges(return_nodes[0].id, "output")
                or len(self.graph._get_input_edges(return_nodes[0].id, "output")) != 1
                or len(self.graph._get_input_edges(return_nodes[0].id, "loop_linkage", include_loop_linkage=True)) != 1
                or self.graph._get_input_edges(return_nodes[0].id, "loop_linkage", include_loop_linkage=True)[
                    0
                ].source.node_id
                != for_nodes[0].id
                or self.graph._get_input_edges(return_nodes[0].id, "loop_linkage", include_loop_linkage=True)[
                    0
                ].source.field
                != "loop_linkage"
                or self.graph._get_for_final_output_edges(for_nodes[0].id)
            ):
                return False
        return return_node_id == return_nodes[0].id and not any(
            isinstance(self.graph.get_node(node_id), (ForInvocation, ForReturnInvocation))
            for node_id in body_node_ids
            if node_id != return_node_id
        )

    def _can_use_generic_nested_iterate_scheduler(
        self, for_nodes: list[ForInvocation], return_nodes: list[ForReturnInvocation]
    ) -> bool:
        """Admit only the bounded single-outer-For Iterate/Collect contract."""

        if len(for_nodes) != 1 or len(return_nodes) != 1:
            return False
        outer_for = for_nodes[0]
        if not outer_for.collection and not self.graph._get_input_edges(outer_for.id, COLLECTION_FIELD):
            return False

        iterate_nodes = [node for node in self.graph.nodes.values() if isinstance(node, IterateInvocation)]
        collect_nodes = [node for node in self.graph.nodes.values() if isinstance(node, CollectInvocation)]
        source_graph = self._get_source_graph_flat()
        if len(iterate_nodes) == 2 and len(collect_nodes) == 1:
            serial_chain = self.graph._get_supported_for_serial_nested_iterate_chain(outer_for.id, source_graph)
            if (
                can_use_nested_iterate_planner(self)
                and serial_chain is not None
                and len(self.graph.nodes) == 9
                and len(self.graph.edges) == 9
            ):
                body_nodes = set(serial_chain.body_path_nodes)
                outer_consumers = [
                    node for node in self.graph.nodes.values() if node.id not in body_nodes and node.id != outer_for.id
                ]
                if len(outer_consumers) != 1:
                    return False
                consumer = outer_consumers[0]
                output_edges = self.graph._get_output_edges(outer_for.id, "output_collection")
                consumer_inputs = self.graph._get_input_edges(consumer.id)
                return (
                    not isinstance(consumer, (ForInvocation, ForReturnInvocation, IterateInvocation, CollectInvocation))
                    and len(output_edges) == 1
                    and output_edges[0].destination.node_id == consumer.id
                    and output_edges[0].destination.field == "value"
                    and consumer_inputs == output_edges
                )
        if len(iterate_nodes) != 1 or len(collect_nodes) != 1:
            return False
        iterate_node = iterate_nodes[0]
        collect_node = collect_nodes[0]
        return_node = return_nodes[0]

        if (
            not can_use_nested_iterate_planner(self)
            or self.graph._get_supported_for_nested_iterate_body(outer_for.id, source_graph) is None
        ):
            return False

        iterate_collection_edges = self.graph._get_input_edges(iterate_node.id, COLLECTION_FIELD)
        if len(iterate_collection_edges) != 1:
            return False
        preparation_node_id = iterate_collection_edges[0].source.node_id
        preparation_node = self.graph.get_node(preparation_node_id)
        if isinstance(preparation_node, (ForInvocation, ForReturnInvocation, IterateInvocation, CollectInvocation)):
            return False
        preparation_inputs = self.graph._get_input_edges(preparation_node_id)
        if len(preparation_inputs) != 1 or (
            preparation_inputs[0].source.node_id != outer_for.id or preparation_inputs[0].source.field != ITEM_FIELD
        ):
            return False
        if self.graph._get_output_edges(outer_for.id, ITEM_FIELD) != preparation_inputs:
            return False

        collect_item_edges = self.graph._get_input_edges(collect_node.id, ITEM_FIELD)
        if len(collect_item_edges) != 1:
            return False
        body_node_id = collect_item_edges[0].source.node_id
        body_node = self.graph.get_node(body_node_id)
        if body_node_id == preparation_node_id or isinstance(
            body_node, (ForInvocation, ForReturnInvocation, IterateInvocation, CollectInvocation)
        ):
            return False
        body_inputs = self.graph._get_input_edges(body_node_id)
        if len(body_inputs) != 1 or (
            body_inputs[0].source.node_id != iterate_node.id or body_inputs[0].source.field != ITEM_FIELD
        ):
            return False
        if self.graph._get_output_edges(iterate_node.id, ITEM_FIELD) != body_inputs:
            return False
        if self.graph._get_output_edges(body_node_id) != collect_item_edges:
            return False

        if self.graph._get_input_edges(collect_node.id, COLLECTION_FIELD):
            return False
        collect_output_edges = self.graph._get_output_edges(collect_node.id, COLLECTION_FIELD)
        if len(collect_output_edges) != 1 or (
            collect_output_edges[0].destination.node_id != return_node.id
            or collect_output_edges[0].destination.field != "output"
        ):
            return False
        if self.graph._get_input_edges(return_node.id) != collect_output_edges:
            return False
        if self.graph._get_linked_for_return_id(outer_for.id) != return_node.id:
            return False

        final_output_edges = self.graph._get_for_final_output_edges(outer_for.id)
        if len(final_output_edges) != 1:
            return False
        final_consumer_id = final_output_edges[0].destination.node_id
        final_consumer = self.graph.get_node(final_consumer_id)
        if final_consumer_id in {
            outer_for.id,
            iterate_node.id,
            body_node_id,
            collect_node.id,
            return_node.id,
        } or isinstance(final_consumer, (ForInvocation, ForReturnInvocation, IterateInvocation, CollectInvocation)):
            return False
        if self.graph._get_input_edges(final_consumer_id) != final_output_edges:
            return False

        expected_nodes = {
            outer_for.id,
            preparation_node_id,
            iterate_node.id,
            body_node_id,
            collect_node.id,
            return_node.id,
            final_consumer_id,
        }
        outer_collection_edges = self.graph._get_input_edges(outer_for.id, COLLECTION_FIELD)
        if outer_collection_edges:
            expected_nodes.add(outer_collection_edges[0].source.node_id)
        return set(self.graph.nodes) == expected_nodes

    def _can_use_generic_scheduler(self) -> bool:
        """Use generic readiness for static graphs and supported direct control flow."""

        if any(isinstance(node, IfInvocation) for node in self.graph.nodes.values()):
            if not self._legacy_snapshot_loaded and not self._can_use_fresh_flat_if_activation():
                return False

        control_nodes = (ForInvocation, ForReturnInvocation)
        if any(isinstance(node, control_nodes) for node in self.graph.nodes.values()):
            return self._can_use_generic_for_scheduler()
        if any(isinstance(node, (IterateInvocation, CollectInvocation)) for node in self.graph.nodes.values()):
            if any(
                isinstance(node, (*control_nodes, CallSavedWorkflowInvocation)) for node in self.graph.nodes.values()
            ):
                return False
            if any(isinstance(node, IfInvocation) for node in self.graph.nodes.values()):
                return self._can_use_fresh_mixed_if_iterate_collect()
            return True
        return not any(isinstance(node, CallSavedWorkflowInvocation) for node in self.graph.nodes.values())

    def _scheduler(self) -> _ExecutionScheduler | _GenericGraphSchedulerAdapter:
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.execution_scheduler is None:
            if self._can_use_generic_scheduler():
                runtime.generic_graph_scheduler = _GenericGraphSchedulerAdapter(self)
                runtime.execution_scheduler = runtime.generic_graph_scheduler
            else:
                runtime.execution_scheduler = _ExecutionScheduler(self)
        return runtime.execution_scheduler

    def _is_generic_graph_scheduler(self, scheduler: Any) -> bool:
        return isinstance(scheduler, _GenericGraphSchedulerAdapter)

    def _new_execution_node_id(self) -> str:
        return uuid_string()

    def _runtime(self) -> _ExecutionRuntime:
        runtime = self.__pydantic_private__["_runtime_state"]
        if runtime.execution_runtime is None:
            runtime.execution_runtime = _ExecutionRuntime(self)
        return runtime.execution_runtime

    def _generic_runtime(self) -> ExecutionEngineRuntime:
        """Return the private typed runtime used by legacy-control adapters."""

        runtime_state = self.__pydantic_private__["_runtime_state"]
        if runtime_state.generic_execution_runtime is None:
            runtime_state.generic_execution_runtime = ExecutionEngineRuntime()
        return runtime_state.generic_execution_runtime

    def _clear_transient_runtime(self) -> None:
        """Clear scheduling projections after execution reaches a terminal error."""

        if self._generic_execution_runtime is not None:
            self._generic_execution_runtime.gates.clear()
            self._generic_execution_runtime.streams.clear()
            self._generic_execution_runtime.continuations.clear()
        self._ready_queues = {}
        self._ready_node_ids = set()
        self._active_class = None
        if isinstance(self._execution_scheduler, _GenericGraphSchedulerAdapter):
            self._execution_scheduler = None
            self._generic_graph_scheduler = None

    def _engine_frame(self, iteration_path: tuple[int, ...]) -> EngineExecutionFrame:
        frame_id = f"{self.id}:{len(self.workflow_call_stack)}:{','.join(str(i) for i in iteration_path)}"
        return EngineExecutionFrame(
            state_id=self.id,
            frame_id=frame_id,
            iteration_path=iteration_path,
            workflow_call_depth=len(self.workflow_call_stack),
        )

    def _activation_gate(self, exec_node_id: str) -> ActivationGate:
        return graph_if_runtime._activation_gate(self, exec_node_id)

    def _record_compatibility_activation_token(self, exec_node_id: str, selected_field: str) -> None:
        return graph_if_runtime._record_compatibility_activation_token(self, exec_node_id, selected_field)

    def _is_activation_dependency_satisfied(self, dependency: ActivationDependency) -> bool:
        return graph_if_runtime._is_activation_dependency_satisfied(self, dependency)

    def _is_activation_dependency_rejected(self, dependency: ActivationDependency) -> bool:
        return graph_if_runtime._is_activation_dependency_rejected(self, dependency)

    def _resolve_activation_gate(self, exec_node_id: str, branch: str) -> bool:
        return graph_if_runtime._resolve_activation_gate(self, exec_node_id, branch)

    def _iteration_stream_id(self, source_node_id: str, parent_path: tuple[int, ...]) -> str:
        path = ",".join(str(index) for index in parent_path)
        return f"{self.id}:iterate:{source_node_id}:{path}"

    def _record_iterate_stream(
        self,
        exec_node_id: str,
        output: BaseInvocationOutput,
        *,
        prefer_existing: bool = False,
        trusted: bool = False,
    ) -> None:
        if not isinstance(output, IterateInvocationOutput):
            return
        source_node_id = self.prepared_source_mapping.get(exec_node_id)
        if source_node_id is None:
            return
        iteration_path = self._get_iteration_path(exec_node_id)
        parent_path = iteration_path[:-1] if iteration_path else ()
        stream_id = self._iteration_stream_id(source_node_id, parent_path)
        runtime = self._generic_runtime()
        stream = runtime.streams.get(stream_id)
        if stream is None:
            self._tx_record(lambda: runtime.remove_stream(stream_id))
            stream = runtime.get_or_create_stream(stream_id, source_node_id, self._engine_frame(parent_path))
        elif self._apply_transaction is not None:
            transaction_key = ("stream", stream_id)
            if not self._apply_transaction.is_recorded(transaction_key):
                snapshot = (len(stream.events), stream.next_sequence, stream.closed, stream.end_sequence)
                self._tx_record_once(
                    transaction_key,
                    lambda stream=stream, snapshot=snapshot: stream._restore(*snapshot),
                )
        skip_data = False
        if prefer_existing:
            existing = next((event for event in stream.events if event.sequence == output.index), None)
            if existing is not None:
                if isinstance(existing, StreamData) and existing.value != output.item:
                    # Durable effect values are authoritative when an older result mirror is stale.
                    skip_data = True
                elif isinstance(existing, StreamData):
                    skip_data = True
                else:
                    raise ValueError(f"Iterate result conflicts with closed stream {stream_id}")
        if not skip_data:
            event = (
                StreamData.model_construct(kind="data", sequence=output.index, value=output.item)
                if trusted
                else StreamData(sequence=output.index, value=output.item)
            )
            stream.accept(event, trusted=trusted)
        if output.index + 1 >= output.total:
            if stream.closed:
                if stream.end_sequence != output.total:
                    raise ValueError(f"Iterate result conflicts with closed stream {stream_id}")
            else:
                stream.close(sequence=output.total)

    def _record_empty_iterate_stream(self, source_node_id: str, parent_path: tuple[int, ...] = ()) -> None:
        runtime = self._generic_runtime()
        stream_id = self._iteration_stream_id(source_node_id, parent_path)
        stream = runtime.streams.get(stream_id)
        if stream is None:
            self._tx_record(lambda: runtime.remove_stream(stream_id))
            stream = runtime.get_or_create_stream(stream_id, source_node_id, self._engine_frame(parent_path))
        elif stream.closed:
            return
        elif self._apply_transaction is not None:
            transaction_key = ("stream", stream_id)
            if not self._apply_transaction.is_recorded(transaction_key):
                snapshot = (len(stream.events), stream.next_sequence, stream.closed, stream.end_sequence)
                self._tx_record_once(
                    transaction_key,
                    lambda stream=stream, snapshot=snapshot: stream._restore(*snapshot),
                )
        stream.close(sequence=stream.next_sequence)

    def _record_effect_streams(
        self, execution_ref: ExecutionReference, effects: Iterable[Any], *, trusted: bool = False
    ) -> None:
        """Mirror generic stream effects into the private stream registry."""

        for effect in effects:
            effect_kind = (
                effect.kind
                if isinstance(effect, ExecutionEffect)
                else self._value_from_object(effect, "kind", "effect_type", "type")
            )
            if effect_kind not in {"emit", "close_stream"}:
                continue
            token = (
                effect.token
                if isinstance(effect, (EmitEffect, CloseStreamEffect))
                else self._value_from_object(effect, "token")
            )
            port = (
                token.field
                if isinstance(token, EffectExecutionToken)
                else self._value_from_object(token, "field", "port", "output", "output_name")
            )
            if not isinstance(port, str):
                continue
            token_kind = (
                token.token_kind
                if isinstance(token, EffectExecutionToken)
                else self._value_from_object(token, "token_kind")
            ) or "data"
            if token_kind == "activation":
                continue
            stream_id = f"{self.id}:effect:{execution_ref.reference_id}:{port}"
            stream_owner_id = execution_ref.exec_node_id
            stream_frame = EngineExecutionFrame(
                state_id=execution_ref.frame.state_id or self.id,
                frame_id=execution_ref.frame.frame_id or f"{self.id}:{execution_ref.exec_node_id}",
                iteration_path=execution_ref.frame.iteration_path,
                workflow_call_depth=execution_ref.frame.workflow_call_depth,
            )
            source_node = self.execution_graph.nodes.get(execution_ref.exec_node_id)
            source_node_id = self.prepared_source_mapping.get(execution_ref.exec_node_id)
            if isinstance(source_node, IterateInvocation) and port == ITEM_FIELD and source_node_id is not None:
                iteration_path = self._get_iteration_path(execution_ref.exec_node_id)
                parent_path = iteration_path[:-1] if iteration_path else ()
                stream_id = self._iteration_stream_id(source_node_id, parent_path)
                stream_owner_id = source_node_id
                stream_frame = self._engine_frame(parent_path)
            runtime = self._generic_runtime()
            stream = runtime.streams.get(stream_id)
            if stream is None:
                self._tx_record(lambda runtime=runtime, stream_id=stream_id: runtime.remove_stream(stream_id))
                stream = runtime.get_or_create_stream(stream_id, stream_owner_id, stream_frame)
            elif self._apply_transaction is not None:
                transaction_key = ("stream", stream_id)
                if not self._apply_transaction.is_recorded(transaction_key):
                    snapshot = (len(stream.events), stream.next_sequence, stream.closed, stream.end_sequence)
                    self._tx_record_once(
                        transaction_key,
                        lambda stream=stream, snapshot=snapshot: stream._restore(*snapshot),
                    )

            sequence = (
                token.sequence
                if isinstance(token, EffectExecutionToken)
                else self._value_from_object(token, "sequence")
            )
            if sequence is None:
                sequence = stream.next_sequence
            if effect_kind == "close_stream":
                if stream.closed:
                    if stream.end_sequence == sequence:
                        continue
                    raise ValueError("conflicting close sequence for stream")
                stream.close(sequence=sequence)
                continue
            if token_kind == "stream_end":
                continue
            value = effect.value if isinstance(effect, EmitEffect) else self._value_from_object(effect, "value")
            if value is None:
                value = (
                    token.value if isinstance(token, EffectExecutionToken) else self._value_from_object(token, "value")
                )
            event = (
                StreamData.model_construct(kind="data", sequence=sequence, value=value)
                if trusted
                else StreamData(sequence=sequence, value=value)
            )
            stream.accept(event, trusted=trusted)

    def _stream_for_iterate_edge(self, edge: Edge) -> tuple[str, tuple[Any, ...]] | None:
        source_node_id = self.prepared_source_mapping.get(edge.source.node_id)
        if source_node_id is None or edge.source.field != ITEM_FIELD:
            return None
        if not isinstance(self.execution_graph.nodes.get(edge.source.node_id), IterateInvocation):
            return None
        iteration_path = self._get_iteration_path(edge.source.node_id)
        parent_path = iteration_path[:-1] if iteration_path else ()
        return self._iteration_stream_id(source_node_id, parent_path), parent_path

    def _collect_streams_ready(self, exec_node_id: str) -> bool:
        """Require any available Iterate streams to close before Collect is scheduled."""

        collector_source_id = self.prepared_source_mapping.get(exec_node_id)
        fan_in = self._get_direct_iterate_collect_nodes() if collector_source_id is not None else None
        if (
            isinstance(fan_in, (_DirectIterateCollectFanIn, _BodyIterateCollectFanIn))
            and collector_source_id == fan_in.collector_id
        ):
            for branch in fan_in.branches:
                iterator_id = branch[1]
                stream = self._generic_runtime().streams.get(self._iteration_stream_id(iterator_id, ()))
                if stream is None or not stream.closed:
                    return False

        for edge in self.execution_graph._get_input_edges(exec_node_id, ITEM_FIELD):
            stream_info = self._stream_for_iterate_edge(edge)
            if stream_info is None:
                continue
            stream_id, _parent_path = stream_info
            stream = self._generic_runtime().streams.get(stream_id)
            if stream is not None and not stream.closed:
                return False
        return True

    def _for_continuation(self, for_exec_node_id: str) -> ContinuationRecord[Any]:
        runtime = self._generic_runtime()
        continuation_id = f"{self.id}:for:{for_exec_node_id}"
        continuation = runtime.continuations.get(continuation_id)
        if continuation is None:
            self._tx_record(lambda: runtime.remove_continuation(continuation_id))
            continuation = runtime.register_continuation(
                continuation_id,
                for_exec_node_id,
                self._engine_frame(self._get_iteration_path(for_exec_node_id)),
                "for",
            )
        if continuation.status == "pending":
            previous = ContinuationRecord[Any].model_validate(continuation.model_dump(mode="python"))
            self._tx_record_once(("continuation", continuation_id), lambda: runtime.replace_continuation(previous))
            continuation.start()
        return continuation

    def _record_continuation_effects(
        self,
        execution_ref: ExecutionReference,
        effects: Iterable[Any],
        *,
        allow_return_state_override: bool = False,
    ) -> None:
        """Reconcile durable For continuation effects with the private runtime record."""

        node = self.execution_graph.get_node(execution_ref.exec_node_id)
        for effect in effects:
            effect_kind = (
                effect.kind
                if isinstance(effect, ExecutionEffect)
                else self._value_from_object(effect, "kind", "effect_type", "type")
            )
            if effect_kind != "continuation":
                continue

            operation = (
                effect.operation
                if isinstance(effect, ContinuationEffect)
                else self._value_from_object(effect, "operation")
            )
            effect_payload = (
                effect.payload if isinstance(effect, ContinuationEffect) else self._value_from_object(effect, "payload")
            )
            payload = self._normalize_continuation_payload(effect_payload)
            if isinstance(node, ForInvocation) and operation == "start":
                expected_payload = self._prepared_for_continuation_payload(node)
                if self._continuation_payload_key(payload) != self._continuation_payload_key(expected_payload):
                    raise ValueError("continuation start payload does not match prepared For")
                continuation = self._for_continuation(execution_ref.exec_node_id)
                if payload is not None:
                    if continuation.payload is not None and self._continuation_payload_key(
                        continuation.payload
                    ) != self._continuation_payload_key(payload):
                        raise ValueError("started continuation has conflicting payload")
                    if continuation.payload is None:
                        self._tx_set_attr(continuation, "payload", copydeep(payload))
            elif isinstance(node, ForReturnInvocation) and operation == "complete":
                expected_payload = self._prepared_for_continuation_payload(node)
                if expected_payload is not None and not self._for_return_payloads_match(
                    payload, expected_payload, allow_state_override=allow_return_state_override
                ):
                    raise ValueError("completed continuation payload does not match ForReturn output")
                for_exec_node_id = self._get_for_parent(execution_ref.exec_node_id)
                if for_exec_node_id is None:
                    continue
                continuation = self._for_continuation(for_exec_node_id)
                if continuation.terminal:
                    if continuation.status == "completed" and self._continuation_payload_key(
                        continuation.result
                    ) != self._continuation_payload_key(payload):
                        raise ValueError("completed continuation has conflicting result")
                    continue
                previous = ContinuationRecord[Any].model_validate(continuation.model_dump(mode="python"))
                self._tx_record_once(
                    ("continuation-complete", continuation.continuation_id),
                    lambda previous=previous: self._generic_runtime().replace_continuation(previous),
                )
                continuation.complete(copydeep(payload))

    @staticmethod
    def _normalize_continuation_payload(payload: Any) -> Any:
        try:
            return _JSON_SERIALIZER.dump_python(payload, mode="json", warnings="error")
        except (PydanticSerializationError, TypeError, ValueError) as exc:
            raise ValueError("Continuation effect payload must be JSON-serializable") from exc

    @classmethod
    def _continuation_payload_key(cls, payload: Any) -> str:
        normalized = cls._normalize_continuation_payload(payload)
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def _payload_values_match(cls, left: Any, right: Any) -> bool:
        try:
            return cls._continuation_payload_key(left) == cls._continuation_payload_key(right)
        except ValueError:
            try:
                return left is right or left == right
            except Exception:
                return False

    @classmethod
    def _for_return_payloads_match(cls, left: Any, right: Any, *, allow_state_override: bool = False) -> bool:
        """Compare durable return identity, optionally allowing compatibility callers to replace loop state."""

        if not isinstance(left, dict) or not isinstance(right, dict):
            return cls._payload_values_match(left, right)
        if not cls._payload_values_match(left.get("output"), right.get("output")):
            return False
        if left.get("continue_condition") != right.get("continue_condition"):
            return False
        return allow_state_override or cls._payload_values_match(left.get("state"), right.get("state"))

    @classmethod
    def _for_start_continuation_payload(
        cls, node: ForInvocation, output: ForInvocationOutput | None = None
    ) -> dict[str, Any]:
        index = output.index if output is not None else node.index
        total = output.total if output is not None else len(node.collection)
        state = (output.state if output is not None else node.state) or LoopState()
        return {
            "index": index,
            "total": total,
            "state": cls._normalize_continuation_payload(state.model_dump(mode="json")),
        }

    @classmethod
    def _for_return_continuation_payload(
        cls, node: ForReturnInvocation, output: ForReturnInvocationOutput | None = None
    ) -> dict[str, Any] | None:
        if output is not None:
            output_value = output.output
            state = output.state
        else:
            output_value = node.output
            state = node.state
        return {
            "output": cls._normalize_continuation_payload(output_value),
            "state": cls._normalize_continuation_payload(state.model_dump(mode="json")) if state is not None else None,
            "continue_condition": node.continue_condition,
        }

    def _for_collection_for_validation(self, node: ForInvocation) -> list[Any]:
        if node.collection:
            return node.collection

        source_node_id = self.prepared_source_mapping.get(node.id)
        if source_node_id is None:
            return node.collection
        collection_edges = self.graph._get_input_edges(source_node_id, COLLECTION_FIELD)
        if collection_edges:
            source_output = self.results.get(collection_edges[0].source.node_id)
            if source_output is None:
                parent_iteration_path = self._get_for_parent_iteration_path(node.id)
                source_output_ids = [
                    exec_node_id
                    for exec_node_id in self.source_prepared_mapping.get(collection_edges[0].source.node_id, ())
                    if exec_node_id in self.results
                ]
                source_output = next(
                    (
                        self.results[exec_node_id]
                        for exec_node_id in source_output_ids
                        if self._get_iteration_path(exec_node_id) == parent_iteration_path
                    ),
                    None,
                )
            collection = self._value_from_object(source_output, collection_edges[0].source.field)
            if isinstance(collection, list):
                return collection
        source_node = self.graph.get_node(source_node_id)
        if isinstance(source_node, ForInvocation) and source_node.collection:
            return source_node.collection
        return node.collection

    def _prepared_for_continuation_payload(self, node: ForInvocation | ForReturnInvocation) -> dict[str, Any] | None:
        if isinstance(node, ForInvocation):
            collection = self._for_collection_for_validation(node)
            state = node.state or LoopState()
            return {
                "index": node.index,
                "total": len(collection),
                "state": self._normalize_continuation_payload(state.model_dump(mode="json")),
            }
        return self._for_return_continuation_payload(node)

    def _for_item_for_validation(self, node: ForInvocation) -> Any:
        if node.index < 0:
            return None
        collection = self._for_collection_for_validation(node)
        if node.index >= len(collection):
            raise ValueError("For output item does not match prepared For")
        return collection[node.index]

    def _validate_for_continuation_output(
        self,
        execution_ref: ExecutionReference,
        output: BaseInvocationOutput,
        *,
        allow_return_state_override: bool = False,
    ) -> None:
        node = self.execution_graph.get_node(execution_ref.exec_node_id)
        if isinstance(node, ForInvocation) and isinstance(output, ForInvocationOutput):
            expected_payload = self._prepared_for_continuation_payload(node)
            actual_payload = self._for_start_continuation_payload(node, output)
            if self._continuation_payload_key(actual_payload) != self._continuation_payload_key(expected_payload):
                raise ValueError("For output does not match prepared For")
            if not self._payload_values_match(output.item, self._for_item_for_validation(node)):
                raise ValueError("For output item does not match prepared For")
            source_node_id = self.prepared_source_mapping.get(node.id)
            parent_iteration_path = self._get_for_parent_iteration_path(node.id)
            final_for_id = None
            if source_node_id is not None:
                self._get_prepared_for_index()
                assert self._final_prepared_for_index is not None
                final_for_id = self._final_prepared_for_index.get((source_node_id, parent_iteration_path))
            if (
                source_node_id is not None
                and final_for_id == node.id
                and self._is_loop_context_finalized(source_node_id, parent_iteration_path)
            ):
                body_path_to_return = self.graph._get_for_body_path_to_return(
                    source_node_id, self._get_source_graph_flat()
                )
                if body_path_to_return is not None:
                    _body_path_nodes, source_return_id = body_path_to_return
                    return_outputs = self._get_ordered_for_return_outputs(node.id, source_return_id)
                    expected_collection = [return_output.output for return_output in return_outputs]
                    if self._continuation_payload_key(output.output_collection) != self._continuation_payload_key(
                        expected_collection
                    ):
                        raise ValueError("For output output_collection is not authoritative")
                    final_return = return_outputs[-1] if return_outputs else None
                    expected_final_state = (
                        self._get_loop_state_for_next_iteration(node.id, final_return)
                        if final_return is not None
                        else node.state
                    )
                    if self._continuation_payload_key(output.final_state) != self._continuation_payload_key(
                        expected_final_state
                    ):
                        raise ValueError("For output final_state is not authoritative")

        elif isinstance(node, ForReturnInvocation) and isinstance(output, ForReturnInvocationOutput):
            expected_payload = self._prepared_for_continuation_payload(node)
            actual_payload = self._for_return_continuation_payload(node, output)
            if not self._for_return_payloads_match(
                actual_payload, expected_payload, allow_state_override=allow_return_state_override
            ):
                raise ValueError("ForReturn output does not match prepared ForReturn")

    def _requires_continuation_effect(self, execution_ref: ExecutionReference) -> bool:
        node = self.execution_graph.get_node(execution_ref.exec_node_id)
        if isinstance(node, ForReturnInvocation):
            return True
        if not isinstance(node, ForInvocation):
            return False
        output = self.results.get(execution_ref.exec_node_id)
        return isinstance(output, ForInvocationOutput) and output.total > 0

    def _build_compatibility_continuation_effect(
        self, execution_ref: ExecutionReference, output: BaseInvocationOutput
    ) -> ContinuationEffect:
        node = self.execution_graph.get_node(execution_ref.exec_node_id)
        if isinstance(node, ForInvocation) and isinstance(output, ForInvocationOutput):
            operation = "start"
            payload = self._for_start_continuation_payload(node, output)
        elif isinstance(node, ForReturnInvocation) and isinstance(output, ForReturnInvocationOutput):
            operation = "complete"
            payload = self._for_return_continuation_payload(node, output)
        else:
            raise TypeError("Compatibility continuation effects require a For or ForReturn output")

        effect_ref = EffectExecutionRef(
            execution_node_id=execution_ref.exec_node_id,
            state_id=execution_ref.state_id,
            frame_path=execution_ref.frame.iteration_path,
            frame_id=execution_ref.frame.frame_id,
            workflow_call_depth=execution_ref.frame.workflow_call_depth,
        )
        return ContinuationEffect(
            execution_ref=effect_ref,
            operation=operation,
            continuation_kind="for",
            payload=payload,
        )

    def _complete_for_continuation(self, for_exec_node_id: str, result: Any) -> None:
        result = self._normalize_continuation_payload(result)
        continuation = self._for_continuation(for_exec_node_id)
        if continuation.terminal:
            if continuation.status == "completed" and self._continuation_payload_key(
                continuation.result
            ) != self._continuation_payload_key(result):
                raise ValueError("completed continuation has conflicting result")
            return
        previous = ContinuationRecord[Any].model_validate(continuation.model_dump(mode="python"))
        self._tx_record_once(
            ("continuation-complete", continuation.continuation_id),
            lambda: self._generic_runtime().replace_continuation(previous),
        )
        continuation.complete(copydeep(result))

    def _register_generic_child_dependency(self) -> ChildDependencyRecord | None:
        execution = self.waiting_workflow_call_execution
        if execution is None or not execution.child_item_ids:
            return None
        existing = self._generic_child_dependencies.get(execution.id)
        if existing is not None:
            expected_ref = self.execution_refs.get(execution.prepared_call_node_id)
            if expected_ref is None:
                expected_ref = self._expected_execution_ref(execution.prepared_call_node_id)
            if (
                existing.parent_execution_id != expected_ref.exec_node_id
                or existing.parent_reference_id != expected_ref.reference_id
                or existing.parent_frame != expected_ref.frame.iteration_path
                or existing.child_execution_ids != [str(item_id) for item_id in execution.child_item_ids]
            ):
                raise ValueError("Child dependency does not belong to active workflow call")
            return existing
        persisted = self.execution_child_dependencies.get(execution.id)
        if persisted is not None:
            expected_ref = self.execution_refs.get(execution.prepared_call_node_id)
            if expected_ref is None:
                expected_ref = self._expected_execution_ref(execution.prepared_call_node_id)
            if (
                persisted.parent_execution_id != expected_ref.exec_node_id
                or persisted.parent_reference_id != expected_ref.reference_id
                or persisted.parent_frame != expected_ref.frame.iteration_path
                or persisted.child_execution_ids != [str(item_id) for item_id in execution.child_item_ids]
            ):
                raise ValueError("Persisted child dependency does not belong to active workflow call")
            self._generic_child_dependencies[execution.id] = persisted
            return persisted
        execution_ref = self.execution_refs.get(execution.prepared_call_node_id)
        if execution_ref is None:
            execution_ref = self._expected_execution_ref(execution.prepared_call_node_id)
        parent_reference_id = execution_ref.reference_id
        capability = ChildExecutionCapability(
            parent_execution_id=execution_ref.exec_node_id,
            parent_frame=execution_ref.frame.iteration_path,
            parent_reference_id=parent_reference_id,
            depth=max(execution.depth - 1, 0),
            max_depth=max(self.max_workflow_call_depth, execution.depth),
            max_children=max(execution.expected_child_count, 1),
            capacity=max(execution.expected_child_count, 1),
        )
        record = capability.create_dependency(
            [str(item_id) for item_id in execution.child_item_ids],
            dependency_id=execution.id,
        )
        for child_item_id in execution.completed_child_item_ids:
            output_values = execution.child_outputs.get(child_item_id)
            if output_values is not None:
                record.complete_child(str(child_item_id), output_values)
        self._generic_child_dependencies[execution.id] = record
        self.execution_child_dependencies[execution.id] = record
        return record

    def build_child_execution_capability(
        self,
        execution_ref: ExecutionReference,
        *,
        authorization_context: dict[str, Any] | None = None,
        max_children: int = 1,
    ) -> ChildExecutionCapability:
        """Issue lifecycle authority bound to one prepared node and exact frame."""

        ref = self._validate_execution_ref(execution_ref)
        if max_children < 1:
            raise ValueError("max_children must be at least one")
        return ChildExecutionCapability(
            parent_execution_id=ref.exec_node_id,
            parent_frame=ref.frame.iteration_path,
            parent_reference_id=ref.reference_id,
            authorization_context=authorization_context,
            depth=ref.frame.workflow_call_depth,
            max_depth=self.max_workflow_call_depth,
            max_children=max_children,
            capacity=max_children,
        )

    def record_generic_child_completion(
        self, child_item_id: int, output_values: dict[str, Any]
    ) -> ChildDependencyUpdate | None:
        dependency = self._register_generic_child_dependency()
        if dependency is None:
            return None
        return dependency.complete_child(str(child_item_id), output_values)

    def fail_generic_child(self, child_item_id: int, error_message: str) -> ChildDependencyUpdate | None:
        dependency = self._register_generic_child_dependency()
        if dependency is None:
            return None
        return dependency.fail_child(str(child_item_id), error_message)

    def cancel_generic_child(self, child_item_id: int, error_message: str) -> ChildDependencyUpdate | None:
        dependency = self._register_generic_child_dependency()
        if dependency is None:
            return None
        return dependency.cancel_child(str(child_item_id), error_message)

    def _register_prepared_exec_node(self, exec_node_id: str, source_node_id: str) -> None:
        self._prepared_if_exec_indexed_sources.discard(source_node_id)
        for key in tuple(self._prepared_if_exec_ids_by_source_and_frame):
            if key[0] == source_node_id:
                self._prepared_if_exec_ids_by_source_and_frame.pop(key, None)
        self._prepared_if_exec_frames_by_source.pop(source_node_id, None)
        is_new = exec_node_id not in self.source_prepared_mapping.get(source_node_id, ())
        self._prepared_registry().register(exec_node_id, source_node_id)
        if is_new and self._unexecuted_prepared_counts is not None and exec_node_id not in self.executed:
            counts = self._unexecuted_prepared_counts
            old_count = counts.get(source_node_id, 0)
            self._tx_record_once(
                ("unexecuted_prepared_count", source_node_id),
                lambda: counts.__setitem__(source_node_id, old_count),
            )
            counts[source_node_id] = old_count + 1
        self._tx_discard_set(self.executed, source_node_id)
        if self._completed_source_ids_cache is not None:
            self._tx_discard_set(self._completed_source_ids_cache, source_node_id)
        self._invalidate_loop_caches_for_source(source_node_id)
        if (
            self._prepared_for_index is not None
            and self._prepared_registry().get_iteration_path(exec_node_id) is not None
        ):
            self._update_prepared_for_index(exec_node_id)

    @property
    def prepared_execution_refs(self) -> dict[str, ExecutionReference]:
        """Compatibility view using the protocol's prepared-reference terminology."""

        return self.execution_refs

    def _get_execution_frame(self, exec_node_id: str) -> ExecutionFrame:
        iteration_path = self.prepared_iteration_paths.get(exec_node_id, ())
        frame_id = f"{self.id}:{len(self.workflow_call_stack)}:{','.join(str(i) for i in iteration_path)}"
        return ExecutionFrame(
            frame_id=frame_id,
            state_id=self.id,
            iteration_path=iteration_path,
            workflow_call_depth=len(self.workflow_call_stack),
        )

    def _expected_execution_ref(self, exec_node_id: str, effect_count: Optional[int] = None) -> ExecutionReference:
        if exec_node_id not in self.execution_graph.nodes:
            raise NodeNotFoundError(f"Node {exec_node_id} not found in execution graph")
        source_node_id = self.prepared_source_mapping.get(exec_node_id)
        if source_node_id is None:
            raise ValueError(f"Node {exec_node_id} is not a prepared execution node")
        return ExecutionReference(
            reference_id=f"{self.id}:{exec_node_id}",
            state_id=self.id,
            exec_node_id=exec_node_id,
            source_node_id=source_node_id,
            frame=self._get_execution_frame(exec_node_id),
            effect_count=effect_count,
        )

    def get_execution_ref(self, exec_node_id: str, *, effect_count: Optional[int] = None) -> ExecutionReference:
        """Return stable reference for prepared execution node."""

        expected = self._expected_execution_ref(exec_node_id, effect_count=effect_count)
        existing = self.execution_refs.get(exec_node_id)
        if existing is not None:
            if (
                existing.reference_id not in ("", expected.reference_id)
                or existing.state_id not in ("", expected.state_id)
                or existing.exec_node_id not in ("", expected.exec_node_id)
                or existing.source_node_id not in ("", expected.source_node_id)
            ):
                raise ValueError(f"Execution reference for {exec_node_id} does not belong to this state")
            if existing.effect_count is not None and effect_count is None:
                expected.effect_count = existing.effect_count
        self.execution_refs[exec_node_id] = expected
        persisted_effects = self.execution_effects.get(expected.reference_id)
        if persisted_effects is not None:
            self._update_pending_lifecycle_execution(exec_node_id, persisted_effects)
        return expected.model_copy(deep=True)

    get_execution_reference = get_execution_ref

    def _coerce_execution_ref(self, execution_ref: ExecutionReference | str | Any) -> ExecutionReference:
        if isinstance(execution_ref, str):
            return self._expected_execution_ref(execution_ref)
        if isinstance(execution_ref, ExecutionReference):
            return execution_ref.model_copy(deep=True)

        if isinstance(execution_ref, BaseModel):
            values = execution_ref.model_dump(mode="python", warnings=False, exclude_none=True)
        elif isinstance(execution_ref, dict):
            values = dict(execution_ref)
        else:
            values = {
                name: getattr(execution_ref, name)
                for name in (
                    "reference_id",
                    "id",
                    "state_id",
                    "session_id",
                    "exec_node_id",
                    "execution_node_id",
                    "node_id",
                    "prepared_node_id",
                    "source_node_id",
                    "frame",
                    "frame_path",
                    "effect_count",
                    "expected_effect_count",
                )
                if hasattr(execution_ref, name)
            }

        if "reference_id" not in values and "id" in values:
            values["reference_id"] = values["id"]
        if "state_id" not in values and "session_id" in values:
            values["state_id"] = values["session_id"]
        if "exec_node_id" not in values or not values["exec_node_id"]:
            values["exec_node_id"] = (
                values.get("execution_node_id") or values.get("node_id") or values.get("prepared_node_id", "")
            )
        if "effect_count" not in values and "expected_effect_count" in values:
            values["effect_count"] = values["expected_effect_count"]
        token = values.get("token")
        if "exec_node_id" not in values or not values["exec_node_id"]:
            token_node_id = self._value_from_object(
                token, "exec_node_id", "prepared_node_id", "node_id", "invocation_id"
            )
            if token_node_id is not None:
                values["exec_node_id"] = token_node_id
        if "source_node_id" not in values:
            values["source_node_id"] = ""
        frame = values.get("frame")
        if frame is None:
            frame = values.get("frame_path")
        if frame is None:
            frame = self._value_from_object(token, "frame", "iteration_path", "frame_path")
        if isinstance(frame, (tuple, list)):
            frame = {"iteration_path": tuple(frame)}
        if isinstance(frame, str):
            values["frame"] = {"frame_id": frame}
        elif frame is not None:
            values["frame"] = frame
        return ExecutionReference.model_validate(values, strict=False)

    @staticmethod
    def _value_from_object(value: Any, *names: str) -> Any:
        if isinstance(value, dict):
            for name in names:
                if name in value:
                    return value[name]
            return None
        if isinstance(value, BaseModel):
            fields = value.__dict__
            extras = value.__pydantic_extra__ or {}
            for name in names:
                if name in fields:
                    return fields[name]
                if name in extras:
                    return extras[name]
            return next((getattr(value, name) for name in names if hasattr(type(value), name)), None)
        for name in names:
            if hasattr(value, name):
                return getattr(value, name)
        return None

    @staticmethod
    def _same_execution_owner(owner: Any, execution_ref: ExecutionReference) -> bool:
        if isinstance(owner, (ExecutionReference, str)):
            return owner in {
                execution_ref.reference_id,
                execution_ref.exec_node_id,
            }
        if isinstance(owner, EffectExecutionRef):
            owner_id = owner.node_id
            owner_state_id = owner.state_id
            return owner_id in {execution_ref.reference_id, execution_ref.exec_node_id} and owner_state_id in {
                None,
                execution_ref.state_id,
            }
        if isinstance(owner, EffectExecutionToken):
            return owner.node_id == execution_ref.exec_node_id
        if isinstance(owner, BaseModel):
            owner = owner.model_dump(mode="python", warnings=False)
        if isinstance(owner, dict):
            token = owner.get("token")
            if token is not None:
                return GraphExecutionState._same_execution_owner(token, execution_ref)
            owner_id = (
                owner.get("reference_id")
                or owner.get("id")
                or owner.get("exec_node_id")
                or owner.get("execution_node_id")
                or owner.get("node_id")
            )
            owner_state_id = owner.get("state_id") or owner.get("session_id")
            return owner_id in {execution_ref.reference_id, execution_ref.exec_node_id} and owner_state_id in {
                None,
                execution_ref.state_id,
            }
        owner_id = GraphExecutionState._value_from_object(
            owner,
            "reference_id",
            "exec_node_id",
            "execution_node_id",
            "prepared_node_id",
            "node_id",
            "invocation_id",
        )
        return owner_id in {execution_ref.reference_id, execution_ref.exec_node_id}

    def _validate_execution_frame(
        self, frame: Any, execution_ref: ExecutionReference, owner_name: str = "Execution effect"
    ) -> None:
        if frame is None:
            return
        if isinstance(frame, str):
            if frame not in ("", execution_ref.frame.frame_id):
                raise ValueError(f"{owner_name} belongs to another execution frame")
            return
        if isinstance(frame, (tuple, list)):
            if tuple(frame) != execution_ref.frame.iteration_path:
                raise ValueError(f"{owner_name} belongs to another execution frame")
            return

        frame_id = self._value_from_object(frame, "frame_id", "id")
        if frame_id not in (None, "", execution_ref.frame.frame_id):
            raise ValueError(f"{owner_name} belongs to another execution frame")
        frame_state_id = self._value_from_object(frame, "state_id", "session_id")
        if frame_state_id not in (None, "", execution_ref.frame.state_id):
            raise ValueError(f"{owner_name} belongs to another graph execution state")
        frame_path = self._value_from_object(frame, "iteration_path", "frame_path")
        if frame_path is not None and tuple(frame_path) != execution_ref.frame.iteration_path:
            raise ValueError(f"{owner_name} belongs to another execution frame")
        frame_depth = self._value_from_object(frame, "workflow_call_depth", "call_depth")
        if frame_depth not in (None, execution_ref.frame.workflow_call_depth):
            raise ValueError(f"{owner_name} belongs to another workflow-call depth")

    def _validate_execution_token(self, token: Any, execution_ref: ExecutionReference) -> None:
        if isinstance(token, EffectExecutionToken):
            if token.node_id not in ("", execution_ref.exec_node_id):
                raise ValueError("Execution token is not owned by execution reference")
            if token.frame and token.frame != execution_ref.frame.iteration_path:
                raise ValueError("Execution token belongs to another execution frame")
            return

        token_node_id = self._value_from_object(
            token,
            "node_id",
            "invocation_id",
            "source_node_id",
            "owner_node_id",
            "exec_node_id",
            "execution_node_id",
            "prepared_node_id",
        )
        if token_node_id not in (None, "", execution_ref.exec_node_id):
            raise ValueError("Execution token is not owned by execution reference")
        token_reference_id = self._value_from_object(token, "reference_id", "execution_ref_id")
        if token_reference_id not in (None, "", execution_ref.reference_id):
            raise ValueError("Execution token belongs to another execution reference")
        token_state_id = self._value_from_object(token, "state_id", "session_id")
        if token_state_id not in (None, "", execution_ref.state_id):
            raise ValueError("Execution token belongs to another graph execution state")

        token_frame = self._value_from_object(token, "frame")
        # Typed persisted tokens materialize the legacy-omitted frame as the model default ``()``. Treat that
        # empty value as absent so loading old stream effects does not invent a conflicting execution frame.
        if token_frame not in (None, (), [], ""):
            self._validate_execution_frame(token_frame, execution_ref, "Execution token")
        token_frame_id = self._value_from_object(token, "frame_id")
        if token_frame_id not in (None, "", execution_ref.frame.frame_id):
            raise ValueError("Execution token belongs to another execution frame")
        token_path = self._value_from_object(token, "iteration_path", "frame_path")
        if token_path not in (None, (), [], "") and tuple(token_path) != execution_ref.frame.iteration_path:
            raise ValueError("Execution token belongs to another execution frame")
        token_depth = self._value_from_object(token, "workflow_call_depth", "call_depth")
        if token_depth not in (None, execution_ref.frame.workflow_call_depth):
            raise ValueError("Execution token belongs to another workflow-call depth")

    def _validate_execution_ref(self, execution_ref: ExecutionReference | str | Any) -> ExecutionReference:
        ref = self._coerce_execution_ref(execution_ref)
        expected = self._expected_execution_ref(ref.exec_node_id)
        if ref.state_id not in ("", expected.state_id):
            raise ValueError("Execution reference belongs to another graph execution state")
        if ref.reference_id not in ("", expected.reference_id):
            raise ValueError("Execution reference id is stale")
        if ref.source_node_id not in ("", expected.source_node_id):
            raise ValueError("Execution reference source node does not match prepared node")
        if ref.frame.state_id not in ("", expected.frame.state_id):
            raise ValueError("Execution reference frame belongs to another state")
        if ref.frame.frame_id not in ("", expected.frame.frame_id):
            raise ValueError("Execution reference frame is stale")
        if ref.frame.iteration_path and ref.frame.iteration_path != expected.frame.iteration_path:
            raise ValueError("Execution reference iteration frame does not match prepared node")
        if ref.frame.workflow_call_depth not in (0, expected.frame.workflow_call_depth):
            raise ValueError("Execution reference workflow frame does not match prepared node")
        ref.reference_id = expected.reference_id
        ref.state_id = expected.state_id
        ref.source_node_id = expected.source_node_id
        ref.frame = expected.frame
        return ref

    def _validate_output_owner(self, execution_ref: ExecutionReference, output: Any) -> BaseInvocationOutput:
        wrapper = output
        if not isinstance(output, BaseInvocationOutput) and isinstance(output, dict):
            wrapper = output.get("output", output.get("result", output))
        owner = self._value_from_object(output, "execution_ref", "execution_reference", "owner_ref", "owner")
        if owner is not None and not self._same_execution_owner(owner, execution_ref):
            raise ValueError("Invocation output is owned by another execution")
        output_node_id = self._value_from_object(output, "exec_node_id", "prepared_node_id", "node_id")
        if output_node_id is not None and output_node_id != execution_ref.exec_node_id:
            raise ValueError("Invocation output node does not match execution reference")
        if not isinstance(wrapper, BaseInvocationOutput):
            raise TypeError("GraphExecutionState.apply() requires a BaseInvocationOutput")

        node = self.execution_graph.get_node(execution_ref.exec_node_id)
        expected_output_type = type(node).get_output_annotation()
        if not isinstance(wrapper, expected_output_type):
            raise TypeError(
                f"Output type {type(wrapper).__name__} does not belong to execution node "
                f"{execution_ref.exec_node_id} ({expected_output_type.__name__})"
            )

        source_node_id = self.prepared_source_mapping[execution_ref.exec_node_id]
        if isinstance(node, ForInvocation):
            linkage_edges = self.graph._get_loop_linkage_edges(source_node_id)
            if len(linkage_edges) != 1 or linkage_edges[0].source.field != LOOP_LINKAGE_FIELD:
                raise ValueError("For execution is missing loop-linkage metadata")
            if getattr(wrapper, LOOP_LINKAGE_FIELD, None) != LOOP_LINKAGE_FIELD:
                raise ValueError("For output is missing loop-linkage metadata")
        elif isinstance(node, ForReturnInvocation):
            linkage_edges = self.graph._get_loop_linkage_edges(source_node_id)
            if len(linkage_edges) != 1 or linkage_edges[0].destination.field != LOOP_LINKAGE_FIELD:
                raise ValueError("ForReturn execution is missing loop-linkage metadata")
        return wrapper

    def _validate_effects(
        self,
        execution_ref: ExecutionReference,
        effects: list[Any],
        effect_count: Optional[int],
        *,
        require_continuation: bool = False,
        validate_json: bool = True,
    ) -> list[Any]:
        expected_count = effect_count if effect_count is not None else execution_ref.effect_count
        if expected_count is not None and len(effects) != expected_count:
            raise ValueError(f"Effect count mismatch: expected {expected_count}, got {len(effects)}")

        node = self.execution_graph.get_node(execution_ref.exec_node_id)
        output_fields = type(node).get_output_annotation().model_fields
        continuation_payloads: dict[tuple[str, str], str] = {}
        duplicate_continuations: set[tuple[str, str]] = set()
        continuation_operations: set[str] = set()
        spawned_child_ids: set[str] = set()
        awaited_child_ids: list[str] = []
        lifecycle_effect_kinds: set[str] = set()
        for effect in effects:
            if validate_json:
                try:
                    _JSON_SERIALIZER.dump_python(effect, mode="json", warnings="error")
                except (PydanticSerializationError, TypeError, ValueError) as exc:
                    raise ValueError("Execution effect must be JSON-serializable") from exc
            effect_kind = (
                effect.kind
                if isinstance(effect, ExecutionEffect)
                else self._value_from_object(effect, "kind", "effect_type", "type")
            )
            if effect_kind in {"spawn_execution", "await", "fail"}:
                lifecycle_effect_kinds.add(effect_kind)
                lifecycle_owner = (
                    effect.execution_ref
                    if isinstance(effect, ExecutionEffect)
                    else self._value_from_object(effect, "execution_ref", "execution_reference")
                )
                if lifecycle_owner is not None:
                    if isinstance(lifecycle_owner, EffectExecutionRef):
                        owner_reference_id = None
                        owner_state_id = lifecycle_owner.state_id
                        owner_frame = lifecycle_owner.frame
                        owner_frame_id = lifecycle_owner.frame_id
                        owner_depth = lifecycle_owner.workflow_call_depth
                    else:
                        owner_reference_id = self._value_from_object(lifecycle_owner, "reference_id", "id")
                        owner_state_id = self._value_from_object(lifecycle_owner, "state_id", "session_id")
                        owner_frame = self._value_from_object(lifecycle_owner, "frame", "frame_path")
                        owner_frame_id = self._value_from_object(lifecycle_owner, "frame_id")
                        owner_depth = self._value_from_object(
                            lifecycle_owner, "workflow_call_depth", "call_depth", "depth"
                        )
                    if owner_reference_id not in (None, "", execution_ref.reference_id):
                        raise ValueError("Lifecycle effect belongs to another execution reference")
                    if owner_state_id not in (None, "", execution_ref.state_id):
                        raise ValueError("Lifecycle effect belongs to another graph execution state")
                    self._validate_execution_frame(
                        owner_frame,
                        execution_ref,
                        "Lifecycle effect",
                    )
                    if owner_frame_id not in (None, "", execution_ref.frame.frame_id):
                        raise ValueError("Lifecycle effect belongs to another execution frame")
                    if owner_depth not in (None, execution_ref.frame.workflow_call_depth):
                        raise ValueError("Lifecycle effect belongs to another workflow-call depth")
                if effect_kind == "spawn_execution":
                    child_execution_id = (
                        effect.child_execution_id
                        if isinstance(effect, SpawnExecutionEffect)
                        else self._value_from_object(effect, "child_execution_id", "child_id")
                    )
                    if not isinstance(child_execution_id, str) or not child_execution_id.strip():
                        raise ValueError("Spawn effect requires a child execution identity")
                    if child_execution_id in spawned_child_ids:
                        raise ValueError("Spawn effect repeats a child execution identity")
                    spawned_child_ids.add(child_execution_id)
                    parent = (
                        effect.parent
                        if isinstance(effect, SpawnExecutionEffect)
                        else self._value_from_object(effect, "parent")
                    )
                    if parent is None or not self._same_execution_owner(parent, execution_ref):
                        raise ValueError("Spawn effect parent is not owned by execution reference")
                    self._validate_execution_frame(
                        parent.frame
                        if isinstance(parent, EffectExecutionRef)
                        else self._value_from_object(parent, "frame", "frame_path"),
                        execution_ref,
                        "Spawn effect parent",
                    )
                elif effect_kind == "await":
                    dependency = (
                        effect.dependency
                        if isinstance(effect, AwaitEffect)
                        else self._value_from_object(effect, "dependency")
                    )
                    child_execution_id = (
                        dependency.node_id
                        if isinstance(dependency, EffectExecutionRef)
                        else self._value_from_object(
                            dependency,
                            "execution_node_id",
                            "exec_node_id",
                            "node_id",
                            "child_execution_id",
                            "execution_id",
                        )
                    )
                    if not isinstance(child_execution_id, str) or not child_execution_id.strip():
                        raise ValueError("Await effect requires a child execution identity")
                    awaited_child_ids.append(child_execution_id)
                elif effect_kind == "fail":
                    message = (
                        effect.message if isinstance(effect, FailEffect) else self._value_from_object(effect, "message")
                    )
                    if not isinstance(message, str):
                        raise ValueError("Fail effect requires an error message")

            if isinstance(effect, (EmitEffect, CloseStreamEffect)):
                token = effect.token
            elif isinstance(effect, ExecutionEffect):
                token = None
            else:
                token = self._value_from_object(effect, "token")
            token_port = (
                token.field
                if isinstance(token, EffectExecutionToken)
                else self._value_from_object(token, "field", "port", "output", "output_name")
            )
            preliminary_port = (
                token_port
                if isinstance(effect, ExecutionEffect)
                else self._value_from_object(effect, "source_port", "output_port", "source_field", "port")
            )
            if preliminary_port is None:
                preliminary_port = token_port
            if isinstance(preliminary_port, str) and preliminary_port in _RESERVED_EFFECT_PORTS:
                raise ValueError(f"Execution effect references reserved output port '{preliminary_port}'")
            if effect_kind == "close_stream":
                token_kind = (
                    token.token_kind
                    if isinstance(token, EffectExecutionToken)
                    else self._value_from_object(token, "token_kind")
                )
                if token is None or token_port in (None, LOOP_LINKAGE_FIELD):
                    raise ValueError("Close-stream effect requires a data output token")
                if token_kind != "stream_end":
                    raise ValueError("Close-stream effect requires a stream_end token")
            elif effect_kind == "emit":
                if token is None or token_port in (None, LOOP_LINKAGE_FIELD):
                    raise ValueError("Emit effect requires a data output token")
                token_kind = (
                    token.token_kind
                    if isinstance(token, EffectExecutionToken)
                    else self._value_from_object(token, "token_kind")
                )
                if token_kind == "stream_end":
                    raise ValueError("Emit effect cannot use a stream_end token")
            elif effect_kind == "continuation":
                if isinstance(effect, ContinuationEffect):
                    continuation_kind = effect.continuation_kind
                    operation = effect.operation
                    continuation_owner = effect.execution_ref
                    continuation_payload = effect.payload
                else:
                    continuation_kind = self._value_from_object(effect, "continuation_kind")
                    operation = self._value_from_object(effect, "operation")
                    continuation_owner = self._value_from_object(
                        effect,
                        "execution_ref",
                        "execution_reference",
                        "owner_ref",
                        "owner",
                    )
                    continuation_payload = self._value_from_object(effect, "payload")
                expected_operation = "start" if isinstance(node, ForInvocation) else "complete"
                if not isinstance(node, (ForInvocation, ForReturnInvocation)):
                    raise ValueError("Continuation effects are only supported by For control-flow nodes")
                if continuation_kind != "for":
                    raise ValueError("Unsupported continuation effect kind")
                if operation != expected_operation:
                    raise ValueError(
                        f"{type(node).__name__} continuation effect must use operation '{expected_operation}'"
                    )
                continuation_operations.add(operation)
                if isinstance(continuation_owner, EffectExecutionRef):
                    continuation_state_id = continuation_owner.state_id
                    continuation_frame_path = continuation_owner.frame
                    continuation_frame_id = continuation_owner.frame_id
                    continuation_depth = continuation_owner.workflow_call_depth
                else:
                    continuation_state_id = self._value_from_object(continuation_owner, "state_id", "session_id")
                    continuation_frame_path = self._value_from_object(
                        continuation_owner, "frame", "frame_path", "iteration_path"
                    )
                    continuation_frame_id = self._value_from_object(continuation_owner, "frame_id")
                    continuation_depth = self._value_from_object(
                        continuation_owner, "workflow_call_depth", "call_depth", "depth"
                    )
                if continuation_state_id not in (None, "", execution_ref.state_id):
                    raise ValueError("Continuation effect belongs to another graph execution state")
                if continuation_frame_id not in (None, "", execution_ref.frame.frame_id):
                    raise ValueError("Continuation effect belongs to another execution frame")
                if continuation_depth not in (None, execution_ref.frame.workflow_call_depth):
                    raise ValueError("Continuation effect belongs to another workflow-call depth")
                if continuation_state_id in (None, ""):
                    raise ValueError("Continuation effect is missing graph execution-state identity")
                if continuation_frame_id in (None, ""):
                    raise ValueError("Continuation effect is missing durable frame identity")
                if continuation_depth is None:
                    raise ValueError("Continuation effect is missing workflow-call depth")
                if continuation_frame_path is None:
                    raise ValueError("Continuation effect is missing iteration-frame identity")
                self._validate_execution_frame(
                    continuation_frame_path,
                    execution_ref,
                    "Continuation effect",
                )
                continuation_key = (operation, continuation_kind)
                serialized_payload = self._continuation_payload_key(continuation_payload)
                if continuation_key in continuation_payloads:
                    if continuation_payloads[continuation_key] != serialized_payload:
                        if operation == "complete":
                            raise ValueError("completed continuation has conflicting result")
                        raise ValueError("started continuation has conflicting payload")
                    duplicate_continuations.add(continuation_key)
                else:
                    continuation_payloads[continuation_key] = serialized_payload

            if isinstance(effect, ExecutionEffect):
                owner = effect.execution_ref
                if owner is None:
                    if isinstance(effect, SpawnExecutionEffect):
                        owner = effect.parent
                    elif isinstance(effect, AwaitEffect):
                        owner = effect.dependency
                    elif isinstance(effect, SetValueEffect):
                        owner = effect.target
                    elif isinstance(effect, (AddEdgeEffect, RemoveEdgeEffect)):
                        owner = effect.source
                    elif isinstance(effect, (EmitEffect, CloseStreamEffect)):
                        owner = effect.token
            else:
                owner = self._value_from_object(
                    effect,
                    "execution_ref",
                    "execution_reference",
                    "owner_ref",
                    "owner",
                    "parent",
                    "dependency",
                    "owner_node_id",
                    "source_node_id",
                    "node_id",
                )
                if owner is None:
                    owner = self._value_from_object(effect, "target", "source", "token")
            if owner is None or not self._same_execution_owner(owner, execution_ref):
                raise ValueError("Execution effect is not owned by execution reference")
            if effect_kind is not None and (
                not isinstance(effect_kind, str)
                or effect_kind not in {"emit", "close_stream", "continuation", "spawn_execution", "await", "fail"}
            ):
                raise ValueError(f"Unsupported execution effect kind: {effect_kind}")

            effect_state_id = (
                None
                if isinstance(effect, ExecutionEffect)
                else self._value_from_object(effect, "state_id", "session_id")
            )
            if effect_state_id is not None and effect_state_id != execution_ref.state_id:
                raise ValueError("Execution effect belongs to another graph execution state")
            effect_frame = None if isinstance(effect, ExecutionEffect) else self._value_from_object(effect, "frame")
            self._validate_execution_frame(effect_frame, execution_ref)

            if isinstance(effect, (AddEdgeEffect, RemoveEdgeEffect)):
                source_ref = effect.source
            elif isinstance(effect, SetValueEffect):
                source_ref = effect.target
            elif isinstance(effect, (EmitEffect, CloseStreamEffect)):
                source_ref = effect.token
            elif isinstance(effect, ExecutionEffect):
                source_ref = None
            else:
                source_ref = self._value_from_object(effect, "source", "target", "token")
            if token is not None:
                self._validate_execution_token(token, execution_ref)
            destination_ref = (
                effect.destination
                if isinstance(effect, (AddEdgeEffect, RemoveEdgeEffect))
                else None
                if isinstance(effect, ExecutionEffect)
                else self._value_from_object(effect, "destination")
            )
            source_port = (
                self._value_from_object(effect, "source_port", "output_port", "source_field", "port")
                if not isinstance(effect, ExecutionEffect)
                else None
            )
            if source_port is None:
                source_port = (
                    source_ref.field
                    if isinstance(source_ref, (EffectExecutionRef, EffectExecutionToken))
                    else self._value_from_object(source_ref, "field", "port", "output", "output_name")
                )
            destination_port = (
                self._value_from_object(effect, "destination_port", "input_port", "destination_field")
                if not isinstance(effect, ExecutionEffect)
                else None
            )
            if destination_port is None:
                destination_port = (
                    destination_ref.field
                    if isinstance(destination_ref, EffectExecutionRef)
                    else self._value_from_object(destination_ref, "field", "port", "input", "input_name")
                )
            effect_type = (
                effect.kind
                if isinstance(effect, ExecutionEffect)
                else self._value_from_object(effect, "edge_type", "connection_type", "kind")
            )
            is_loop_linkage = effect_type == "loop_linkage" or source_port == LOOP_LINKAGE_FIELD
            if source_port is not None:
                if not isinstance(source_port, str):
                    raise ValueError("Execution effect output port must be a string")
                if source_port in _RESERVED_EFFECT_PORTS:
                    raise ValueError(f"Execution effect references reserved output port '{source_port}'")
                is_activation = (
                    effect_kind == "emit"
                    and (
                        token.token_kind
                        if isinstance(token, EffectExecutionToken)
                        else self._value_from_object(token, "token_kind")
                    )
                    == "activation"
                )
                if is_activation:
                    activation_fields = getattr(type(node), "execution_activation_fields", frozenset())
                    if source_port not in activation_fields:
                        raise ValueError(f"Execution effect references unknown activation port '{source_port}'")
                    activation_value = effect.value if isinstance(effect, EmitEffect) else None
                    if not isinstance(effect, ExecutionEffect):
                        activation_value = self._value_from_object(effect, "value")
                    if activation_value is None:
                        activation_value = (
                            token.value
                            if isinstance(token, EffectExecutionToken)
                            else self._value_from_object(token, "value")
                        )
                    if activation_value != source_port:
                        raise ValueError(
                            f"Execution effect activation value '{activation_value}' does not match port '{source_port}'"
                        )
                    if isinstance(node, IfInvocation):
                        resolved_branch = self._resolved_if_exec_branches.get(execution_ref.exec_node_id)
                        if resolved_branch is None:
                            raise ValueError(
                                f"Execution effect activation for If execution node {execution_ref.exec_node_id} "
                                "has no resolved branch"
                            )
                        if source_port != resolved_branch:
                            raise ValueError(
                                f"Execution effect activation port '{source_port}' does not match resolved If branch "
                                f"'{resolved_branch}'"
                            )
                elif source_port not in output_fields:
                    raise ValueError(f"Execution effect references unknown output port '{source_port}'")
                if is_loop_linkage:
                    if source_port != LOOP_LINKAGE_FIELD or destination_port != LOOP_LINKAGE_FIELD:
                        raise ValueError("Loop-linkage effect must use loop_linkage ports")
                elif source_port == LOOP_LINKAGE_FIELD:
                    raise ValueError("Association edge cannot be stored as data effect")
            if destination_port is not None:
                destination_node_id = (
                    None
                    if isinstance(effect, ExecutionEffect)
                    else self._value_from_object(effect, "destination_node_id", "target_node_id", "consumer_node_id")
                )
                if destination_node_id is None:
                    destination_node_id = (
                        destination_ref.node_id
                        if isinstance(destination_ref, EffectExecutionRef)
                        else self._value_from_object(
                            destination_ref, "exec_node_id", "prepared_node_id", "node_id", "invocation_id"
                        )
                    )
                if destination_node_id is None:
                    raise ValueError("Execution effect destination port has no destination node")
                destination_node = self.execution_graph.nodes.get(destination_node_id)
                if destination_node is None:
                    raise ValueError("Execution effect destination node is not prepared")
                if not isinstance(destination_port, str):
                    raise ValueError("Execution effect input port must be a string")
                if destination_port not in type(destination_node).model_fields:
                    raise ValueError(f"Execution effect references unknown input port '{destination_port}'")
                if is_loop_linkage:
                    source_source_id = self.prepared_source_mapping[execution_ref.exec_node_id]
                    destination_source_id = self.prepared_source_mapping.get(destination_node_id)
                    linkage_edges = self.graph._get_loop_linkage_edges(source_source_id)
                    if not any(
                        edge.destination.node_id == destination_source_id
                        and edge.source.field == LOOP_LINKAGE_FIELD
                        and edge.destination.field == LOOP_LINKAGE_FIELD
                        for edge in linkage_edges
                    ):
                        raise ValueError("Execution effect loop linkage does not match graph metadata")
                elif effect_kind not in {"add_edge", "remove_edge"}:
                    if not any(
                        edge.source.node_id == execution_ref.exec_node_id
                        and edge.source.field == source_port
                        and edge.destination.node_id == destination_node_id
                        and edge.destination.field == destination_port
                        for edge in self.execution_graph.edges
                        if edge.type == "default"
                    ):
                        raise ValueError("Execution effect ports do not match prepared graph edge")

        if lifecycle_effect_kinds and not isinstance(node, CallSavedWorkflowInvocation):
            raise ValueError("Lifecycle effects are only supported by saved-workflow calls")

        if len(spawned_child_ids) > 1:
            raise ValueError("Execution reference may spawn only one child dependency")
        if len(awaited_child_ids) > 1:
            raise ValueError("Execution reference may await only one child dependency")
        if awaited_child_ids and spawned_child_ids and awaited_child_ids[0] not in spawned_child_ids:
            raise ValueError("Await effect does not match spawned child execution")
        if "fail" in lifecycle_effect_kinds and len(lifecycle_effect_kinds) != 1:
            raise ValueError("Fail effect cannot share execution reference with another lifecycle effect")
        if "await" in lifecycle_effect_kinds and "spawn_execution" not in lifecycle_effect_kinds:
            raise ValueError("Await effect requires a spawn effect")

        if require_continuation and isinstance(node, ForInvocation) and "start" not in continuation_operations:
            raise ValueError("For execution must include a continuation effect")
        if require_continuation and isinstance(node, ForReturnInvocation) and "complete" not in continuation_operations:
            raise ValueError("ForReturn execution must include a continuation effect")

        if not duplicate_continuations:
            return effects

        seen_continuations: set[tuple[str, str]] = set()
        unique_effects: list[Any] = []
        for effect in effects:
            if isinstance(effect, ContinuationEffect) or (
                not isinstance(effect, ExecutionEffect)
                and self._value_from_object(effect, "kind", "effect_type", "type") == "continuation"
            ):
                continuation_key = (
                    (effect.operation, effect.continuation_kind)
                    if isinstance(effect, ContinuationEffect)
                    else (
                        self._value_from_object(effect, "operation"),
                        self._value_from_object(effect, "continuation_kind"),
                    )
                )
                if continuation_key in seen_continuations:
                    continue
                seen_continuations.add(continuation_key)
            unique_effects.append(effect)
        return unique_effects

    def _build_execution_tokens(
        self, execution_ref: ExecutionReference, output: BaseInvocationOutput, effects: Iterable[Any] = ()
    ) -> dict[str, ExecutionToken]:
        tokens: dict[str, ExecutionToken] = {}
        output_fields = type(output).model_fields
        for port in output_fields:
            if port in {"type", "output_meta", LOOP_LINKAGE_FIELD}:
                continue
            token_id = f"{execution_ref.reference_id}:{port}"
            tokens[token_id] = ExecutionToken(
                token_id=token_id,
                reference_id=execution_ref.reference_id,
                owner_node_id=execution_ref.exec_node_id,
                port=port,
                frame=execution_ref.frame,
                value=getattr(output, port),
            )
        for effect in effects:
            effect_kind = (
                effect.kind
                if isinstance(effect, ExecutionEffect)
                else self._value_from_object(effect, "kind", "effect_type", "type")
            )
            if effect_kind not in {"emit", "close_stream"}:
                continue
            token = (
                effect.token
                if isinstance(effect, (EmitEffect, CloseStreamEffect))
                else self._value_from_object(effect, "token")
            )
            port = (
                token.field
                if isinstance(token, EffectExecutionToken)
                else self._value_from_object(token, "field", "port", "output", "output_name")
            )
            if not isinstance(port, str):
                raise ValueError("Execution effect output port must be a string")
            if port in _RESERVED_EFFECT_PORTS:
                raise ValueError(f"Execution effect references reserved output port '{port}'")
            token_node_id = (
                token.node_id
                if isinstance(token, EffectExecutionToken)
                else self._value_from_object(
                    token, "node_id", "invocation_id", "source_node_id", "owner_node_id", "exec_node_id"
                )
            )
            if token_node_id != execution_ref.exec_node_id:
                raise ValueError("Execution token is not owned by execution reference")
            token_value = (
                effect.value
                if isinstance(effect, EmitEffect)
                else None
                if isinstance(effect, ExecutionEffect)
                else self._value_from_object(effect, "value")
            )
            if token_value is None:
                token_value = (
                    token.value if isinstance(token, EffectExecutionToken) else self._value_from_object(token, "value")
                )
            token_kind = (
                token.token_kind
                if isinstance(token, EffectExecutionToken)
                else self._value_from_object(token, "token_kind")
            ) or "data"
            sequence = (
                token.sequence
                if isinstance(token, EffectExecutionToken)
                else self._value_from_object(token, "sequence")
            )
            if effect_kind == "close_stream":
                token_id_base = (
                    f"{execution_ref.reference_id}:{port}:stream_end:{sequence if sequence is not None else 'effect'}"
                )
            elif token_kind == "activation":
                # Compatibility If scheduling may already have lowered the same decision. Reuse its durable identity
                # so the invocation-declared effect replaces, rather than duplicates, the activation token.
                token_id_base = f"{execution_ref.reference_id}:activation:{port}"
            else:
                token_id_base = f"{execution_ref.reference_id}:{port}:{sequence if sequence is not None else 'effect'}"
            token_id = token_id_base
            duplicate_index = 1
            while token_id in tokens:
                token_id = f"{token_id_base}:{duplicate_index}"
                duplicate_index += 1
            tokens[token_id] = ExecutionToken(
                token_id=token_id,
                reference_id=execution_ref.reference_id,
                owner_node_id=execution_ref.exec_node_id,
                port=port,
                frame=execution_ref.frame,
                value=token_value,
                token_kind="stream_end" if effect_kind == "close_stream" else token_kind,
                sequence=sequence,
            )
        return tokens

    @classmethod
    def _lifecycle_effect_kinds(cls, effects: Iterable[Any]) -> set[str]:
        lifecycle_kinds = {"spawn_execution", "await", "fail"}
        return {
            effect_kind
            for effect in effects
            if (
                effect_kind := (
                    effect.kind
                    if isinstance(effect, ExecutionEffect)
                    else cls._value_from_object(effect, "kind", "effect_type", "type")
                )
            )
            in lifecycle_kinds
        }

    @classmethod
    def _is_pending_lifecycle_effects(cls, effects: Iterable[Any]) -> bool:
        effect_kinds = cls._lifecycle_effect_kinds(effects)
        return bool(effect_kinds & {"spawn_execution", "await"}) and "fail" not in effect_kinds

    def _is_pending_lifecycle_execution(self, exec_node_id: str, effects: Iterable[Any]) -> bool:
        return exec_node_id not in self.executed and self._is_pending_lifecycle_effects(effects)

    def _update_pending_lifecycle_execution(self, exec_node_id: str, effects: Iterable[Any]) -> None:
        if self._is_pending_lifecycle_execution(exec_node_id, effects):
            self._pending_lifecycle_execution_nodes.add(exec_node_id)
        else:
            self._pending_lifecycle_execution_nodes.discard(exec_node_id)

    def apply(
        self,
        execution_ref: ExecutionReference | str | Any,
        output: BaseInvocationOutput | Any = None,
        effects: Optional[Iterable[Any]] = None,
        *,
        effect_count: Optional[int] = None,
        _compatibility_completion: bool = False,
    ) -> list[tuple[BaseInvocation, BaseInvocationOutput]]:
        """Apply output/effects through current scheduler while retaining old ``complete()`` behavior."""

        ref = self._validate_execution_ref(execution_ref)
        persisted_effects = self.execution_effects.get(ref.reference_id)
        pending_resume = persisted_effects is not None and self._is_pending_lifecycle_execution(
            ref.exec_node_id, persisted_effects
        )
        if ref.exec_node_id in self.executed or (persisted_effects is not None and not pending_resume):
            raise ValueError(f"Execution reference {ref.reference_id} has already been applied")
        if pending_resume and self.is_waiting_on_workflow_call():
            raise ValueError(f"Execution reference {ref.reference_id} is waiting on a child dependency")
        if isinstance(output, BaseInvocationOutput):
            result_effects = None
            result_output = None
        else:
            result_effects = self._value_from_object(output, "effects", "effect_batch")
            result_output = self._value_from_object(output, "output", "invocation_output", "result")
            if result_output is not None:
                output = result_output
                if effects is None:
                    effects = result_effects
        output_value = self._validate_output_owner(ref, output)
        self._validate_for_continuation_output(ref, output_value, allow_return_state_override=_compatibility_completion)
        require_continuation = effects is not None or effect_count is not None
        if pending_resume and effects is None:
            effect_values = list(persisted_effects)
        elif effects is None:
            effect_values = []
            if effect_count is None and isinstance(
                self.execution_graph.get_node(ref.exec_node_id), (ForInvocation, ForReturnInvocation)
            ):
                effect_values = [self._build_compatibility_continuation_effect(ref, output_value)]
                require_continuation = True
        else:
            batch_values = self._value_from_object(effects, "effects")
            effect_values = list(batch_values if batch_values is not None else effects)
        if pending_resume and effects is not None and list(persisted_effects) != effect_values:
            raise ValueError("Pending lifecycle effects do not match resumed execution")
        submitted_effect_values = effect_values
        effect_values = self._validate_effects(
            ref,
            effect_values,
            effect_count,
            require_continuation=require_continuation,
        )
        # Keep compatibility callers' raw effect records in the durable ledger. Runtime operations use the
        # normalized models returned above; snapshot loading normalizes persisted records once at its boundary.
        persisted_effect_values = submitted_effect_values
        if len(effect_values) != len(submitted_effect_values):
            persisted_effect_values = effect_values
        persisted_effects = copydeep(persisted_effect_values)

        # All validation above is side-effect free. Preserve complete() as the scheduler compatibility boundary,
        # but make the scheduler transition and ledger update one atomic operation. The journal records only
        # containers and object attributes touched by the scheduler, avoiding a full-state deep copy per apply.
        transaction = _ApplyTransaction()
        object.__setattr__(self, "_apply_transaction", transaction)
        try:
            lifecycle_effect_kinds = self._lifecycle_effect_kinds(effect_values)
            if lifecycle_effect_kinds and not pending_resume:
                self._tx_set_mapping(self.execution_refs, ref.exec_node_id, ref)
                self._tx_set_mapping(self.execution_effects, ref.reference_id, persisted_effects)
                self._update_pending_lifecycle_execution(ref.exec_node_id, effect_values)
                if "fail" in lifecycle_effect_kinds:
                    failure = next(
                        str(
                            effect.message
                            if isinstance(effect, FailEffect)
                            else self._value_from_object(effect, "message")
                        )
                        for effect in effect_values
                        if (
                            effect.kind
                            if isinstance(effect, ExecutionEffect)
                            else self._value_from_object(effect, "kind", "effect_type", "type")
                        )
                        == "fail"
                    )
                    self._tx_set_mapping(self.errors, ref.exec_node_id, failure)
                return []
            # Capture and record the continuation before scheduler completion can clear the prepared For
            # collection or otherwise mutate the node used to validate its durable payload.
            self._record_continuation_effects(ref, effect_values, allow_return_state_override=_compatibility_completion)
            finalized_outputs = self._complete(ref.exec_node_id, output_value)
            tokens = self._build_execution_tokens(ref, output_value, effect_values)
            self._record_effect_streams(ref, effect_values)
            ref.effect_count = len(effect_values)
            self._tx_set_mapping(self.execution_refs, ref.exec_node_id, ref)
            for token_id, token in tokens.items():
                self._tx_set_mapping(self.execution_tokens, token_id, token)
            self._tx_set_mapping(self.execution_effects, ref.reference_id, persisted_effects)
            self._update_pending_lifecycle_execution(ref.exec_node_id, persisted_effects)
            return finalized_outputs
        except Exception:
            try:
                transaction.rollback()
            finally:
                self._reset_apply_derived_caches()
                self._rehydrate_ready_queues()
            raise
        finally:
            object.__setattr__(self, "_apply_transaction", None)

    def _invalidate_loop_caches_for_source(self, source_node_id: str) -> None:
        self._all_for_contexts_finalized_cache.pop(source_node_id, None)

    def _invalidate_loop_caches_for_exec_node(self, exec_node_id: str) -> None:
        source_node_id = self.prepared_source_mapping.get(exec_node_id)
        if source_node_id is not None:
            self._prepared_if_exec_indexed_sources.discard(source_node_id)
            for key in tuple(self._prepared_if_exec_ids_by_source_and_frame):
                if key[0] == source_node_id:
                    self._prepared_if_exec_ids_by_source_and_frame.pop(key, None)
            self._prepared_if_exec_frames_by_source.pop(source_node_id, None)
            self._invalidate_loop_caches_for_source(source_node_id)
            prepared_node = self.execution_graph.nodes.get(exec_node_id)
            if isinstance(prepared_node, ForInvocation):
                parent_iteration_path = self._get_for_parent_iteration_path(exec_node_id)
                cached_paths = self._for_parent_iteration_paths_cache.get(source_node_id)
                if cached_paths is not None:
                    cached_paths.add(parent_iteration_path)
        self._update_prepared_for_index(exec_node_id)

    def _get_prepared_exec_metadata(self, exec_node_id: str) -> _PreparedExecNodeMetadata:
        return self._prepared_registry().get_metadata(exec_node_id)

    def _set_prepared_exec_state(self, exec_node_id: str, state: PreparedExecState) -> None:
        self._prepared_registry().set_state(exec_node_id, state)

    def _get_iteration_path(self, exec_node_id: str) -> tuple[int, ...]:
        return self._runtime().get_iteration_path(exec_node_id)

    def _get_for_parent_iteration_path(self, exec_node_id: str) -> tuple[int, ...]:
        iteration_path = self._get_iteration_path(exec_node_id)
        node = self.execution_graph.get_node(exec_node_id)
        if isinstance(node, ForInvocation) and node.index == -1:
            return iteration_path
        return iteration_path[:-1]

    def _get_prepared_for_index(self) -> dict[tuple[str, tuple[int, ...]], str]:
        """Index For executions by exact iteration path and final candidates by parent context.

        The scheduler creates For executions in iteration order, so the highest index in a context is the current
        final candidate. Indexes are built lazily once and maintained as prepared paths are registered; repeated
        completion checks then use constant-time context lookups.
        """
        if self._prepared_for_index is not None:
            return self._prepared_for_index

        index: dict[tuple[str, tuple[int, ...]], str] = {}
        self._prepared_for_index_by_exec = {}
        self._final_prepared_for_index = {}
        for source_for_id, prepared_ids in self.source_prepared_mapping.items():
            for prepared_for_id in prepared_ids:
                self._prepared_for_index_by_exec[prepared_for_id] = (
                    source_for_id,
                    self._get_iteration_path(prepared_for_id),
                    self._get_for_parent_iteration_path(prepared_for_id),
                )

        self._prepared_for_index = index
        for prepared_for_id in list(self._prepared_for_index_by_exec):
            self._update_prepared_for_index(prepared_for_id)
        assert self._prepared_for_index is not None
        return self._prepared_for_index

    def _update_prepared_for_index(self, prepared_for_id: str) -> None:
        if self._prepared_for_index is None:
            return

        old_key = self._prepared_for_index_by_exec.pop(prepared_for_id, None)
        if old_key is not None:
            old_source_id, old_path, old_parent_path = old_key
            if self._prepared_for_index.get((old_source_id, old_path)) == prepared_for_id:
                del self._prepared_for_index[(old_source_id, old_path)]
            if (
                self._final_prepared_for_index is not None
                and self._final_prepared_for_index.get((old_source_id, old_parent_path)) == prepared_for_id
            ):
                del self._final_prepared_for_index[(old_source_id, old_parent_path)]

        prepared_for_node = self.execution_graph.nodes.get(prepared_for_id)
        source_for_id = self.prepared_source_mapping.get(prepared_for_id)
        if not isinstance(prepared_for_node, ForInvocation) or source_for_id is None:
            return

        iteration_path = self._get_iteration_path(prepared_for_id)
        parent_iteration_path = self._get_for_parent_iteration_path(prepared_for_id)
        key = (source_for_id, iteration_path)
        existing_id = self._prepared_for_index.get(key)
        if existing_id is None:
            self._prepared_for_index[key] = prepared_for_id
        else:
            existing_node = self.execution_graph.get_node(existing_id)
            if isinstance(existing_node, ForInvocation) and prepared_for_node.index > existing_node.index:
                self._prepared_for_index[key] = prepared_for_id
        assert self._final_prepared_for_index is not None
        final_key = (source_for_id, parent_iteration_path)
        final_existing_id = self._final_prepared_for_index.get(final_key)
        if final_existing_id is None:
            self._final_prepared_for_index[final_key] = prepared_for_id
        else:
            final_existing_node = self.execution_graph.get_node(final_existing_id)
            if isinstance(final_existing_node, ForInvocation) and prepared_for_node.index > final_existing_node.index:
                self._final_prepared_for_index[final_key] = prepared_for_id
        self._prepared_for_index_by_exec[prepared_for_id] = (source_for_id, iteration_path, parent_iteration_path)

    def _mark_loop_context_finalized(self, source_for_id: str, prepared_for_id: str) -> None:
        parent_iteration_path = self._get_for_parent_iteration_path(prepared_for_id)
        self._tx_add_set(self.finalized_loop_contexts, (source_for_id, parent_iteration_path))
        self._all_for_contexts_finalized_cache.pop(source_for_id, None)

    def _mark_for_source_complete(self, source_for_id: str) -> None:
        if not self._all_for_contexts_finalized(source_for_id):
            return

        source_node_ids = {source_for_id}
        body_path_to_return = self.graph._get_for_body_path_to_return(source_for_id, self._get_source_graph_flat())
        if body_path_to_return is not None:
            body_path_nodes, _return_node_id = body_path_to_return
            source_node_ids.update(body_path_nodes)

        for source_node_id in source_node_ids:
            if source_node_id not in self.executed:
                self._mark_source_executed(source_node_id)

    def _get_for_parent_iteration_paths(self, source_for_id: str) -> set[tuple[int, ...]]:
        cached = self._for_parent_iteration_paths_cache.get(source_for_id)
        if cached is not None:
            return cached

        paths = {
            self._get_for_parent_iteration_path(prepared_for_id)
            for prepared_for_id in self._prepared_registry().get_prepared_ids(source_for_id)
            if isinstance(self.execution_graph.get_node(prepared_for_id), ForInvocation)
        }
        self._for_parent_iteration_paths_cache[source_for_id] = paths
        return paths

    def _is_loop_context_finalized(self, source_for_id: str, parent_iteration_path: tuple[int, ...]) -> bool:
        return (source_for_id, parent_iteration_path) in self.finalized_loop_contexts

    def _all_for_contexts_finalized(self, source_for_id: str) -> bool:
        cached = self._all_for_contexts_finalized_cache.get(source_for_id)
        if cached is not None:
            return cached

        parent_iteration_paths = self._get_for_parent_iteration_paths(source_for_id)
        finalized = (bool(parent_iteration_paths) or source_for_id in self.executed) and all(
            self._is_loop_context_finalized(source_for_id, parent_iteration_path)
            for parent_iteration_path in parent_iteration_paths
        )
        self._all_for_contexts_finalized_cache[source_for_id] = finalized
        return finalized

    def _queue_for(self, cls_name: str) -> Deque[str]:
        return self._scheduler().queue_for(cls_name)

    def _is_deferred_by_unresolved_if(self, exec_node_id: str) -> bool:
        return graph_if_runtime._is_deferred_by_unresolved_if(self, exec_node_id)

    def _has_rejected_activation_dependency(self, exec_node_id: str) -> bool:
        return graph_if_runtime._has_rejected_activation_dependency(self, exec_node_id)

    def _get_activation_dependencies(self, exec_node_id: str) -> tuple[ActivationDependency, ...]:
        return graph_if_runtime._get_activation_dependencies(self, exec_node_id)

    def _try_resolve_if_node(self, exec_node_id: str, *, enqueue: bool = True) -> None:
        scheduler = self._execution_scheduler
        if isinstance(scheduler, _GenericGraphSchedulerAdapter):
            scheduler.resolve_if_node(exec_node_id, enqueue=enqueue)
            return

        if exec_node_id in self._resolved_if_exec_branches:
            return
        node = self.execution_graph.get_node(exec_node_id)
        if not isinstance(node, IfInvocation) or not self._apply_if_condition_inputs(exec_node_id, node):
            return

        selected_field = "true_input" if node.condition else "false_input"
        self._resolve_activation_gate(exec_node_id, selected_field)
        self._tx_set_mapping(self._resolved_if_exec_branches, exec_node_id, selected_field)
        self._record_compatibility_activation_token(exec_node_id, selected_field)
        assert isinstance(scheduler, _ExecutionScheduler)
        scheduler._discard_rejected_activation_nodes()
        scheduler._enqueue_activation_ready_nodes()
        if enqueue:
            self._enqueue_if_ready(exec_node_id)

    def _is_pending_if(self, exec_node_id: str) -> bool:
        node = self.execution_graph.nodes.get(exec_node_id)
        if not isinstance(node, IfInvocation):
            return False
        selected_field = self._resolved_if_exec_branches.get(exec_node_id)
        if selected_field is None:
            return True
        source_node_id = self.prepared_source_mapping.get(exec_node_id)
        if source_node_id is None or not self.graph._get_input_edges(source_node_id, selected_field):
            return False
        return not any(
            edge.destination.field == selected_field for edge in self.execution_graph._get_input_edges(exec_node_id)
        )

    def set_ready_order(self, order: Iterable[Type[BaseInvocation] | str]) -> None:
        names: list[str] = []
        for x in order:
            names.append(x.__name__ if hasattr(x, "__name__") else str(x))
        self.ready_order = names
        if self._generic_graph_scheduler is not None:
            self._generic_graph_scheduler.set_ready_order(names)

    def _enqueue_if_ready(self, nid: str) -> None:
        self._scheduler().enqueue_if_ready(nid)

    def _prepare_until_node_ready(self) -> Optional[BaseInvocation]:
        base_graph = self._get_source_graph_flat()
        if self._pending_if_exec_nodes:
            self._materializer()._attach_pending_if_inputs()
        self._rehydrate_ready_queues()
        next_node = self._get_next_node()
        if next_node is not None:
            return next_node

        if (
            isinstance(self._scheduler(), _GenericGraphSchedulerAdapter)
            and self._can_use_direct_iterate_collect_planner()
        ):
            self._prepare_direct_iterate_collect()
            return self._get_next_node()
        if (
            isinstance(self._scheduler(), _GenericGraphSchedulerAdapter)
            and (sequence_depth := get_nested_iterate_sequence_depth(self)) is not None
        ):
            prepare_nested_iterate_sequence(self, depth=sequence_depth)
            return self._get_next_node()

        planner = (
            self._for_planner()
            if isinstance(self._scheduler(), _GenericGraphSchedulerAdapter) and self._can_use_generic_for_scheduler()
            else self._materializer()
        )
        prepared_id = planner.prepare(base_graph)

        while prepared_id is not None:
            prepared_id = planner.prepare(base_graph)
            if next_node is None:
                next_node = self._get_next_node()

        return next_node

    def _get_direct_iterate_collect_fan_in(
        self, iterator_ids: list[str], collector_id: str
    ) -> Optional[_DirectIterateCollectFanIn]:
        return _get_direct_iterate_collect_fan_in(self, iterator_ids, collector_id)

    def _get_body_iterate_collect_fan_in(
        self, iterator_ids: list[str], collector_id: str
    ) -> Optional[_BodyIterateCollectFanIn]:
        return _get_body_iterate_collect_fan_in(self, iterator_ids, collector_id)

    def _get_direct_iterate_collect_nodes(
        self,
    ) -> Optional[tuple[str, str, str, str, tuple[str, ...]] | _DirectIterateCollectFanIn | _BodyIterateCollectFanIn]:
        return _get_direct_iterate_collect_nodes(self)

    def _can_use_direct_iterate_collect_planner(self) -> bool:
        return _can_use_direct_iterate_collect_planner(self)

    def _create_direct_execution_node_copy(
        self, source_node_id: str, iteration_index: int = -1, iteration_path: tuple[int, ...] = ()
    ) -> BaseInvocation:
        return _create_direct_execution_node_copy(self, source_node_id, iteration_index, iteration_path)

    def _attach_direct_execution_edges(self, exec_node_id: str, edges: Iterable[Edge]) -> list[Edge]:
        return _attach_direct_execution_edges(self, exec_node_id, edges)

    def _initialize_direct_execution_node(
        self, exec_node_id: str, input_edges: Iterable[Edge], *, project: bool = True
    ) -> None:
        _initialize_direct_execution_node(self, exec_node_id, input_edges, project=project)

    def _mark_direct_source_empty(self, source_node_id: str) -> None:
        _mark_direct_source_empty(self, source_node_id)

    def _prepare_direct_iterate_collect_fan_in_unchecked(self, fan_in: _DirectIterateCollectFanIn) -> None:
        _prepare_direct_iterate_collect_fan_in_unchecked(self, fan_in)

    def _prepare_body_iterate_collect_fan_in_unchecked(self, fan_in: _BodyIterateCollectFanIn) -> None:
        _prepare_body_iterate_collect_fan_in_unchecked(self, fan_in)

    def _prepare_direct_iterate_collect(self) -> None:
        _prepare_direct_iterate_collect(self)

    def _prepare_direct_iterate_collect_unchecked(self) -> None:
        _prepare_direct_iterate_collect_unchecked(self)

    def _reset_runtime_caches(self) -> None:
        self._ready_queues = {}
        self._ready_node_ids = set()
        self._active_class = None
        self._resolved_if_exec_branches = {}
        self._pending_if_exec_nodes = set()
        self._if_branch_sources_cache = {}
        self._prepared_exec_metadata = {}
        self._prepared_exec_registry = None
        self._execution_materializer = None
        self._generic_for_planner = None
        self._execution_scheduler = None
        self._generic_graph_scheduler = None
        self._execution_runtime = None
        self._if_activation_controller_instance = None
        self._if_activation_dependencies_by_source = {}
        self._if_activation_dependencies_by_exec = {}
        self._prepared_if_exec_ids_by_source_and_frame = {}
        self._prepared_if_exec_indexed_sources = set()
        self._source_graph_flat = None
        self._execution_graph_flat = None
        self._completed_source_ids_cache = None
        self._unexecuted_prepared_counts = None
        self._for_parent_iteration_paths_cache = {}
        self._all_for_contexts_finalized_cache = {}
        self._prepared_for_index = None
        self._final_prepared_for_index = None
        self._prepared_for_index_by_exec = {}
        self.__pydantic_private__["_runtime_state"].fresh_flat_if_activation = None

    def _rehydrate_prepared_exec_metadata(self) -> None:
        registry = self._prepared_registry()
        for exec_node_id, source_node_id in self.prepared_source_mapping.items():
            metadata = registry.get_metadata(exec_node_id)
            metadata.source_node_id = source_node_id
            iteration_path = registry.get_iteration_path(exec_node_id)
            if iteration_path is None:
                iteration_path = self._get_iteration_path(exec_node_id)
            metadata.iteration_path = iteration_path
            if exec_node_id in self.executed:
                metadata.state = "executed" if exec_node_id in self.results else "skipped"
            elif self.indegree.get(exec_node_id) == 0:
                metadata.state = "ready"
            else:
                metadata.state = "pending"

    def _apply_if_condition_inputs(self, exec_node_id: str, node: IfInvocation) -> bool:
        condition_edges = self.execution_graph._get_input_edges(exec_node_id, "condition")
        if any(edge.source.node_id not in self.executed for edge in condition_edges):
            return False

        for edge in condition_edges:
            self._tx_set_attr(
                node,
                edge.destination.field,
                copydeep(getattr(self.results[edge.source.node_id], edge.source.field)),
            )
        return True

    def _validate_persisted_activation_tokens(self) -> None:
        for token_key, token in self.execution_tokens.items():
            if token.token_kind != "activation":
                continue

            if token.owner_node_id not in self.prepared_source_mapping:
                raise ValueError("Activation token has an unknown execution owner")
            expected_ref = self._expected_execution_ref(token.owner_node_id)
            owner = self.execution_graph.get_node(token.owner_node_id)

            if token.reference_id != expected_ref.reference_id:
                raise ValueError("Activation token belongs to another execution reference")
            if (
                token.frame.state_id != expected_ref.frame.state_id
                or token.frame.frame_id != expected_ref.frame.frame_id
                or token.frame.iteration_path != expected_ref.frame.iteration_path
                or token.frame.workflow_call_depth != expected_ref.frame.workflow_call_depth
            ):
                raise ValueError("Activation token belongs to another execution frame")
            if token_key != token.token_id:
                raise ValueError("Activation token mapping key does not match token id")
            if not isinstance(owner, IfInvocation):
                activation_fields = getattr(type(owner), "execution_activation_fields", frozenset())
                if token.port not in activation_fields:
                    raise ValueError(f"Activation token references unknown activation port '{token.port}'")
                if token.value != token.port:
                    raise ValueError(f"Activation token value '{token.value}' does not match port '{token.port}'")
                expected_token_id = f"{expected_ref.reference_id}:activation:{token.port}"
                if token.token_id != expected_token_id:
                    raise ValueError("Activation token has a stale token id")

    def _rehydrate_resolved_if_exec_branches(self) -> None:
        self._validate_persisted_activation_tokens()
        activation_tokens_by_owner: dict[str, list[ExecutionToken]] = {}
        for token in self.execution_tokens.values():
            if token.token_kind == "activation":
                activation_tokens_by_owner.setdefault(token.owner_node_id, []).append(token)
        for exec_node_id, node in self.execution_graph.nodes.items():
            if not isinstance(node, IfInvocation):
                continue

            activation_tokens = activation_tokens_by_owner.get(exec_node_id, [])
            if activation_tokens:
                expected_ref = self._expected_execution_ref(exec_node_id)
                selected_fields: set[str] = set()
                for activation_token in activation_tokens:
                    selected_field = activation_token.port
                    if selected_field not in ("true_input", "false_input"):
                        raise ValueError(f"Invalid activation token for If execution node {exec_node_id}")
                    if activation_token.reference_id != expected_ref.reference_id:
                        raise ValueError(f"Activation token for If execution node {exec_node_id} has a stale reference")
                    if activation_token.owner_node_id != exec_node_id:
                        raise ValueError(f"Activation token for If execution node {exec_node_id} has a stale owner")
                    if activation_token.value != selected_field:
                        raise ValueError(f"Activation token for If execution node {exec_node_id} has a stale value")
                    self._validate_execution_token(activation_token, expected_ref)
                    selected_fields.add(selected_field)
                if len(selected_fields) != 1:
                    raise ValueError(f"If execution node {exec_node_id} has conflicting activation tokens")
                selected_field = next(iter(selected_fields))
                expected_token_id = f"{expected_ref.reference_id}:activation:{selected_field}"
                if any(token.token_id != expected_token_id for token in activation_tokens):
                    raise ValueError(f"Activation token for If execution node {exec_node_id} has a stale token id")
                self._resolve_activation_gate(exec_node_id, selected_field)
                self._resolved_if_exec_branches[exec_node_id] = selected_field
                if self._is_pending_if(exec_node_id):
                    self._pending_if_exec_nodes.add(exec_node_id)
                continue

            if not self._apply_if_condition_inputs(exec_node_id, node):
                self._pending_if_exec_nodes.add(exec_node_id)
                continue

            selected_field = "true_input" if node.condition else "false_input"
            self._resolve_activation_gate(exec_node_id, selected_field)
            self._resolved_if_exec_branches[exec_node_id] = selected_field
            self._record_compatibility_activation_token(exec_node_id, selected_field)
            if self._is_pending_if(exec_node_id):
                self._pending_if_exec_nodes.add(exec_node_id)

    def _rehydrate_execution_refs(self) -> None:
        for exec_node_id in self.prepared_source_mapping:
            existing = self.execution_refs.get(exec_node_id)
            expected = self._expected_execution_ref(
                exec_node_id,
                effect_count=existing.effect_count if existing is not None else None,
            )
            if existing is not None:
                if (
                    existing.reference_id not in ("", expected.reference_id)
                    or existing.state_id not in ("", expected.state_id)
                    or existing.exec_node_id not in ("", expected.exec_node_id)
                    or existing.source_node_id not in ("", expected.source_node_id)
                    or existing.frame != expected.frame
                ):
                    raise ValueError("Persisted execution reference does not belong to this execution frame")
            self.execution_refs[exec_node_id] = expected

    def _synthesize_legacy_execution_effects(self) -> None:
        """Upgrade completed legacy For results with the v1 continuation records needed for the next dump."""

        if not self._legacy_snapshot_loaded or all(
            source_node_id in self.executed for source_node_id in self.graph.nodes
        ):
            return
        for exec_node_id, output in self.results.items():
            execution_ref = self.execution_refs.get(exec_node_id)
            if execution_ref is None or execution_ref.reference_id in self.execution_effects:
                continue
            if not self._requires_continuation_effect(execution_ref):
                continue
            if not isinstance(output, (ForInvocationOutput, ForReturnInvocationOutput)):
                continue
            effect = self._build_compatibility_continuation_effect(execution_ref, output)
            self.execution_effects[execution_ref.reference_id] = [effect]
            execution_ref.effect_count = 1

    def _legacy_execution_effects_for_snapshot(self) -> dict[str, list[Any]]:
        """Return missing loop effects needed to convert any loaded legacy state on its next dump."""

        if not self._legacy_snapshot_loaded:
            return {}
        effects: dict[str, list[Any]] = {}
        for exec_node_id, output in self.results.items():
            execution_ref = self.execution_refs.get(exec_node_id)
            if execution_ref is None or execution_ref.reference_id in self.execution_effects:
                continue
            if not self._requires_continuation_effect(execution_ref):
                continue
            if isinstance(output, (ForInvocationOutput, ForReturnInvocationOutput)):
                effects[execution_ref.reference_id] = [
                    self._build_compatibility_continuation_effect(execution_ref, output)
                ]
        return effects

    def _rehydrate_generic_runtime_state(self) -> None:
        """Rebuild private stream/continuation adapters from durable results."""

        iterate_stream_needed_cache: dict[str, bool] = {}

        def iterate_stream_needed(source_node_id: str) -> bool:
            """Keep an Iterate stream only while a prepared item or Collect consumer can still need it."""

            if source_node_id in iterate_stream_needed_cache:
                return iterate_stream_needed_cache[source_node_id]
            # Raw model snapshots predate the compact runtime projection and retain the legacy stream behavior.
            if self._legacy_execution_snapshot and not self._execution_effects_persisted:
                iterate_stream_needed_cache[source_node_id] = True
                return True
            if any(
                isinstance(self.graph.nodes.get(edge.destination.node_id), CollectInvocation)
                and edge.destination.node_id not in self.executed
                for edge in self.graph._get_output_edges(source_node_id, ITEM_FIELD)
            ):
                iterate_stream_needed_cache[source_node_id] = True
                return True
            prepared_ids = self.source_prepared_mapping.get(source_node_id, ())
            if any(exec_node_id not in self.executed for exec_node_id in prepared_ids):
                iterate_stream_needed_cache[source_node_id] = True
                return True
            if not prepared_ids:
                iterate_stream_needed_cache[source_node_id] = False
                return False
            needed = any(
                isinstance(self.execution_graph.nodes.get(edge.destination.node_id), CollectInvocation)
                and edge.destination.node_id not in self.executed
                for exec_node_id in prepared_ids
                for edge in self.execution_graph._get_output_edges(exec_node_id, ITEM_FIELD)
            )
            iterate_stream_needed_cache[source_node_id] = needed
            return needed

        for source_node_id, source_node in self.graph.nodes.items():
            if (
                isinstance(source_node, IterateInvocation)
                and source_node_id in self.executed
                and not self.source_prepared_mapping.get(source_node_id)
                and iterate_stream_needed(source_node_id)
            ):
                self._record_empty_iterate_stream(source_node_id)

        if not self._execution_effects_persisted and not self._legacy_execution_snapshot:
            raise ValueError("Execution effects ledger is missing from the current execution snapshot")

        if self._execution_effects_persisted:
            references_by_id = {ref.reference_id: ref for ref in self.execution_refs.values()}
            for reference_id, effects in self.execution_effects.items():
                execution_ref = references_by_id.get(reference_id)
                if execution_ref is None:
                    raise ValueError("Execution effects contain an unknown execution reference")
                lifecycle_pending = self._is_pending_lifecycle_execution(execution_ref.exec_node_id, effects)
                lifecycle_failed = "fail" in self._lifecycle_effect_kinds(effects)
                if execution_ref.exec_node_id not in self.results and not lifecycle_pending and not lifecycle_failed:
                    raise ValueError("Execution effects belong to a pending execution node")
                if execution_ref.exec_node_id not in self.executed and not lifecycle_pending and not lifecycle_failed:
                    raise ValueError("Execution effects require an executed marker")
                if lifecycle_pending:
                    waiting_frame = self.waiting_workflow_call
                    if waiting_frame is not None and waiting_frame.prepared_call_node_id != execution_ref.exec_node_id:
                        raise ValueError("Pending lifecycle effects belong to another workflow call")
                node = self.execution_graph.get_node(execution_ref.exec_node_id)
                effects = self._validate_effects(
                    execution_ref,
                    list(effects),
                    len(effects),
                    require_continuation=self._requires_continuation_effect(execution_ref),
                    validate_json=not self._execution_effects_persisted,
                )
                self._record_continuation_effects(execution_ref, effects)
                self._record_effect_streams(execution_ref, effects, trusted=self._execution_effects_persisted)
        elif self._legacy_execution_snapshot:
            for reference_id, effects in self.execution_effects.items():
                execution_ref = next(
                    (ref for ref in self.execution_refs.values() if ref.reference_id == reference_id),
                    None,
                )
                if execution_ref is not None:
                    effects = self._validate_effects(execution_ref, list(effects), None)
                    self._record_continuation_effects(execution_ref, effects)
                    self._record_effect_streams(execution_ref, effects)

        for exec_node_id, output in self.results.items():
            execution_ref = self.execution_refs.get(exec_node_id)
            if execution_ref is None:
                continue
            node = self.execution_graph.get_node(exec_node_id)
            if self._execution_effects_persisted and self._requires_continuation_effect(execution_ref):
                if execution_ref.reference_id not in self.execution_effects:
                    raise ValueError(f"{type(node).__name__} execution must include a continuation effect")
                effects = self.execution_effects[execution_ref.reference_id]
                self._validate_effects(
                    execution_ref,
                    list(effects),
                    len(effects),
                    require_continuation=True,
                    validate_json=not self._execution_effects_persisted,
                )
            if self._execution_effects_persisted:
                self._validate_for_continuation_output(execution_ref, output)

        iterate_results: list[tuple[str, IterateInvocationOutput]] = []
        for exec_node_id, output in self.results.items():
            if isinstance(output, IterateInvocationOutput):
                iterate_results.append((exec_node_id, output))
        iterate_results.sort(key=lambda item: (self._get_iteration_path(item[0]), item[1].index, item[0]))
        for exec_node_id, output in iterate_results:
            source_node_id = self.prepared_source_mapping.get(exec_node_id)
            if source_node_id is None or not iterate_stream_needed(source_node_id):
                continue
            self._record_iterate_stream(exec_node_id, output, prefer_existing=True, trusted=True)

        for exec_node_id, node in self.execution_graph.nodes.items():
            if isinstance(node, ForInvocation) and node.index >= 0 and exec_node_id in self.results:
                continuation = self._for_continuation(exec_node_id)
                source_node_id = self.prepared_source_mapping.get(exec_node_id)
                if source_node_id is not None and self._is_loop_context_finalized(
                    source_node_id, self._get_for_parent_iteration_path(exec_node_id)
                ):
                    if not continuation.terminal:
                        continuation.complete(self.results[exec_node_id].model_dump(mode="json"))

        self._register_generic_child_dependency()

    def _rehydrate_ready_queues(self) -> None:
        if self.has_error():
            return

        execution_graph = self._get_execution_graph_flat()
        for exec_node_id in nx.topological_sort(execution_graph):
            if exec_node_id in self.executed:
                continue
            if exec_node_id in self._pending_if_exec_nodes:
                continue
            if self.indegree.get(exec_node_id) != 0:
                continue
            self._enqueue_if_ready(exec_node_id)

    def _rehydrate_runtime_state(self) -> None:
        self._reset_runtime_caches()
        self._rehydrate_prepared_exec_metadata()
        self._rehydrate_resolved_if_exec_branches()
        self._rehydrate_generic_runtime_state()
        if self.has_error():
            self._clear_transient_runtime()
        self._rehydrate_ready_queues()
        if self._pending_if_exec_nodes:
            self._materializer()._attach_pending_if_inputs(enqueue=False)

    def model_post_init(self, __context: Any) -> None:
        if isinstance(__context, dict) and "execution_effects_persisted" in __context:
            self._execution_effects_persisted = __context["execution_effects_persisted"]
            self._legacy_execution_snapshot = __context.get("legacy_execution_snapshot", False)
            self._legacy_snapshot_loaded = self._legacy_execution_snapshot
        if isinstance(__context, dict) and __context.get("queue_read_projection"):
            return
        self._generic_child_dependencies = {
            dependency_id: ChildDependencyRecord.model_validate(dependency.model_dump(mode="python"))
            for dependency_id, dependency in self.execution_child_dependencies.items()
        }
        self.execution_child_dependencies.clear()
        self.execution_child_dependencies.update(self._generic_child_dependencies)
        self._rehydrate_execution_refs()
        self._synthesize_legacy_execution_effects()
        references_by_id = {reference.reference_id: reference for reference in self.execution_refs.values()}
        self._pending_lifecycle_execution_nodes = {
            reference.exec_node_id
            for reference_id, effects in self.execution_effects.items()
            if effects
            and (reference := references_by_id.get(reference_id)) is not None
            and reference.exec_node_id not in self.errors
            and self._is_pending_lifecycle_execution(reference.exec_node_id, effects)
        }
        self._rehydrate_runtime_state()

    model_config = ConfigDict(json_schema_extra=_hide_execution_state_runtime_fields)

    @field_validator("graph")
    def graph_is_valid(cls, v: Graph):
        """Validates that the graph is valid"""
        v.validate_self()
        return v

    def next(self) -> Optional[BaseInvocation]:
        """Gets the next node ready to execute."""

        # TODO: enable multiple nodes to execute simultaneously by tracking currently executing nodes
        #       possibly with a timeout?

        if self.is_waiting_on_workflow_call() or self._has_pending_lifecycle_execution():
            return None
        # Failed graphs stop scheduling immediately; is_complete() treats the error as terminal as well.
        if self.has_error():
            return None

        # If there are no prepared nodes, prepare some nodes
        next_node = self._get_next_node()
        if next_node is None:
            next_node = self._prepare_until_node_ready()

        # Get values from edges
        if next_node is not None:
            try:
                self._prepare_inputs(next_node)
            except ValidationError as e:
                raise NodeInputError(next_node, e)

        # If next is still none, there's no next node, return None
        return next_node

    def complete(self, node_id: str, output: BaseInvocationOutput) -> list[tuple[BaseInvocation, BaseInvocationOutput]]:
        """Validate and apply a direct compatibility completion through the execution ledger."""

        if self._apply_transaction is None:
            existing_ref = self.execution_refs.get(node_id)
            execution_ref = self._expected_execution_ref(
                node_id, effect_count=existing_ref.effect_count if existing_ref is not None else None
            )
            if existing_ref is not None and (
                existing_ref.reference_id not in ("", execution_ref.reference_id)
                or existing_ref.state_id not in ("", execution_ref.state_id)
                or existing_ref.exec_node_id not in ("", execution_ref.exec_node_id)
                or existing_ref.source_node_id not in ("", execution_ref.source_node_id)
            ):
                raise ValueError(f"Execution reference for {node_id} does not belong to this state")
            if execution_ref.exec_node_id in self.executed or execution_ref.reference_id in self.execution_effects:
                # Historical callers may replace a result after invoking a node directly. The scheduler has
                # already applied its transition, so preserve that idempotent result-replacement contract.
                return self._complete(node_id, output)
            if isinstance(output, ForInvocationOutput):
                try:
                    self._normalize_continuation_payload(output.item)
                except ValueError:
                    # In-memory compatibility callers may use arbitrary collection values. Such results cannot
                    # enter the JSON execution ledger, but must retain the old scheduler behavior.
                    return self._complete(node_id, output)
            if isinstance(output, ForReturnInvocationOutput):
                return_node = self.execution_graph.get_node(execution_ref.exec_node_id)
                if isinstance(return_node, ForReturnInvocation):
                    try:
                        self._normalize_continuation_payload(return_node.output)
                    except ValueError:
                        # A return connected to a non-JSON in-memory value has the same non-persistable constraint.
                        return self._complete(node_id, output)
            return self.apply(execution_ref, output, _compatibility_completion=True)
        return self._complete(node_id, output)

    def _complete(
        self, node_id: str, output: BaseInvocationOutput
    ) -> list[tuple[BaseInvocation, BaseInvocationOutput]]:
        """Apply a result after the caller has established the transaction and ledger boundary."""

        finalized_outputs = self._scheduler().complete(node_id, output)
        if self._mark_completed_sources():
            self.execution_graph._invalidate_edge_indexes()
            self._tx_set_attr(self, "_ready_queues", {})
            self._tx_set_attr(self, "_ready_node_ids", set())
            self._tx_set_attr(self, "_active_class", None)
        return finalized_outputs

    def set_node_error(self, node_id: str, error: str):
        """Marks a node as errored"""
        self.errors[node_id] = error
        self._pending_lifecycle_execution_nodes.discard(node_id)
        self._clear_transient_runtime()

    def is_complete(self) -> bool:
        """Returns true if the graph is complete"""
        if self.has_error():
            return self._is_complete_with_completed_sources()
        if self.is_waiting_on_workflow_call() or self._has_pending_lifecycle_execution():
            return False
        return self._is_complete_with_completed_sources()

    def _completed_source_ids(self) -> set[str]:
        completed_source_ids = self._get_completed_source_ids_cache().copy()
        for source_node_id in tuple(completed_source_ids):
            source_node = self.graph.nodes.get(source_node_id)
            if isinstance(source_node, ForInvocation) and not self._all_for_contexts_finalized(source_node_id):
                completed_source_ids.discard(source_node_id)
        return completed_source_ids

    def _is_complete_with_completed_sources(self) -> bool:
        if self.has_error():
            return True
        completed_source_ids = self._completed_source_ids()
        node_ids = set(self._get_source_graph_flat().nodes)
        return all(node_id in completed_source_ids for node_id in node_ids)

    def _mark_completed_sources(self) -> bool:
        completed_source_ids = self._completed_source_ids()
        is_complete = self.has_error() or all(
            node_id in completed_source_ids for node_id in self._get_source_graph_flat().nodes
        )
        if not is_complete:
            return False

        for source_node_id in nx.topological_sort(self._get_source_graph_flat()):
            if (
                source_node_id in completed_source_ids
                and source_node_id not in self.executed
                and not self._is_source_inactive(source_node_id)
            ):
                self._mark_source_executed(source_node_id)
        return True

    def has_error(self) -> bool:
        """Returns true if the graph has any errors"""
        return len(self.errors) > 0

    def get_workflow_call_depth(self) -> int:
        return len(self.workflow_call_stack)

    def is_waiting_on_workflow_call(self) -> bool:
        return self.waiting_workflow_call is not None

    def _has_pending_lifecycle_execution(self) -> bool:
        return any(exec_node_id not in self.executed for exec_node_id in self._pending_lifecycle_execution_nodes)

    def build_workflow_call_frame(self, exec_node_id: str, workflow_id: str) -> WorkflowCallFrame:
        if exec_node_id not in self.execution_graph.nodes:
            raise NodeNotFoundError(f"Node {exec_node_id} not found in execution graph")
        if exec_node_id not in self.prepared_source_mapping:
            raise ValueError(f"Node {exec_node_id} is not a prepared execution node")

        next_depth = self.get_workflow_call_depth() + 1
        if next_depth > self.max_workflow_call_depth:
            raise ValueError(
                f"Maximum workflow call depth exceeded ({self.max_workflow_call_depth}) for workflow '{workflow_id}'"
            )

        return WorkflowCallFrame(
            prepared_call_node_id=exec_node_id,
            source_call_node_id=self.prepared_source_mapping[exec_node_id],
            workflow_id=workflow_id,
            depth=next_depth,
        )

    def begin_waiting_on_workflow_call(self, frame: WorkflowCallFrame) -> None:
        if self.waiting_workflow_call is not None:
            raise ValueError("Execution state is already waiting on a workflow call")
        self.waiting_workflow_call = frame
        self.waiting_workflow_call_execution = WorkflowCallExecution(
            parent_session_id=self.id,
            prepared_call_node_id=frame.prepared_call_node_id,
            source_call_node_id=frame.source_call_node_id,
            workflow_id=frame.workflow_id,
            depth=frame.depth,
            status="waiting_for_child",
        )

    def attach_waiting_workflow_call_child_session(self, child_session: "GraphExecutionState") -> None:
        if self.waiting_workflow_call is None:
            raise ValueError("Execution state must be waiting on a workflow call before attaching a child session")
        if self.waiting_workflow_call_execution is None:
            raise ValueError("Execution state is waiting on a workflow call but has no workflow call execution")
        self.waiting_workflow_call_child_session = child_session
        self.waiting_workflow_call_execution.child_session_id = child_session.id
        self.waiting_workflow_call_execution.child_session_ids = [child_session.id]
        self.waiting_workflow_call_execution.expected_child_count = 1
        self.waiting_workflow_call_execution.status = "running_child"
        child_session.workflow_call_parent = WorkflowCallParentRef(
            workflow_call_id=self.waiting_workflow_call_execution.id,
            parent_session_id=self.waiting_workflow_call_execution.parent_session_id,
            prepared_call_node_id=self.waiting_workflow_call_execution.prepared_call_node_id,
            source_call_node_id=self.waiting_workflow_call_execution.source_call_node_id,
            workflow_id=self.waiting_workflow_call_execution.workflow_id,
            depth=self.waiting_workflow_call_execution.depth,
        )

    def attach_waiting_workflow_call_child_sessions(self, child_sessions: list["GraphExecutionState"]) -> None:
        if not child_sessions:
            raise ValueError("Workflow call must attach at least one child session")
        if self.waiting_workflow_call_execution is None:
            raise ValueError("Execution state is waiting on a workflow call but has no workflow call execution")
        self.waiting_workflow_call_child_session = child_sessions[0] if len(child_sessions) == 1 else None
        self.waiting_workflow_call_execution.child_session_id = child_sessions[0].id
        self.waiting_workflow_call_execution.child_session_ids = [child_session.id for child_session in child_sessions]
        self.waiting_workflow_call_execution.expected_child_count = len(child_sessions)
        self.waiting_workflow_call_execution.status = "running_child"
        for child_session in child_sessions:
            child_session.workflow_call_parent = WorkflowCallParentRef(
                workflow_call_id=self.waiting_workflow_call_execution.id,
                parent_session_id=self.waiting_workflow_call_execution.parent_session_id,
                prepared_call_node_id=self.waiting_workflow_call_execution.prepared_call_node_id,
                source_call_node_id=self.waiting_workflow_call_execution.source_call_node_id,
                workflow_id=self.waiting_workflow_call_execution.workflow_id,
                depth=self.waiting_workflow_call_execution.depth,
            )

    def set_waiting_workflow_call_child_item_ids(self, child_item_ids: list[int]) -> None:
        if self.waiting_workflow_call_execution is None:
            raise ValueError("Execution state is not waiting on a workflow call.")
        if len(child_item_ids) != self.waiting_workflow_call_execution.expected_child_count:
            raise ValueError("Workflow call child item count does not match expected child count.")
        if len(set(child_item_ids)) != len(child_item_ids):
            raise ValueError("Workflow call child item ids must be unique.")
        self.waiting_workflow_call_execution.child_item_ids = list(child_item_ids)
        self._register_generic_child_dependency()

    def record_waiting_workflow_call_child_completion(
        self, child_item_id: int, output_values: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        if self.waiting_workflow_call_execution is None:
            raise ValueError("Execution state is not waiting on a workflow call.")
        execution = self.waiting_workflow_call_execution
        if execution.child_item_ids and child_item_id not in execution.child_item_ids:
            raise ValueError(f"Child queue item {child_item_id} does not belong to the active workflow call.")
        if child_item_id not in execution.completed_child_item_ids:
            if (
                execution.expected_child_count > 1
                and execution.child_outputs
                and set(output_values.keys()) != set(next(iter(execution.child_outputs.values())).keys())
            ):
                raise ValueError("Batched child workflows returned different workflow return keys.")
            dependency = self._register_generic_child_dependency()
            if dependency is not None:
                dependency.complete_child(str(child_item_id), output_values)
            execution.completed_child_item_ids.append(child_item_id)
            execution.child_outputs[child_item_id] = dict(output_values)

            ordered_item_ids = execution.child_item_ids or execution.completed_child_item_ids
            execution.aggregated_values = {
                key: [
                    execution.child_outputs[item_id][key]
                    for item_id in ordered_item_ids
                    if item_id in execution.child_outputs
                ]
                for key in output_values
            }
        is_complete = len(execution.completed_child_item_ids) >= execution.expected_child_count
        if execution.expected_child_count == 1:
            return (
                is_complete,
                {key: values[0] for key, values in execution.aggregated_values.items()},
            )
        return (
            is_complete,
            {key: list(values) for key, values in execution.aggregated_values.items()},
        )

    def end_waiting_on_workflow_call(
        self,
        status: Literal["completed", "failed"] = "completed",
        error_message: Optional[str] = None,
    ) -> None:
        if self.waiting_workflow_call_execution is not None:
            self.waiting_workflow_call_execution.status = status
            self.waiting_workflow_call_execution.error_message = error_message
            self.workflow_call_history.append(self.waiting_workflow_call_execution.model_copy(deep=True))
        self.waiting_workflow_call = None
        self.waiting_workflow_call_execution = None
        self.waiting_workflow_call_child_session = None

    def create_child_workflow_execution_state(self, graph: Graph, frame: WorkflowCallFrame) -> "GraphExecutionState":
        return GraphExecutionState(
            graph=graph,
            workflow_call_stack=[*self.workflow_call_stack, frame],
            max_workflow_call_depth=self.max_workflow_call_depth,
        )

    def _create_execution_node(self, node_id: str, iteration_node_map: list[tuple[str, str]]) -> list[str]:
        return self._materializer().create_execution_node(node_id, iteration_node_map, enforce_admission=False)

    def _iterator_graph(self, base: Optional["nx.DiGraph"] = None) -> "nx.DiGraph":
        if (
            isinstance(self._execution_scheduler, _GenericGraphSchedulerAdapter)
            and self._can_use_generic_for_scheduler()
        ):
            return self._for_planner().iterator_graph(base)
        return self._materializer().iterator_graph(base)

    def _prepare(self, base_g: Optional["nx.DiGraph"] = None) -> Optional[str]:
        return self._materializer().prepare(base_g)

    def _get_iteration_node(
        self,
        source_node_id: str,
        graph: "nx.DiGraph",
        execution_graph: "nx.DiGraph",
        prepared_iterator_nodes: list[str],
    ) -> Optional[str]:
        return self._materializer().get_iteration_node(source_node_id, graph, execution_graph, prepared_iterator_nodes)

    def _get_next_node(self) -> Optional[BaseInvocation]:
        return self._scheduler().get_next_node()

    def _prepare_inputs(self, node: BaseInvocation):
        self._runtime().prepare_inputs(node)

    # Reserved for a future graph-editing API that validates changes against the current execution state.
    def _is_edge_valid(self, edge: Edge) -> bool:
        try:
            self.graph._validate_edge(edge)
        except InvalidEdgeError:
            return False

        # Invalid if destination has already been prepared or executed
        if edge.destination.node_id in self.source_prepared_mapping:
            return False

        # Otherwise, the edge is valid
        return True

    def _is_node_updatable(self, node_id: str) -> bool:
        # The node is updatable as long as it hasn't been prepared or executed
        return node_id not in self.source_prepared_mapping

    def add_node(self, node: BaseInvocation) -> None:
        self.graph.add_node(node)
        self._invalidate_source_graph_cache()

    def update_node(self, node_id: str, new_node: BaseInvocation) -> None:
        if not self._is_node_updatable(node_id):
            raise NodeAlreadyExecutedError(
                f"Node {node_id} has already been prepared or executed and cannot be updated"
            )
        self.graph.update_node(node_id, new_node)
        self._invalidate_source_graph_cache()

    def delete_node(self, node_id: str) -> None:
        if not self._is_node_updatable(node_id):
            raise NodeAlreadyExecutedError(
                f"Node {node_id} has already been prepared or executed and cannot be deleted"
            )
        self.graph.delete_node(node_id)
        self._invalidate_source_graph_cache()

    def add_edge(self, edge: Edge) -> None:
        if not self._is_node_updatable(edge.destination.node_id):
            raise NodeAlreadyExecutedError(
                f"Destination node {edge.destination.node_id} has already been prepared or executed and cannot be linked to"
            )
        self.graph.add_edge(edge)
        self._invalidate_source_graph_cache()

    def delete_edge(self, edge: Edge) -> None:
        if not self._is_node_updatable(edge.destination.node_id):
            raise NodeAlreadyExecutedError(
                f"Destination node {edge.destination.node_id} has already been prepared or executed and cannot have a source edge deleted"
            )
        self.graph.delete_edge(edge)
        self._invalidate_source_graph_cache()


_GRAPH_RUNTIME_ATTRIBUTE_NAMES = (
    "ready_queues",
    "ready_node_ids",
    "active_class",
    "resolved_if_exec_branches",
    "pending_if_exec_nodes",
    "if_branch_sources_cache",
    "if_activation_dependencies_by_source",
    "if_activation_dependencies_by_exec",
    "prepared_if_exec_ids_by_source_and_frame",
    "prepared_if_exec_frames_by_source",
    "prepared_if_exec_indexed_sources",
    "prepared_exec_metadata",
    "prepared_exec_registry",
    "if_activation_controller_instance",
    "execution_materializer",
    "generic_for_planner",
    "execution_scheduler",
    "execution_runtime",
    "for_parent_iteration_paths_cache",
    "all_for_contexts_finalized_cache",
    "prepared_for_index",
    "final_prepared_for_index",
    "prepared_for_index_by_exec",
    "source_graph_flat",
    "execution_graph_flat",
    "completed_source_ids_cache",
    "unexecuted_prepared_counts",
    "for_source_by_return_id",
    "apply_transaction",
    "generic_execution_runtime",
    "generic_graph_scheduler",
    "generic_child_dependencies",
    "pending_lifecycle_execution_nodes",
    "execution_effects_persisted",
    "legacy_execution_snapshot",
    "legacy_snapshot_loaded",
)


def _runtime_state_property(name: str) -> property:
    def getter(state: GraphExecutionState) -> Any:
        # Access the single PrivateAttr through pydantic's storage directly. Using ``state._runtime_state``
        # here would put every compatibility-property access back through BaseModel.__getattr__.
        return getattr(state.__pydantic_private__["_runtime_state"], name)

    def setter(state: GraphExecutionState, value: Any) -> None:
        setattr(state.__pydantic_private__["_runtime_state"], name, value)

    return property(getter, setter)


for _runtime_attribute_name in _GRAPH_RUNTIME_ATTRIBUTE_NAMES:
    setattr(GraphExecutionState, f"_{_runtime_attribute_name}", _runtime_state_property(_runtime_attribute_name))
