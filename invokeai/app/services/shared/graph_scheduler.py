"""Scheduler implementations used by the graph execution-state façade."""

from collections import deque
from typing import TYPE_CHECKING, Deque, Iterable, Optional

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForInvocationOutput
from invokeai.app.services.shared.execution_engine.scheduler import (
    ExecutionPlan,
    ExecutionScheduler,
)
from invokeai.app.services.shared.graph_nested_iterate_planner import (
    can_use_nested_iterate_planner,
    get_nested_iterate_sequence_depth,
    prepare_nested_iterate_bodies,
    prepare_nested_iterate_sequence,
)
from invokeai.app.services.shared.graph_validation import CollectInvocation, IterateInvocation, nx

if TYPE_CHECKING:
    from invokeai.app.services.shared.graph import GraphExecutionState
    from invokeai.app.services.shared.graph_for_planner import _GenericForPlanner
    from invokeai.app.services.shared.graph_materializer import _ExecutionMaterializer


def _prepare_deferred_nested_for_body(
    state: "GraphExecutionState",
    exec_node_id: str,
    planner: "_GenericForPlanner | _ExecutionMaterializer",
) -> None:
    """Prepare a deferred nested For body through the caller's node planner."""

    completed_source_id = state._prepared_registry().get_source_node_id(exec_node_id)
    graph = state._get_source_graph_flat()
    for source_for_id, source_node in state.graph.nodes.items():
        if not isinstance(source_node, ForInvocation):
            continue
        serial_nested_body = state.graph._get_supported_for_serial_nested_iterate_chain(source_for_id, graph)
        if serial_nested_body is not None:
            if completed_source_id not in serial_nested_body.body_path_nodes:
                continue
            prepared_exec_path = state._get_iteration_path(exec_node_id)
            for prepared_for_id in state._prepared_registry().get_prepared_ids(source_for_id):
                prepared_for_path = state._get_iteration_path(prepared_for_id)
                prepared_for_node = state.execution_graph.get_node(prepared_for_id)
                if isinstance(prepared_for_node, ForInvocation) and prepared_for_node.index >= 0:
                    prepared_for_path = (
                        *state._get_for_parent_iteration_path(prepared_for_id),
                        prepared_for_node.index,
                    )
                if prepared_exec_path[: len(prepared_for_path)] == prepared_for_path:
                    planner.create_for_body_iteration(source_for_id=source_for_id, prepared_for_id=prepared_for_id)
                    return
        nested_body = state.graph._get_supported_for_nested_iterate_body(source_for_id, graph)
        nested_for_body = state.graph._get_supported_for_nested_for_body(source_for_id, graph)
        if nested_body is not None:
            body_path_nodes = nested_body.body_path_nodes
            deferred_node_ids = (nested_body.iterate_node_id,)
        elif nested_for_body is None:
            continue
        else:
            body_path_nodes = nested_for_body.body_path_nodes
            deferred_node_ids = nested_for_body.inner_for_ids
        if completed_source_id != source_for_id and (
            completed_source_id not in body_path_nodes
            or not any(
                nx.has_path(graph, completed_source_id, deferred_node_id) for deferred_node_id in deferred_node_ids
            )
        ):
            continue
        for prepared_for_id in state._prepared_registry().get_prepared_ids(source_for_id):
            prepared_for_node = state.execution_graph.get_node(prepared_for_id)
            prepared_for_path = state._get_iteration_path(prepared_for_id)
            if isinstance(prepared_for_node, ForInvocation) and prepared_for_node.index >= 0:
                prepared_for_path = (
                    *state._get_for_parent_iteration_path(prepared_for_id),
                    prepared_for_node.index,
                )
            if prepared_for_path != state._get_iteration_path(exec_node_id):
                continue
            if nested_for_body is not None:
                if not all(
                    any(
                        state._get_for_parent_iteration_path(prepared_child_id) == prepared_for_path
                        for prepared_child_id in state._prepared_registry().get_prepared_ids(child_id)
                    )
                    for child_id in deferred_node_ids
                ):
                    planner.create_for_body_iteration(source_for_id=source_for_id, prepared_for_id=prepared_for_id)
                    return
                continue
            if any(
                (iterate_path := state._get_iteration_path(prepared_iterate_id))[: len(prepared_for_path)]
                == prepared_for_path
                and len(iterate_path) > len(prepared_for_path)
                for prepared_iterate_id in state._prepared_registry().get_prepared_ids(deferred_node_ids[0])
            ):
                continue
            planner.create_for_body_iteration(source_for_id=source_for_id, prepared_for_id=prepared_for_id)
            return


class _ExecutionScheduler:
    """Owns ready-queue ordering and indegree-driven execution transitions."""

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state
        self._runtime_state = state.__pydantic_private__["_runtime_state"]

    def _validate_exec_node_ready_state(self, exec_node_id: str) -> None:
        if exec_node_id not in self._state.execution_graph.nodes:
            raise KeyError(f"exec node {exec_node_id} missing from execution_graph")
        if exec_node_id not in self._state.indegree:
            raise KeyError(f"indegree missing for exec node {exec_node_id}")

    def _should_skip_ready_enqueue(self, exec_node_id: str) -> bool:
        if (
            self._state.indegree[exec_node_id] != 0
            or exec_node_id in self._state.executed
            or self._state._is_pending_if(exec_node_id)
            or self._state._is_deferred_by_unresolved_if(exec_node_id)
        ):
            return True
        node = self._state.execution_graph.nodes[exec_node_id]
        return isinstance(node, CollectInvocation) and not self._state._collect_streams_ready(exec_node_id)

    def _discard_rejected_activation_node(self, exec_node_id: str) -> bool:
        if exec_node_id not in self._state.indegree:
            return False
        metadata = self._state._get_prepared_exec_metadata(exec_node_id)
        if metadata.state in ("executed", "skipped"):
            return False
        if not self._state._has_rejected_activation_dependency(exec_node_id):
            return False

        self.remove_from_ready_queues(exec_node_id)
        self._state._set_prepared_exec_state(exec_node_id, "skipped")
        self._state._mark_exec_node_executed(exec_node_id)

        source_node_id = self._state._prepared_registry().get_source_node_id(exec_node_id)
        if self._state._count_unexecuted_prepared(source_node_id) == 0:
            if source_node_id not in self._state.executed:
                self._state._mark_source_executed(source_node_id)
        self.mark_skipped(exec_node_id)
        return True

    def _discard_rejected_activation_nodes(self) -> None:
        for exec_node_id in tuple(nx.topological_sort(self._state._get_execution_graph_flat())):
            self._discard_rejected_activation_node(exec_node_id)

    def _enqueue_activation_ready_nodes(self) -> None:
        for exec_node_id in tuple(self._state.prepared_source_mapping):
            if exec_node_id not in self._state.indegree or exec_node_id in self._state.executed:
                continue
            self.enqueue_if_ready(exec_node_id)

    def _get_ready_queue(self, exec_node_id: str) -> Deque[str]:
        node_obj = self._state.execution_graph.nodes[exec_node_id]
        return self.queue_for(self._state._type_key(node_obj))

    def _insert_ready_node(self, queue: Deque[str], exec_node_id: str) -> None:
        exec_node_path = self._state._get_iteration_path(exec_node_id)
        if not queue or self._state._get_iteration_path(queue[-1]) <= exec_node_path:
            self._state._tx_queue_append(queue, exec_node_id)
            return
        for i, existing in enumerate(queue):
            if self._state._get_iteration_path(existing) > exec_node_path:
                self._state._tx_queue_insert(queue, i, exec_node_id)
                return
        self._state._tx_queue_append(queue, exec_node_id)

    def _record_completed_node(self, exec_node_id: str, output: BaseInvocationOutput) -> None:
        self._state._set_prepared_exec_state(exec_node_id, "executed")
        self._state._mark_exec_node_executed(exec_node_id)
        self._state._tx_set_mapping(self._state.results, exec_node_id, output)
        # Keep the generic stream ledger in sync while the materializer remains
        # the compatibility authority for runtime node creation.
        if isinstance(self._state.execution_graph.nodes.get(exec_node_id), IterateInvocation):
            self._state._record_iterate_stream(exec_node_id, output)
        if isinstance(self._state.execution_graph.nodes.get(exec_node_id), ForInvocation):
            for_node = self._state.execution_graph.nodes[exec_node_id]
            assert isinstance(for_node, ForInvocation)
            if for_node.index >= 0:
                self._state._for_continuation(exec_node_id)
        node = self._state.execution_graph.nodes[exec_node_id]
        if isinstance(node, (IterateInvocation, CollectInvocation)):
            self._state._tx_set_attr(node, "collection", [])

    def _mark_source_node_complete(self, exec_node_id: str) -> None:
        registry = self._state._prepared_registry()
        source_node_id = registry.get_source_node_id(exec_node_id)
        if self._state._count_unexecuted_prepared(source_node_id) == 0 and source_node_id not in self._state.executed:
            self._state._mark_source_executed(source_node_id)

    def _try_schedule_next_for_iteration(self, exec_node_id: str, output: BaseInvocationOutput) -> Optional[str]:
        return self._state._try_schedule_next_for_iteration(exec_node_id, output)

    def _try_materialize_deferred_nested_for_body(self, exec_node_id: str) -> None:
        _prepare_deferred_nested_for_body(self._state, exec_node_id, self._state._materializer())

    def _decrement_child_indegree(self, child_exec_node_id: str, parent_exec_node_id: str) -> None:
        if child_exec_node_id not in self._state.indegree:
            raise KeyError(f"indegree missing for exec node {child_exec_node_id}")
        if self._state.indegree[child_exec_node_id] == 0:
            raise RuntimeError(f"indegree underflow for {child_exec_node_id} from parent {parent_exec_node_id}")
        self._state._tx_set_mapping(
            self._state.indegree, child_exec_node_id, self._state.indegree[child_exec_node_id] - 1
        )

    def _release_downstream_nodes(self, exec_node_id: str) -> None:
        for edge in self._state.execution_graph._get_output_edges(exec_node_id):
            child = edge.destination.node_id
            self._decrement_child_indegree(child, exec_node_id)
            self._state._try_resolve_if_node(child)
            if self._state.indegree[child] == 0:
                self.enqueue_if_ready(child)

    def queue_for(self, cls_name: str) -> Deque[str]:
        runtime = self._runtime_state
        q = runtime.ready_queues.get(cls_name)
        if q is None:
            q = deque()
            self._state._tx_set_mapping(runtime.ready_queues, cls_name, q)
        return q

    def remove_from_ready_queues(self, exec_node_id: str) -> None:
        runtime = self._runtime_state
        for q in runtime.ready_queues.values():
            try:
                self._state._tx_queue_remove(q, exec_node_id)
            except ValueError:
                continue
        self._state._tx_discard_set(runtime.ready_node_ids, exec_node_id)

    def mark_skipped(self, exec_node_id: str) -> None:
        """Satisfy downstream indegrees without resolving inputs from a skipped node."""
        self._validate_exec_node_ready_state(exec_node_id)
        for edge in self._state.execution_graph._get_output_edges(exec_node_id):
            child = edge.destination.node_id
            self._decrement_child_indegree(child, exec_node_id)
            if self._state.indegree[child] == 0:
                self.enqueue_if_ready(child)

    def enqueue_if_ready(self, exec_node_id: str) -> None:
        """Push exec_node_id to its class queue if unmet inputs == 0."""
        self._validate_exec_node_ready_state(exec_node_id)
        if self._discard_rejected_activation_node(exec_node_id):
            return
        if self._should_skip_ready_enqueue(exec_node_id):
            return
        queue = self._get_ready_queue(exec_node_id)
        runtime = self._runtime_state
        if exec_node_id in runtime.ready_node_ids:
            return
        self._state._set_prepared_exec_state(exec_node_id, "ready")
        self._insert_ready_node(queue, exec_node_id)
        self._state._tx_add_set(runtime.ready_node_ids, exec_node_id)

    def get_next_node(self) -> Optional[BaseInvocation]:
        """Gets the next ready node: FIFO within class, drain class before switching."""
        runtime = self._runtime_state
        while True:
            if not runtime.active_class and self._state.results:
                last_exec_node_id = next(reversed(self._state.results))
                last_node = self._state.execution_graph.nodes.get(last_exec_node_id)
                if last_node is not None:
                    last_class = self._state._type_key(last_node)
                    if runtime.ready_queues.get(last_class):
                        runtime.active_class = last_class
            if runtime.active_class:
                q = runtime.ready_queues.get(runtime.active_class)
                while q:
                    exec_node_id = q.popleft()
                    runtime.ready_node_ids.discard(exec_node_id)
                    if exec_node_id not in self._state.executed:
                        return self._state.execution_graph.nodes[exec_node_id]
                runtime.active_class = None
                continue

            seen = set(self._state.ready_order)
            next_class = next(
                (cls_name for cls_name in self._state.ready_order if runtime.ready_queues.get(cls_name)),
                None,
            )
            if next_class is None:
                next_class = next(
                    (
                        cls_name
                        for cls_name in sorted(k for k in runtime.ready_queues.keys() if k not in seen)
                        if runtime.ready_queues[cls_name]
                    ),
                    None,
                )
            if next_class is None:
                return None

            runtime.active_class = next_class

    def complete(
        self, exec_node_id: str, output: BaseInvocationOutput
    ) -> list[tuple[BaseInvocation, BaseInvocationOutput]]:
        if exec_node_id not in self._state.execution_graph.nodes:
            return []

        self._record_completed_node(exec_node_id, output)
        finalized_for_exec_node_id = self._try_schedule_next_for_iteration(exec_node_id, output)
        self._mark_source_node_complete(exec_node_id)
        self._release_downstream_nodes(exec_node_id)
        completed_node = self._state.execution_graph.get_node(exec_node_id)
        if isinstance(completed_node, ForInvocation) and completed_node.index >= 0:
            source_for_id = self._state._prepared_registry().get_source_node_id(exec_node_id)
            nested_body = self._state.graph._get_supported_for_nested_iterate_body(
                source_for_id, self._state._get_source_graph_flat()
            )
            prepared_for_node = self._state.execution_graph.get_node(exec_node_id)
            prepared_for_path = self._state._get_iteration_path(exec_node_id)
            if isinstance(prepared_for_node, ForInvocation) and prepared_for_node.index >= 0:
                prepared_for_path = (
                    *self._state._get_for_parent_iteration_path(exec_node_id),
                    prepared_for_node.index,
                )
            if nested_body is not None and not any(
                (iterate_path := self._state._get_iteration_path(prepared_iterate_id))[: len(prepared_for_path)]
                == prepared_for_path
                and len(iterate_path) > len(prepared_for_path)
                for prepared_iterate_id in self._state._prepared_registry().get_prepared_ids(
                    nested_body.iterate_node_id
                )
            ):
                self._state._create_for_body_iteration(source_for_id=source_for_id, prepared_for_id=exec_node_id)
            elif nested_body is None:
                self._try_materialize_deferred_nested_for_body(exec_node_id)
        else:
            self._try_materialize_deferred_nested_for_body(exec_node_id)
        if finalized_for_exec_node_id is None:
            return []
        finalized_for_node = self._state.execution_graph.get_node(finalized_for_exec_node_id)
        finalized_for_output = self._state.results.get(finalized_for_exec_node_id)
        if not isinstance(finalized_for_node, ForInvocation) or not isinstance(
            finalized_for_output, ForInvocationOutput
        ):
            return []
        return [(finalized_for_node, finalized_for_output)]


class _GenericGraphSchedulerAdapter:
    """Projects the generic scheduler into GraphExecutionState's legacy queues."""

    def __init__(self, state: "GraphExecutionState") -> None:
        self._state = state
        self._runtime_state = state.__pydantic_private__["_runtime_state"]
        self._initializing = True
        self._if_exec_ids: dict[str, None] = {}
        self._scheduler = ExecutionScheduler(
            ExecutionPlan(),
            state.ready_order,
            ready_predicate=self._is_node_activation_ready,
        )
        self._known_result_ids = set(state.results)
        self._register_existing_nodes()
        prepared_ids = set(state.prepared_source_mapping).intersection(state.results)
        discarded_ids = {
            exec_node_id
            for exec_node_id in state.prepared_source_mapping
            if state._get_prepared_exec_metadata(exec_node_id).state == "skipped"
        }
        self._scheduler.executed = prepared_ids
        self._scheduler.discarded = discarded_ids
        self._scheduler.rebuild_ready()
        self._initializing = False
        self._sync_indegree()
        self._project_ready_nodes()
        self._restore_active_class()

    def _restore_active_class(self) -> None:
        """Continue draining the class that produced the latest durable result."""

        if not self._state.results:
            return
        last_exec_node_id = next(reversed(self._state.results))
        last_plan_node = self._scheduler.plan.nodes.get(last_exec_node_id)
        if last_plan_node is None:
            return
        self._scheduler._active_class = last_plan_node.class_name

    def _is_node_activation_ready(self, exec_node_id: str) -> bool:
        """Require every frame-local activation dependency to be satisfied."""

        if self._state._is_pending_if(exec_node_id):
            return False
        if not all(
            self._state._is_activation_dependency_satisfied(dependency)
            for dependency in self._scheduler.plan.nodes[exec_node_id].activation_dependencies
        ):
            return False
        node = self._state.execution_graph.nodes[exec_node_id]
        return not isinstance(node, CollectInvocation) or self._state._collect_streams_ready(exec_node_id)

    def _retire_unselected_node(self, exec_node_id: str) -> None:
        """Retire one unselected prepared node through generic scheduler state."""

        if exec_node_id not in self._scheduler.plan.nodes or exec_node_id in self._state.executed:
            return
        if exec_node_id in self._scheduler.discarded:
            return
        self._remove_projected(exec_node_id)
        self._state._set_prepared_exec_state(exec_node_id, "skipped")
        self._state._mark_exec_node_executed(exec_node_id)
        self._scheduler.discard(exec_node_id)
        source_node_id = self._state._prepared_registry().get_source_node_id(exec_node_id)
        if self._state._count_unexecuted_prepared(source_node_id) == 0:
            if source_node_id not in self._state.executed:
                self._state._mark_source_executed(source_node_id)

    def resolve_if_node(self, exec_node_id: str, *, enqueue: bool = True) -> None:
        """Resolve legacy If inputs without pruning graph edges or using type-specific skips."""

        runtime = self._runtime_state
        if exec_node_id in runtime.resolved_if_exec_branches:
            return
        node = self._state.execution_graph.get_node(exec_node_id)
        if not isinstance(node, IfInvocation):
            return
        if not self._state._apply_if_condition_inputs(exec_node_id, node):
            return

        selected_field = "true_input" if node.condition else "false_input"
        self._state._resolve_activation_gate(exec_node_id, selected_field)
        self._state._tx_set_mapping(runtime.resolved_if_exec_branches, exec_node_id, selected_field)
        self._state._record_compatibility_activation_token(exec_node_id, selected_field)
        self._discard_rejected_nodes()
        self._sync_indegree()
        if enqueue:
            self._project_ready_nodes()

    def _discard_rejected_node(self, exec_node_id: str) -> None:
        node = self._scheduler.plan.nodes.get(exec_node_id)
        if node is None or not node.activation_dependencies:
            return
        if any(
            self._state._is_activation_dependency_rejected(dependency) for dependency in node.activation_dependencies
        ):
            self._retire_unselected_node(exec_node_id)

    def _discard_rejected_nodes(self) -> None:
        """Discard plan nodes whose opaque activation requirements are rejected."""

        for exec_node_id in tuple(self._scheduler.plan.nodes):
            self._discard_rejected_node(exec_node_id)

    def _sync_executed_state(self, excluded: Iterable[str] = ()) -> None:
        """Synchronize externally completed plan nodes without rebuilding on every normal completion."""
        if len(self._state.results) == len(self._known_result_ids):
            return
        excluded_ids = set(excluded)
        newly_executed = {
            exec_node_id
            for exec_node_id in self._state.results
            if exec_node_id not in self._known_result_ids
            if exec_node_id not in excluded_ids
            and exec_node_id in self._scheduler.plan.nodes
            and exec_node_id not in self._scheduler.executed
        }
        self._known_result_ids.update(self._state.results)
        if not newly_executed:
            return
        self._scheduler.executed.update(newly_executed)
        self._scheduler.rebuild_ready()
        self._sync_indegree()

    def _register_existing_nodes(self) -> None:
        execution_graph = self._state._get_execution_graph_flat()
        for exec_node_id in nx.topological_sort(execution_graph):
            self.register_node(exec_node_id)

    def register_node(self, exec_node_id: str, *, project: bool = True) -> None:
        if exec_node_id in self._scheduler.plan.nodes:
            return
        node = self._state.execution_graph.nodes.get(exec_node_id)
        if node is None:
            raise KeyError(f"exec node {exec_node_id} missing from execution_graph")
        if isinstance(node, IfInvocation):
            self._if_exec_ids[exec_node_id] = None
        dependencies = tuple(edge.source.node_id for edge in self._state.execution_graph._get_input_edges(exec_node_id))
        activation_dependencies = self._state._get_activation_dependencies(exec_node_id)
        for dependency in dependencies:
            if dependency not in self._scheduler.plan.nodes:
                self.register_node(dependency)
        if exec_node_id in self._state.executed:
            self._scheduler.executed.add(exec_node_id)
        self._scheduler.add_node(
            self._scheduler.plan.add_node(
                exec_node_id,
                type(node).__name__,
                self._state._get_iteration_path(exec_node_id),
                dependencies,
                activation_dependencies,
            )
        )
        self._state._tx_set_mapping(self._state.indegree, exec_node_id, self._scheduler.indegree[exec_node_id])
        self._discard_rejected_node(exec_node_id)
        if not self._initializing and project:
            self._project_ready_node(exec_node_id)

    def _sync_indegree(self) -> None:
        for exec_node_id, degree in self._scheduler.indegree.items():
            self._state._tx_set_mapping(self._state.indegree, exec_node_id, degree)

    def _project_ready_node(self, exec_node_id: str) -> None:
        runtime = self._runtime_state
        if not self._scheduler.is_ready(exec_node_id) or exec_node_id in runtime.ready_node_ids:
            return
        node = self._state.execution_graph.nodes[exec_node_id]
        cls_name = self._state._type_key(node)
        queue = runtime.ready_queues.get(cls_name)
        if queue is None:
            queue = deque()
            self._state._tx_set_mapping(runtime.ready_queues, cls_name, queue)
        iteration_path = self._state._get_iteration_path(exec_node_id)
        if not queue or self._state._get_iteration_path(queue[-1]) <= iteration_path:
            self._state._tx_queue_append(queue, exec_node_id)
        else:
            insert_at = next(
                index
                for index, queued_id in enumerate(queue)
                if self._state._get_iteration_path(queued_id) > iteration_path
            )
            self._state._tx_queue_insert(queue, insert_at, exec_node_id)
        self._state._set_prepared_exec_state(exec_node_id, "ready")
        self._state._tx_add_set(runtime.ready_node_ids, exec_node_id)

    def _project_ready_nodes(self) -> None:
        for exec_node_id in self._scheduler.ready_ids:
            self._project_ready_node(exec_node_id)

    def _enqueue_activation_ready_nodes(self) -> None:
        """Recheck prepared nodes whose activation input arrived after plan registration."""

        for exec_node_id in tuple(self._if_exec_ids):
            if exec_node_id not in self._state.indegree or exec_node_id in self._state.executed:
                continue
            self._state._tx_set_mapping(
                self._state.indegree, exec_node_id, self._scheduler.indegree.get(exec_node_id, 0)
            )
            self.enqueue_if_ready(exec_node_id)

    def _remove_projected(self, exec_node_id: str) -> None:
        runtime = self._runtime_state
        for queue in runtime.ready_queues.values():
            try:
                self._state._tx_queue_remove(queue, exec_node_id)
            except ValueError:
                continue
        self._state._tx_discard_set(runtime.ready_node_ids, exec_node_id)

    def queue_for(self, cls_name: str) -> Deque[str]:
        runtime = self._runtime_state
        queue = runtime.ready_queues.get(cls_name)
        if queue is None:
            queue = deque()
            self._state._tx_set_mapping(runtime.ready_queues, cls_name, queue)
        return queue

    def remove_from_ready_queues(self, exec_node_id: str) -> None:
        self._scheduler.discard(exec_node_id)
        self._remove_projected(exec_node_id)

    def mark_skipped(self, exec_node_id: str) -> None:
        """Mirror a compatibility skip in the opaque scheduler projection."""

        self._scheduler.discard(exec_node_id)
        self._sync_indegree()
        self._project_ready_nodes()

    def enqueue_if_ready(self, exec_node_id: str) -> None:
        self.register_node(exec_node_id)
        if self._state.indegree.get(exec_node_id) != 0 or exec_node_id in self._state.executed:
            return
        self._scheduler.enqueue(exec_node_id)
        self._project_ready_node(exec_node_id)

    def get_next_node(self) -> Optional[BaseInvocation]:
        exec_node_id = self._scheduler.pop_next()
        if exec_node_id is None:
            return None
        self._remove_projected(exec_node_id)
        if exec_node_id in self._state.executed:
            return self.get_next_node()
        return self._state.execution_graph.nodes[exec_node_id]

    def _record_completed_node(self, exec_node_id: str, output: BaseInvocationOutput) -> None:
        node = self._state.execution_graph.nodes.get(exec_node_id)
        self._state._set_prepared_exec_state(exec_node_id, "executed")
        self._state._mark_exec_node_executed(exec_node_id)
        self._state._tx_set_mapping(self._state.results, exec_node_id, output)
        if isinstance(node, ForInvocation) and node.index >= 0:
            self._state._for_continuation(exec_node_id)
        if isinstance(node, IterateInvocation):
            self._state._record_iterate_stream(exec_node_id, output)
        if isinstance(node, (IterateInvocation, CollectInvocation)):
            self._state._tx_set_attr(node, "collection", [])

    def _mark_source_node_complete(self, exec_node_id: str) -> None:
        registry = self._state._prepared_registry()
        source_node_id = registry.get_source_node_id(exec_node_id)
        if self._state._count_unexecuted_prepared(source_node_id) == 0 and source_node_id not in self._state.executed:
            self._state._mark_source_executed(source_node_id)

    def complete(
        self, exec_node_id: str, output: BaseInvocationOutput
    ) -> list[tuple[BaseInvocation, BaseInvocationOutput]]:
        if exec_node_id not in self._state.execution_graph.nodes:
            return []
        # A few compatibility callers register prepared execution nodes directly on the
        # execution graph. Ensure those nodes are represented in the opaque plan before
        # applying their completion, just as materialized nodes are.
        if exec_node_id not in self._scheduler.plan.nodes:
            self.register_node(exec_node_id)
        if exec_node_id not in self._state.indegree:
            raise KeyError(f"indegree missing for exec node {exec_node_id}")
        if exec_node_id in self._scheduler.executed:
            # Compatibility callers may advance with the invocation result and then replace that result explicitly.
            self._state._tx_set_mapping(self._state.results, exec_node_id, output)
            return []
        dependents = self._scheduler.plan.dependents(exec_node_id)
        for dependent in dependents:
            if dependent not in self._state.indegree:
                raise KeyError(f"indegree missing for exec node {dependent}")
        self._remove_projected(exec_node_id)
        self._record_completed_node(exec_node_id, output)
        self._known_result_ids.add(exec_node_id)
        finalized_for_exec_node_id = self._state._apply_generic_for_continuation(exec_node_id, output)
        nested_iterate_sequence_depth = get_nested_iterate_sequence_depth(self._state)
        if nested_iterate_sequence_depth is None:
            self._mark_source_node_complete(exec_node_id)
        # A condition may become resolvable when this node completes. Resolve it
        # while the state has the completed result, before recalculating generic
        # indegrees and readiness.
        for dependent in dependents:
            self._state._try_resolve_if_node(dependent)
        self._sync_executed_state(excluded=(exec_node_id,))
        newly_ready = self._scheduler.complete(exec_node_id)
        for dependent in set(dependents):
            self._state._tx_set_mapping(self._state.indegree, dependent, self._scheduler.indegree[dependent])
        for ready_node_id in newly_ready:
            self._project_ready_node(ready_node_id)
        # Fresh If branch inputs may be materialized after the If dependency was
        # registered, so no ordinary plan edge exists to re-enqueue the If when
        # its selected branch completes. Recheck activation readiness here.
        if any(isinstance(node, IfInvocation) for node in self._state.graph.nodes.values()):
            self._enqueue_activation_ready_nodes()
        if can_use_nested_iterate_planner(self._state):
            prepare_nested_iterate_bodies(self._state)
        elif nested_iterate_sequence_depth is not None:
            prepare_nested_iterate_sequence(self._state, depth=nested_iterate_sequence_depth)
        else:
            _prepare_deferred_nested_for_body(self._state, exec_node_id, self._state._for_planner())
        if finalized_for_exec_node_id is None:
            return []
        finalized_for_node = self._state.execution_graph.get_node(finalized_for_exec_node_id)
        finalized_for_output = self._state.results.get(finalized_for_exec_node_id)
        if not isinstance(finalized_for_node, ForInvocation) or not isinstance(
            finalized_for_output, ForInvocationOutput
        ):
            return []
        return [(finalized_for_node, finalized_for_output)]

    def set_ready_order(self, ready_order: Iterable[str]) -> None:
        self._scheduler.set_ready_order(ready_order)
        self._project_ready_nodes()
