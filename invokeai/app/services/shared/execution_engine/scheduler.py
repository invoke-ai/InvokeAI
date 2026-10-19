"""Generic dependency planning and deterministic execution scheduling.

This module intentionally stores only opaque node identifiers, class names, and
frame values. Graph and invocation semantics belong to adapters above it.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

NodeId = str
Frame = tuple[object, ...]
ReadyPredicate = Callable[[NodeId], bool]


def _frame_key(frame: Frame) -> tuple[tuple[int, int | str], ...]:
    """Return a stable ordering key that preserves numeric iteration order."""

    return tuple(
        (0, part)
        if isinstance(part, int) and not isinstance(part, bool)
        else (1, part)
        if isinstance(part, str)
        else (2, f"{type(part).__name__}:{part!r}")
        for part in frame
    )


@dataclass(frozen=True, slots=True)
class ActivationDependency:
    """Opaque requirement for one branch activation in one frame."""

    owner_id: NodeId
    branch: str
    frame: Frame = ()

    def __post_init__(self) -> None:
        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise ValueError("activation dependency owner must not be blank")
        if not isinstance(self.branch, str) or not self.branch.strip():
            raise ValueError("activation dependency branch must not be blank")
        if not isinstance(self.frame, tuple):
            raise TypeError("activation dependency frame must be a tuple")
        for part in self.frame:
            if isinstance(part, bool) or not isinstance(part, (int, str)):
                raise ValueError("activation dependency frame values must be integers or strings")
            if isinstance(part, int) and part < 0:
                raise ValueError("activation dependency frame integers must be non-negative")
            if isinstance(part, str) and not part.strip():
                raise ValueError("activation dependency frame strings must not be blank")


@dataclass(frozen=True, slots=True)
class PlanNode:
    """An opaque executable node, prerequisites, and frame-local requirements."""

    node_id: NodeId
    class_name: str
    frame: Frame
    dependencies: tuple[NodeId, ...]
    activation_dependencies: tuple[ActivationDependency, ...]
    order: int


class ExecutionPlan:
    """Incrementally-built DAG of opaque executable nodes."""

    def __init__(self) -> None:
        self.nodes: dict[NodeId, PlanNode] = {}
        self._dependents: dict[NodeId, list[NodeId]] = {}
        self._next_order = 0

    def add_node(
        self,
        node_id: NodeId,
        class_name: str,
        frame: Frame = (),
        dependencies: Iterable[NodeId] = (),
        activation_dependencies: Iterable[ActivationDependency] = (),
    ) -> PlanNode:
        """Add one node, rejecting duplicate IDs and unknown prerequisites."""

        if not node_id.strip():
            raise ValueError("node id must not be blank")
        if not class_name.strip():
            raise ValueError("class name must not be blank")
        if node_id in self.nodes:
            raise ValueError(f"node already exists: {node_id}")
        # Repeated dependencies represent repeated execution-graph edges and
        # must remain distinct for indegree accounting.
        dependency_ids = tuple(dependencies)
        missing = [dependency for dependency in dependency_ids if dependency not in self.nodes]
        if missing:
            raise KeyError(f"unknown dependency for {node_id}: {missing[0]}")
        activation_dependency_values = tuple(activation_dependencies)
        for activation_dependency in activation_dependency_values:
            if not isinstance(activation_dependency, ActivationDependency):
                raise TypeError("activation dependencies must be ActivationDependency values")
            if not activation_dependency.owner_id.strip() or not activation_dependency.branch.strip():
                raise ValueError("activation dependency owner and branch must not be blank")
        node = PlanNode(
            node_id,
            class_name,
            tuple(frame),
            dependency_ids,
            activation_dependency_values,
            self._next_order,
        )
        self._next_order += 1
        self.nodes[node_id] = node
        self._dependents[node_id] = []
        for dependency in dependency_ids:
            self._dependents[dependency].append(node_id)
        return node

    def dependents(self, node_id: NodeId) -> tuple[NodeId, ...]:
        """Return direct dependents, validating the source node exists."""

        if node_id not in self.nodes:
            raise KeyError(f"unknown node: {node_id}")
        return tuple(self._dependents[node_id])

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe durable projection of this internal plan."""

        return {
            "nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "class_name": node.class_name,
                    "frame": list(node.frame),
                    "dependencies": list(node.dependencies),
                    "activation_dependencies": [
                        {
                            "owner_id": dependency.owner_id,
                            "branch": dependency.branch,
                            "frame": list(dependency.frame),
                        }
                        for dependency in node.activation_dependencies
                    ],
                    "order": node.order,
                }
                for node_id, node in self.nodes.items()
            },
            "next_order": self._next_order,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "ExecutionPlan":
        """Restore a plan using its insertion order and dependency list."""

        if not isinstance(snapshot, Mapping):
            raise ValueError("execution plan snapshot must be a mapping")
        plan = cls()
        raw_nodes = snapshot.get("nodes")
        if not isinstance(raw_nodes, Mapping):
            raise ValueError("execution plan snapshot must contain nodes")
        entries: list[tuple[int, Mapping[str, Any]]] = []
        for node_key, raw in raw_nodes.items():
            if not isinstance(raw, Mapping):
                raise ValueError("execution plan node must be a mapping")
            if raw.get("node_id") != node_key:
                raise ValueError("execution plan node key does not match node id")
            if not isinstance(raw.get("node_id"), str) or not isinstance(raw.get("class_name"), str):
                raise ValueError("execution plan node id and class name must be strings")
            order = raw.get("order")
            if not isinstance(order, int) or order < 0:
                raise ValueError("execution plan node order is invalid")
            frame = raw.get("frame", ())
            dependencies = raw.get("dependencies", ())
            activation_dependencies = raw.get("activation_dependencies", ())
            if (
                not isinstance(frame, (list, tuple))
                or not isinstance(dependencies, (list, tuple))
                or not isinstance(activation_dependencies, (list, tuple))
            ):
                raise ValueError("execution plan frame, dependencies, and activation dependencies must be sequences")
            if not all(isinstance(dependency, str) for dependency in dependencies):
                raise ValueError("execution plan dependencies must be node ids")
            parsed_activation_dependencies: list[ActivationDependency] = []
            for activation_dependency in activation_dependencies:
                if not isinstance(activation_dependency, Mapping):
                    raise ValueError("execution plan activation dependency must be a mapping")
                owner_id = activation_dependency.get("owner_id")
                branch = activation_dependency.get("branch")
                dependency_frame = activation_dependency.get("frame", ())
                if not isinstance(owner_id, str) or not isinstance(branch, str):
                    raise ValueError("execution plan activation dependency owner and branch must be strings")
                if not isinstance(dependency_frame, (list, tuple)):
                    raise ValueError("execution plan activation dependency frame must be a sequence")
                parsed_activation_dependencies.append(
                    ActivationDependency(owner_id=owner_id, branch=branch, frame=tuple(dependency_frame))
                )
            entries.append((order, {**raw, "_parsed_activation_dependencies": tuple(parsed_activation_dependencies)}))
        for _, raw in sorted(entries, key=lambda entry: entry[0]):
            plan.add_node(
                raw["node_id"],
                raw["class_name"],
                tuple(raw.get("frame", ())),
                tuple(raw.get("dependencies", ())),
                activation_dependencies=raw.get("_parsed_activation_dependencies", ()),
            )
        next_order = snapshot.get("next_order", plan._next_order)
        if not isinstance(next_order, int) or next_order < plan._next_order:
            raise ValueError("execution plan next order is invalid")
        plan._next_order = next_order
        return plan


class ExecutionScheduler:
    """Deterministic scheduler for an :class:`ExecutionPlan`."""

    def __init__(
        self,
        plan: ExecutionPlan,
        ready_order: Iterable[str] = (),
        executed: Iterable[NodeId] = (),
        ready_predicate: ReadyPredicate | None = None,
        discarded: Iterable[NodeId] = (),
    ) -> None:
        self.plan = plan
        self.ready_order = tuple(ready_order)
        self.executed: set[NodeId] = set(executed)
        self.discarded: set[NodeId] = set(discarded)
        self._ready_predicate = ready_predicate
        self._claimed: set[NodeId] = set()
        self.indegree: dict[NodeId, int] = {}
        self._enqueued: set[NodeId] = set()
        self._enqueued_by_class: dict[str, int] = {}
        self._ready_heaps: dict[str, list[tuple[Any, int, NodeId]]] = {}
        self._arrival_order: dict[NodeId, int] = {}
        self._next_arrival = 0
        self._active_class: str | None = None
        self.rebuild_ready()

    def _passes_ready_predicate(self, node_id: NodeId) -> bool:
        return self._ready_predicate is None or self._ready_predicate(node_id)

    def _enqueue_if_ready(self, node_id: NodeId) -> bool:
        node = self.plan.nodes.get(node_id)
        if node is None:
            raise KeyError(f"unknown node: {node_id}")
        if node_id in self.executed:
            raise ValueError(f"node already completed: {node_id}")
        if node_id in self.discarded:
            return False
        if node_id in self._claimed:
            return False
        if node_id not in self.indegree:
            raise KeyError(f"indegree missing for node: {node_id}")
        if self.indegree[node_id] != 0:
            raise ValueError(f"node is not ready: {node_id}")
        if not self._passes_ready_predicate(node_id):
            return False
        if node_id not in self._enqueued:
            self._enqueued.add(node_id)
            self._arrival_order[node_id] = self._next_arrival
            self._next_arrival += 1
            class_name = self.plan.nodes[node_id].class_name
            self._enqueued_by_class[class_name] = self._enqueued_by_class.get(class_name, 0) + 1
            heap = self._ready_heaps.setdefault(class_name, [])
            heapq.heappush(
                heap,
                (_frame_key(self.plan.nodes[node_id].frame), self._arrival_order[node_id], node_id),
            )
        return True

    def enqueue(self, node_id: NodeId) -> None:
        """Queue a currently-ready node when its opaque predicate permits it."""

        self._enqueue_if_ready(node_id)

    def _next_class(self) -> str | None:
        classes = {class_name for class_name, count in self._enqueued_by_class.items() if count}
        for class_name in self.ready_order:
            if class_name in classes:
                return class_name
        return min(classes) if classes else None

    def _next_id_for_class(self, class_name: str) -> NodeId:
        heap = self._ready_heaps[class_name]
        while heap and heap[0][2] not in self._enqueued:
            heapq.heappop(heap)
        if not heap:
            raise RuntimeError(f"ready queue for class {class_name!r} is empty")
        return heap[0][2]

    def pop_next(self) -> NodeId | None:
        """Remove and return the next ready node ID, or ``None`` when empty."""

        if not self._enqueued:
            return None
        if self._enqueued_by_class.get(self._active_class or "", 0) == 0:
            self._active_class = self._next_class()
        assert self._active_class is not None
        node_id = self._next_id_for_class(self._active_class)
        self._enqueued.remove(node_id)
        self._enqueued_by_class[self._active_class] -= 1
        heapq.heappop(self._ready_heaps[self._active_class])
        self._claimed.add(node_id)
        return node_id

    @property
    def ready_ids(self) -> tuple[NodeId, ...]:
        """Return queued IDs in the order they will be popped."""

        classes = {self.plan.nodes[node_id].class_name for node_id in self._enqueued}
        ordered_classes: list[str] = []
        if self._active_class in classes:
            ordered_classes.append(self._active_class)
        ordered_classes.extend(class_name for class_name in self.ready_order if class_name in classes)
        ordered_classes.extend(sorted(classes.difference(ordered_classes)))
        return tuple(
            node_id
            for class_name in ordered_classes
            for node_id in sorted(
                (node_id for node_id in self._enqueued if self.plan.nodes[node_id].class_name == class_name),
                key=lambda node_id: (
                    _frame_key(self.plan.nodes[node_id].frame),
                    self._arrival_order[node_id],
                ),
            )
        )

    def is_ready(self, node_id: NodeId) -> bool:
        """Return whether a node is currently queued without sorting the ready set."""

        return node_id in self._enqueued

    def add_node(self, node: PlanNode) -> None:
        """Add a plan node without disturbing already-claimed work."""

        if node.node_id not in self.plan.nodes:
            self.plan.add_node(
                node.node_id,
                node.class_name,
                node.frame,
                node.dependencies,
                node.activation_dependencies,
            )
        self.indegree[node.node_id] = sum(
            dependency not in self.executed and dependency not in self.discarded for dependency in node.dependencies
        )
        if (
            self.indegree[node.node_id] == 0
            and node.node_id not in self.executed
            and node.node_id not in self.discarded
            and node.node_id not in self._claimed
        ):
            self.enqueue(node.node_id)

    def discard(self, node_id: NodeId) -> tuple[NodeId, ...]:
        """Discard a node and satisfy its dependents without marking it executed."""

        if node_id not in self.plan.nodes:
            raise KeyError(f"unknown node: {node_id}")
        if node_id in self.executed:
            raise ValueError(f"node already completed: {node_id}")
        if node_id in self.discarded:
            return ()
        if node_id not in self.indegree:
            raise KeyError(f"indegree missing for node: {node_id}")
        dependents = self.plan.dependents(node_id)
        for dependent in dependents:
            if dependent not in self.indegree:
                raise KeyError(f"indegree missing for node: {dependent}")
            if self.indegree[dependent] <= 0:
                raise ValueError(f"dependency underflow for node: {dependent}")

        if node_id in self._enqueued:
            self._enqueued.discard(node_id)
            class_name = self.plan.nodes[node_id].class_name
            self._enqueued_by_class[class_name] -= 1
        self._claimed.discard(node_id)
        self._arrival_order.pop(node_id, None)
        self.discarded.add(node_id)
        newly_ready: list[NodeId] = []
        for dependent in dependents:
            self.indegree[dependent] -= 1
            if self.indegree[dependent] == 0 and dependent not in self.executed and dependent not in self.discarded:
                if self._enqueue_if_ready(dependent):
                    newly_ready.append(dependent)
        return tuple(newly_ready)

    def skip(self, node_id: NodeId) -> tuple[NodeId, ...]:
        """Alias for :meth:`discard` for callers expressing control-flow skips."""

        return self.discard(node_id)

    def set_ready_order(self, ready_order: Iterable[str]) -> None:
        """Change class priorities while retaining queued and claimed work."""

        self.ready_order = tuple(ready_order)

    def complete(self, node_id: NodeId) -> tuple[NodeId, ...]:
        """Mark node complete; return dependents newly made ready."""

        if node_id not in self.plan.nodes:
            raise KeyError(f"unknown node: {node_id}")
        if node_id in self.executed:
            raise ValueError(f"node already completed: {node_id}")
        if node_id in self.discarded:
            raise ValueError(f"node already discarded: {node_id}")
        if node_id not in self.indegree:
            raise KeyError(f"indegree missing for node: {node_id}")
        if self.indegree[node_id] != 0:
            raise ValueError(f"node is not ready: {node_id}")
        dependents = self.plan.dependents(node_id)
        for dependent in dependents:
            if dependent not in self.indegree:
                raise KeyError(f"indegree missing for node: {dependent}")
            if self.indegree[dependent] <= 0:
                raise ValueError(f"dependency underflow for node: {dependent}")
        self.executed.add(node_id)
        if node_id in self._enqueued:
            self._enqueued.discard(node_id)
            class_name = self.plan.nodes[node_id].class_name
            self._enqueued_by_class[class_name] -= 1
        self._claimed.discard(node_id)
        self._arrival_order.pop(node_id, None)
        newly_ready: list[NodeId] = []
        for dependent in dependents:
            self.indegree[dependent] -= 1
            if self.indegree[dependent] == 0 and dependent not in self.executed and dependent not in self.discarded:
                if self._enqueue_if_ready(dependent):
                    newly_ready.append(dependent)
        return tuple(newly_ready)

    def rebuild_ready(self) -> None:
        """Recompute indegrees and ready queue from plan and executed IDs."""

        overlapping = self.executed.intersection(self.discarded)
        if overlapping:
            raise ValueError(f"node cannot be both executed and discarded: {next(iter(overlapping))}")
        unknown = (self.executed | self.discarded).difference(self.plan.nodes)
        if unknown:
            node_id = next(iter(unknown))
            state = "executed" if node_id in self.executed else "discarded"
            raise KeyError(f"unknown {state} node: {node_id}")
        for node_id in self.executed:
            missing = [
                dependency
                for dependency in self.plan.nodes[node_id].dependencies
                if dependency not in self.executed and dependency not in self.discarded
            ]
            if missing:
                raise ValueError(f"executed node {node_id} is missing prerequisite: {missing[0]}")
        self.indegree = {
            node_id: sum(
                dependency not in self.executed and dependency not in self.discarded for dependency in node.dependencies
            )
            for node_id, node in self.plan.nodes.items()
        }
        previous_arrival = self._arrival_order
        self._enqueued = set()
        self._enqueued_by_class = {}
        self._ready_heaps = {}
        self._arrival_order = {}
        self._next_arrival = max(previous_arrival.values(), default=-1) + 1
        for node_id in self.plan.nodes:
            if (
                node_id not in self.executed
                and node_id not in self.discarded
                and node_id not in self._claimed
                and self.indegree[node_id] == 0
                and self._passes_ready_predicate(node_id)
            ):
                self._enqueued.add(node_id)
                arrival = previous_arrival.get(node_id)
                if arrival is None:
                    arrival = self._next_arrival
                    self._next_arrival += 1
                self._arrival_order[node_id] = arrival
                class_name = self.plan.nodes[node_id].class_name
                self._enqueued_by_class[class_name] = self._enqueued_by_class.get(class_name, 0) + 1
                heap = self._ready_heaps.setdefault(class_name, [])
                heapq.heappush(heap, (_frame_key(self.plan.nodes[node_id].frame), arrival, node_id))


__all__ = ["ActivationDependency", "ExecutionPlan", "ExecutionScheduler", "PlanNode", "ReadyPredicate"]
