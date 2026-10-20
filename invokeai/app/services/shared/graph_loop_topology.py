"""Loop topology: the For and nested-Iterate shapes of one source graph, classified once.

Validation rejects For bodies whose shape is not classified here. Graph state, the schedulers, and the planners read the
classification instead of re-running the recognizers for every completed node. The recognizers read only graph
structure (node types and edges), so a topology stays valid exactly as long as the graph and the flat source graph it
was built from.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping

from invokeai.app.invocations.call_saved_workflow import CallSavedWorkflowInvocation
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.services.shared.graph_validation import (
    COLLECTION_FIELD,
    ITEM_FIELD,
    CollectInvocation,
    Graph,
    IterateInvocation,
    nx,
)

if TYPE_CHECKING:
    import networkx

BodyPath = tuple[frozenset[str], str]
"""The For body nodes on a path to the linked ForReturn (return included), and that ForReturn's id."""


@dataclass(frozen=True)
class NestedForBody:
    """A For whose body contains direct child For loops, each with its own ForReturn."""

    body_path_nodes: frozenset[str]
    outer_return_id: str
    inner_for_ids: tuple[str, ...]
    continuation_nodes: frozenset[str]


@dataclass(frozen=True)
class NestedIterateBody:
    """A For whose body is one internal Iterate collapsed by one Collect before the ForReturn."""

    body_path_nodes: frozenset[str]
    return_node_id: str
    iterate_node_id: str
    collect_node_id: str


@dataclass(frozen=True)
class NestedIterateChain:
    """A For whose body is exactly two serial preparation/Iterate pairs, one body node, and one Collect."""

    body_path_nodes: frozenset[str]
    return_node_id: str
    iterate_node_ids: tuple[str, str]
    collect_node_id: str


@dataclass(frozen=True)
class NestedIterateSequence:
    """A whole graph that is one source feeding a serial chain of preparation/Iterate pairs and one body node."""

    body_path_nodes: frozenset[str]
    source_node_id: str
    iterate_node_ids: tuple[str, ...]
    preparation_node_ids: tuple[str, ...]
    body_node_id: str


ForBodyShape = NestedForBody | NestedIterateBody | NestedIterateChain

_CONTROL_TYPES = (
    CallSavedWorkflowInvocation,
    IfInvocation,
    ForInvocation,
    ForReturnInvocation,
    IterateInvocation,
    CollectInvocation,
)


@dataclass(frozen=True, eq=False)
class LoopTopology:
    """Loop shapes of one source graph. `flat_graph` is the cache key: a state reuses a topology only while its flat
    source graph is the one the topology was built from."""

    flat_graph: "networkx.DiGraph"
    for_body_paths: Mapping[str, BodyPath]
    """Body path of every For whose linked ForReturn is reachable from an iteration output."""
    for_body_shapes: Mapping[str, ForBodyShape]
    """Supported nested body of each For that has one, in source node order."""
    nested_iterate_sequence: NestedIterateSequence | None
    for_id_by_return_id: Mapping[str, str]

    @classmethod
    def classify(cls, graph: Graph, flat_graph: "networkx.DiGraph") -> "LoopTopology":
        """Classify every For once. `flat_graph` must be `graph.nx_graph_flat()` and acyclic."""
        for_ids = [node_id for node_id, node in graph.nodes.items() if isinstance(node, ForInvocation)]
        for_body_paths: dict[str, BodyPath] = {}
        for for_id in for_ids:
            body_path = graph._get_for_body_path_to_return(for_id, flat_graph)
            if body_path is not None:
                for_body_paths[for_id] = (frozenset(body_path[0]), body_path[1])

        nested_for_bodies: dict[str, NestedForBody | None] = {}

        def nested_for_body(for_id: str) -> NestedForBody | None:
            if for_id not in nested_for_bodies:
                nested_for_bodies[for_id] = _recognize_nested_for_body(
                    graph, flat_graph, for_id, for_body_paths, nested_for_body
                )
            return nested_for_bodies[for_id]

        # The three shapes are mutually exclusive: the chain has two Iterates on its body path, the Iterate body one,
        # and a nested For body none anywhere in its reachable body.
        for_body_shapes: dict[str, ForBodyShape] = {}
        for for_id in for_ids:
            body_path = for_body_paths.get(for_id)
            shape = (
                (_recognize_nested_iterate_chain(graph, flat_graph, for_id, body_path) if body_path else None)
                or (_recognize_nested_iterate_body(graph, flat_graph, for_id, body_path) if body_path else None)
                or nested_for_body(for_id)
            )
            if shape is not None:
                for_body_shapes[for_id] = shape

        return cls(
            flat_graph=flat_graph,
            for_body_paths=MappingProxyType(for_body_paths),
            for_body_shapes=MappingProxyType(for_body_shapes),
            nested_iterate_sequence=_recognize_nested_iterate_sequence(graph),
            for_id_by_return_id=MappingProxyType(
                {return_id: for_id for for_id, (_, return_id) in for_body_paths.items()}
            ),
        )

    # Immutable, and mappingproxy cannot be copied or pickled. A deep-copied state keeps this instance until its own
    # flat graph (a copy) fails the cache-key check and it reclassifies.
    def __copy__(self) -> "LoopTopology":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "LoopTopology":
        return self

    def __reduce__(self) -> tuple[Callable[..., "LoopTopology"], tuple[object, ...]]:
        # Pickling flat_graph alongside the state's flat graph keeps them one object, so the cache key still matches.
        return _restore_loop_topology, (
            self.flat_graph,
            dict(self.for_body_paths),
            dict(self.for_body_shapes),
            self.nested_iterate_sequence,
            dict(self.for_id_by_return_id),
        )

    def nested_for_body(self, for_id: str) -> NestedForBody | None:
        shape = self.for_body_shapes.get(for_id)
        return shape if isinstance(shape, NestedForBody) else None

    def nested_iterate_body(self, for_id: str) -> NestedIterateBody | None:
        shape = self.for_body_shapes.get(for_id)
        return shape if isinstance(shape, NestedIterateBody) else None

    def nested_iterate_chain(self, for_id: str) -> NestedIterateChain | None:
        shape = self.for_body_shapes.get(for_id)
        return shape if isinstance(shape, NestedIterateChain) else None


def _restore_loop_topology(
    flat_graph: "networkx.DiGraph",
    for_body_paths: dict[str, BodyPath],
    for_body_shapes: dict[str, ForBodyShape],
    nested_iterate_sequence: NestedIterateSequence | None,
    for_id_by_return_id: dict[str, str],
) -> LoopTopology:
    return LoopTopology(
        flat_graph=flat_graph,
        for_body_paths=MappingProxyType(for_body_paths),
        for_body_shapes=MappingProxyType(for_body_shapes),
        nested_iterate_sequence=nested_iterate_sequence,
        for_id_by_return_id=MappingProxyType(for_id_by_return_id),
    )


def _recognize_nested_iterate_body(
    graph: Graph, flat_graph: "networkx.DiGraph", node_id: str, body_path: BodyPath
) -> NestedIterateBody | None:
    """Recognize the bounded internal Iterate body.

    Supported shape::

        For.item -> Iterate.item -> body -> Collect.item
                                         Collect.collection -> ForReturn.output

    The Iterate and Collect nodes are scheduler-managed, so no other branch or final For output may escape this
    body contract.
    """
    body_path_nodes, return_node_id = body_path
    iterate_node_ids = [
        body_node_id for body_node_id in body_path_nodes if isinstance(graph.get_node(body_node_id), IterateInvocation)
    ]
    collect_node_ids = [
        body_node_id for body_node_id in body_path_nodes if isinstance(graph.get_node(body_node_id), CollectInvocation)
    ]
    if len(iterate_node_ids) != 1 or len(collect_node_ids) != 1:
        return None

    iterate_node_id = iterate_node_ids[0]
    collect_node_id = collect_node_ids[0]
    if not nx.has_path(flat_graph, iterate_node_id, collect_node_id):
        return None
    iterate_input_edges = graph._get_input_edges(iterate_node_id, COLLECTION_FIELD)
    if len(iterate_input_edges) != 1:
        return None
    iterate_input_source_id = iterate_input_edges[0].source.node_id
    if iterate_input_source_id != node_id and iterate_input_source_id not in body_path_nodes:
        return None

    return_output_edges = graph._get_input_edges(return_node_id, "output")
    if len(return_output_edges) != 1 or (
        return_output_edges[0].source.node_id != collect_node_id
        or return_output_edges[0].source.field != COLLECTION_FIELD
    ):
        return None
    if any(
        edge.destination.field != "output"
        and edge.destination.field != "continue_condition"
        and (edge.destination.field != "state" or edge.source.node_id != node_id or edge.source.field != "state")
        for edge in graph._get_input_edges(return_node_id)
    ):
        return None

    if graph._get_input_edges(collect_node_id, COLLECTION_FIELD):
        return None
    collect_item_edges = graph._get_input_edges(collect_node_id, ITEM_FIELD)
    if len(collect_item_edges) != 1:
        return None
    collect_item_source_id = collect_item_edges[0].source.node_id
    if not nx.has_path(flat_graph, iterate_node_id, collect_item_source_id):
        return None

    for body_node_id in body_path_nodes:
        if body_node_id in {iterate_node_id, collect_node_id, return_node_id}:
            continue
        if not nx.has_path(flat_graph, body_node_id, collect_node_id):
            return None
        if not (
            nx.has_path(flat_graph, body_node_id, iterate_node_id)
            or nx.has_path(flat_graph, iterate_node_id, body_node_id)
        ):
            return None

    return NestedIterateBody(
        body_path_nodes=body_path_nodes,
        return_node_id=return_node_id,
        iterate_node_id=iterate_node_id,
        collect_node_id=collect_node_id,
    )


def _recognize_nested_iterate_chain(
    graph: Graph, flat_graph: "networkx.DiGraph", node_id: str, body_path: BodyPath
) -> NestedIterateChain | None:
    """Recognize the exact two-level serial Iterate body.

    Supported shape::

        For.item -> prep1 -> Iterate1.item -> prep2 -> Iterate2.item -> body -> Collect.item
                                                                                     Collect.collection -> ForReturn.output

    This intentionally excludes sibling iterators, fan-in, and additional control-flow nodes. Those shapes must
    remain on the compatibility path until separately proven equivalent.
    """
    body_path_nodes, return_node_id = body_path
    iterate_node_ids = tuple(
        body_node_id for body_node_id in body_path_nodes if isinstance(graph.get_node(body_node_id), IterateInvocation)
    )
    collect_node_ids = tuple(
        body_node_id for body_node_id in body_path_nodes if isinstance(graph.get_node(body_node_id), CollectInvocation)
    )
    if len(iterate_node_ids) != 2 or len(collect_node_ids) != 1:
        return None

    collect_node_id = collect_node_ids[0]
    if len(body_path_nodes) != 7:
        return None
    return_output_edges = graph._get_input_edges(return_node_id, "output")
    if len(return_output_edges) != 1 or (
        return_output_edges[0].source.node_id != collect_node_id
        or return_output_edges[0].source.field != COLLECTION_FIELD
    ):
        return None
    if graph._get_input_edges(collect_node_id, COLLECTION_FIELD):
        return None
    collect_item_edges = graph._get_input_edges(collect_node_id, ITEM_FIELD)
    if len(collect_item_edges) != 1:
        return None

    first_iterate_id, second_iterate_id = sorted(
        iterate_node_ids, key=lambda iterate_id: nx.shortest_path_length(flat_graph, node_id, iterate_id)
    )
    if not nx.has_path(flat_graph, first_iterate_id, second_iterate_id) or not nx.has_path(
        flat_graph, second_iterate_id, collect_node_id
    ):
        return None
    first_input_edges = graph._get_input_edges(first_iterate_id, COLLECTION_FIELD)
    second_input_edges = graph._get_input_edges(second_iterate_id, COLLECTION_FIELD)
    if len(first_input_edges) != 1 or len(second_input_edges) != 1:
        return None
    first_preparation_id = first_input_edges[0].source.node_id
    second_preparation_id = second_input_edges[0].source.node_id
    body_node_id = collect_item_edges[0].source.node_id
    expected_nodes = {
        first_preparation_id,
        first_iterate_id,
        second_preparation_id,
        second_iterate_id,
        body_node_id,
        collect_node_id,
        return_node_id,
    }
    if body_path_nodes != expected_nodes or first_preparation_id == second_preparation_id:
        return None

    ordinary_nodes = {first_preparation_id, second_preparation_id, body_node_id}
    if any(isinstance(graph.get_node(candidate_id), _CONTROL_TYPES) for candidate_id in ordinary_nodes):
        return None
    if (
        first_input_edges[0].source.node_id != first_preparation_id
        or first_input_edges[0].source.field != COLLECTION_FIELD
        or second_input_edges[0].source.node_id != second_preparation_id
        or second_input_edges[0].source.field != COLLECTION_FIELD
    ):
        return None

    first_preparation_inputs = graph._get_input_edges(first_preparation_id)
    second_preparation_inputs = graph._get_input_edges(second_preparation_id)
    body_inputs = graph._get_input_edges(body_node_id)
    if (
        len(first_preparation_inputs) != 1
        or first_preparation_inputs[0].source.node_id != node_id
        or first_preparation_inputs[0].source.field != ITEM_FIELD
        or len(second_preparation_inputs) != 1
        or second_preparation_inputs[0].source.node_id != first_iterate_id
        or second_preparation_inputs[0].source.field != ITEM_FIELD
        or len(body_inputs) != 1
        or body_inputs[0].source.node_id != second_iterate_id
        or body_inputs[0].source.field != ITEM_FIELD
    ):
        return None
    if graph._get_output_edges(node_id, ITEM_FIELD) != first_preparation_inputs:
        return None
    if graph._get_output_edges(first_iterate_id, ITEM_FIELD) != second_preparation_inputs:
        return None
    if graph._get_output_edges(second_iterate_id, ITEM_FIELD) != body_inputs:
        return None
    if graph._get_output_edges(body_node_id) != collect_item_edges:
        return None

    return NestedIterateChain(
        body_path_nodes=body_path_nodes,
        return_node_id=return_node_id,
        iterate_node_ids=(first_iterate_id, second_iterate_id),
        collect_node_id=collect_node_id,
    )


def _recognize_nested_iterate_sequence(graph: Graph) -> NestedIterateSequence | None:
    """Recognize a whole graph that is one exact serial nested-Iterate chain of two or more Iterates."""
    # An n-Iterate chain has 2n + 1 nodes and 2n edges, so the edge count alone fixes the only possible depth.
    if len(graph.edges) % 2:
        return None
    iterate_count = len(graph.edges) // 2
    if iterate_count < 2 or len(graph.nodes) != iterate_count * 2 + 1:
        return None
    iterate_node_ids = tuple(node_id for node_id, node in graph.nodes.items() if isinstance(node, IterateInvocation))
    if len(iterate_node_ids) != iterate_count:
        return None
    outer_candidates = [
        node_id
        for node_id in iterate_node_ids
        if graph._get_input_edges(node_id, COLLECTION_FIELD)
        and not graph._get_input_edges(graph._get_input_edges(node_id, COLLECTION_FIELD)[0].source.node_id)
    ]
    if len(outer_candidates) != 1:
        return None

    ordered_iterate_ids = [outer_candidates[0]]
    preparation_node_ids: list[str] = []
    source_node_id = graph._get_input_edges(ordered_iterate_ids[0], COLLECTION_FIELD)[0].source.node_id
    remaining_iterate_ids = set(iterate_node_ids) - set(ordered_iterate_ids)
    for _ in range(iterate_count - 1):
        candidates: list[tuple[str, str]] = []
        for candidate_id in remaining_iterate_ids:
            collection_edges = graph._get_input_edges(candidate_id, COLLECTION_FIELD)
            if len(collection_edges) != 1:
                continue
            preparation_id = collection_edges[0].source.node_id
            preparation_inputs = graph._get_input_edges(preparation_id)
            if (
                len(preparation_inputs) == 1
                and preparation_inputs[0].source.node_id == ordered_iterate_ids[-1]
                and preparation_inputs[0].source.field == ITEM_FIELD
            ):
                candidates.append((candidate_id, preparation_id))
        if len(candidates) != 1:
            return None
        next_iterate_id, preparation_id = candidates[0]
        ordered_iterate_ids.append(next_iterate_id)
        preparation_node_ids.append(preparation_id)
        remaining_iterate_ids.remove(next_iterate_id)
    if remaining_iterate_ids:
        return None

    body_candidates = [
        node_id
        for node_id in graph.nodes
        if node_id not in set(ordered_iterate_ids) | set(preparation_node_ids) | {source_node_id}
    ]
    if len(body_candidates) != 1:
        return None
    body_node_id = body_candidates[0]
    if any(
        isinstance(graph.get_node(node_id), _CONTROL_TYPES)
        for node_id in {source_node_id, *preparation_node_ids, body_node_id}
    ):
        return None
    outer_collection_edges = graph._get_input_edges(ordered_iterate_ids[0], COLLECTION_FIELD)
    body_inputs = graph._get_input_edges(body_node_id)
    if graph._get_input_edges(source_node_id) or len(outer_collection_edges) != 1 or len(body_inputs) != 1:
        return None
    if graph._get_output_edges(source_node_id) != outer_collection_edges:
        return None
    for index, preparation_id in enumerate(preparation_node_ids):
        preparation_inputs = graph._get_input_edges(preparation_id)
        next_collection_edges = graph._get_input_edges(ordered_iterate_ids[index + 1], COLLECTION_FIELD)
        if (
            len(preparation_inputs) != 1
            or preparation_inputs[0].source.node_id != ordered_iterate_ids[index]
            or preparation_inputs[0].source.field != ITEM_FIELD
            or len(next_collection_edges) != 1
            or next_collection_edges[0].source.node_id != preparation_id
            or graph._get_output_edges(ordered_iterate_ids[index], ITEM_FIELD) != preparation_inputs
            or graph._get_output_edges(preparation_id) != next_collection_edges
        ):
            return None
    if (
        body_inputs[0].source.node_id != ordered_iterate_ids[-1]
        or body_inputs[0].source.field != ITEM_FIELD
        or graph._get_output_edges(ordered_iterate_ids[-1], ITEM_FIELD) != body_inputs
        or graph._get_output_edges(body_node_id)
    ):
        return None
    return NestedIterateSequence(
        body_path_nodes=frozenset(graph.nodes),
        source_node_id=source_node_id,
        iterate_node_ids=tuple(ordered_iterate_ids),
        preparation_node_ids=tuple(preparation_node_ids),
        body_node_id=body_node_id,
    )


def _recognize_nested_for_body(
    graph: Graph,
    flat_graph: "networkx.DiGraph",
    node_id: str,
    for_body_paths: Mapping[str, BodyPath],
    nested_for_body: Callable[[str], "NestedForBody | None"],
) -> NestedForBody | None:
    """Recognize the supported recursive nested For body.

    Each direct child loop has its own ForReturn. A single child may close the parent directly or through a
    continuation. Multiple independent child loops must all feed an ordinary parent-scoped continuation, which acts
    as an explicit fan-in barrier after every child has finalized for the current parent iteration.

    Accepted shape (the child body can recursively contain this same shape)::

        outer For -> preparation -> inner For(s) -> continuation -> outer ForReturn
                                       |    ^
                                       v    |
                                     child body -> inner ForReturn

    Preparation, child bodies, and continuation partition the reachable outer body. Only finalized child outputs
    may cross into the continuation; iteration outputs stay inside their owning child body. `nested_for_body`
    resolves a child For's own nested body.
    """
    iteration_edges = graph._get_for_iteration_output_edges(node_id)
    if len(iteration_edges) == 0:
        return None
    reachable_body_nodes = graph._get_for_reachable_body_nodes(iteration_edges, flat_graph)
    reachable_return_ids = [
        body_node_id
        for body_node_id in reachable_body_nodes
        if isinstance(graph.get_node(body_node_id), ForReturnInvocation)
    ]
    outer_return_id = graph._get_linked_for_return_id(node_id)
    if outer_return_id is None or outer_return_id not in reachable_return_ids:
        return None

    nested_for_ids = [
        body_node_id
        for body_node_id in reachable_body_nodes
        if isinstance(graph.get_node(body_node_id), ForInvocation) and body_node_id != node_id
    ]
    direct_nested_for_ids = [
        nested_for_id
        for nested_for_id in nested_for_ids
        if not any(
            other_nested_for_id != nested_for_id and nx.has_path(flat_graph, other_nested_for_id, nested_for_id)
            for other_nested_for_id in nested_for_ids
        )
    ]
    if not direct_nested_for_ids:
        return None
    direct_nested_for_ids = tuple(
        nested_for_id for nested_for_id in nx.topological_sort(flat_graph) if nested_for_id in direct_nested_for_ids
    )

    inner_body_path_nodes: set[str] = set()
    for inner_for_id in direct_nested_for_ids:
        inner_return_id = graph._get_linked_for_return_id(inner_for_id)
        if inner_return_id is None:
            return None

        inner_body_path_to_return = for_body_paths.get(inner_for_id)
        if inner_body_path_to_return is None:
            return None
        child_body_path_nodes, resolved_inner_return_id = inner_body_path_to_return
        if resolved_inner_return_id != inner_return_id:
            return None
        if inner_return_id not in reachable_return_ids:
            return None

        inner_nested_for_ids = [
            body_node_id
            for body_node_id in child_body_path_nodes
            if isinstance(graph.get_node(body_node_id), ForInvocation)
        ]
        if any(isinstance(graph.get_node(body_node_id), IterateInvocation) for body_node_id in child_body_path_nodes):
            return None
        inner_nested_body = nested_for_body(inner_for_id) if inner_nested_for_ids else None
        if inner_nested_for_ids and inner_nested_body is None:
            return None
        if inner_nested_body is not None:
            child_body_path_nodes = child_body_path_nodes | inner_nested_body.body_path_nodes
        if any(
            edge.destination.field == "state"
            and edge.source.node_id != inner_for_id
            and edge.source.node_id not in child_body_path_nodes
            for edge in graph._get_input_edges(inner_return_id)
        ):
            return None

        inner_collection_edges = graph._get_input_edges(inner_for_id, COLLECTION_FIELD)
        if len(inner_collection_edges) != 1:
            return None
        inner_collection_source_id = inner_collection_edges[0].source.node_id
        if inner_collection_source_id != node_id and inner_collection_source_id not in reachable_body_nodes:
            return None

        inner_body_path_nodes.update(child_body_path_nodes)

    if set(reachable_return_ids) - inner_body_path_nodes != {outer_return_id}:
        return None

    outer_output_edges = graph._get_input_edges(outer_return_id, "output")
    if len(outer_output_edges) != 1:
        return None
    if any(
        edge.destination.field != "output"
        and edge.destination.field != "continue_condition"
        and (edge.destination.field != "state" or edge.source.node_id != node_id or edge.source.field != "state")
        for edge in graph._get_input_edges(outer_return_id)
    ):
        return None

    outer_preparation_nodes = {
        body_node_id
        for inner_for_id in direct_nested_for_ids
        for body_node_id in reachable_body_nodes & nx.ancestors(flat_graph, inner_for_id)
    } | set(direct_nested_for_ids)
    inner_final_descendants: set[str] = set()
    for inner_for_id in direct_nested_for_ids:
        for edge in graph._get_for_final_output_edges(inner_for_id):
            inner_final_descendants.add(edge.destination.node_id)
            inner_final_descendants.update(nx.descendants(flat_graph, edge.destination.node_id))
    continuation_nodes = reachable_body_nodes - outer_preparation_nodes - inner_body_path_nodes - {outer_return_id}
    if any(
        edge.destination.field == "continue_condition"
        and edge.source.node_id != node_id
        and edge.source.node_id not in continuation_nodes
        and not (
            edge.source.node_id in direct_nested_for_ids and edge.source.field in {"output_collection", "final_state"}
        )
        for edge in graph._get_input_edges(outer_return_id)
    ):
        return None
    if not continuation_nodes <= inner_final_descendants:
        return None
    if any(not nx.has_path(flat_graph, body_node_id, outer_return_id) for body_node_id in continuation_nodes):
        return None
    if any(
        isinstance(graph.get_node(body_node_id), (ForInvocation, IterateInvocation, ForReturnInvocation))
        for body_node_id in continuation_nodes
    ):
        return None
    if any(
        edge.source.node_id in inner_body_path_nodes
        or (edge.source.node_id in direct_nested_for_ids and edge.source.field != "output_collection")
        for body_node_id in continuation_nodes
        for edge in graph._get_input_edges(body_node_id)
    ):
        return None
    output_source_id = outer_output_edges[0].source.node_id
    if output_source_id in direct_nested_for_ids:
        if (
            len(direct_nested_for_ids) != 1
            or outer_output_edges[0].source.field != "output_collection"
            or continuation_nodes
        ):
            return None
    elif output_source_id not in continuation_nodes:
        return None

    if any(
        not any(
            edge.destination.node_id in continuation_nodes or edge.destination.node_id == outer_return_id
            for edge in graph._get_for_final_output_edges(inner_for_id)
        )
        for inner_for_id in direct_nested_for_ids
    ):
        return None

    allowed_body_nodes = outer_preparation_nodes | inner_body_path_nodes | continuation_nodes | {outer_return_id}
    if reachable_body_nodes != allowed_body_nodes:
        return None

    if any(
        isinstance(graph.get_node(body_node_id), (ForInvocation, IterateInvocation))
        for body_node_id in outer_preparation_nodes
        if body_node_id not in direct_nested_for_ids
    ):
        return None

    for body_node_id in allowed_body_nodes - {outer_return_id, *direct_nested_for_ids}:
        if body_node_id in inner_body_path_nodes or body_node_id in continuation_nodes:
            continue
        if not any(nx.has_path(flat_graph, body_node_id, inner_for_id) for inner_for_id in direct_nested_for_ids):
            return None
    return NestedForBody(
        body_path_nodes=frozenset(allowed_body_nodes),
        outer_return_id=outer_return_id,
        inner_for_ids=direct_nested_for_ids,
        continuation_nodes=frozenset(continuation_nodes),
    )
