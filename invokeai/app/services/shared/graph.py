# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import copy
import itertools
from typing import Any, Optional, TypeVar, Union, get_args, get_origin, get_type_hints

import networkx as nx
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationError,
    field_validator,
)
from pydantic.fields import Field
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

# Importing * is bad karma but needed here for node detection
from invokeai.app.invocations import *  # noqa: F401 F403
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import Input, InputField, OutputField, UIType
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import uuid_string

# in 3.10 this would be "from types import NoneType"
NoneType = type(None)


class EdgeConnection(BaseModel):
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
    source: EdgeConnection = Field(description="The connection for the edge's from node and field")
    destination: EdgeConnection = Field(description="The connection for the edge's to node and field")

    def __str__(self):
        return f"{self.source.node_id}.{self.source.field} -> {self.destination.node_id}.{self.destination.field}"


def get_output_field_type(node: BaseInvocation, field: str) -> Any:
    node_type = type(node)
    node_outputs = get_type_hints(node_type.get_output_annotation())
    node_output_field = node_outputs.get(field) or None
    return node_output_field


def get_input_field_type(node: BaseInvocation, field: str) -> Any:
    node_type = type(node)
    node_inputs = get_type_hints(node_type)
    node_input_field = node_inputs.get(field) or None
    return node_input_field


def is_union_subtype(t1, t2):
    t1_args = get_args(t1)
    t2_args = get_args(t2)
    if not t1_args:
        # t1 is a single type
        return t1 in t2_args
    else:
        # t1 is a Union, check that all of its types are in t2_args
        return all(arg in t2_args for arg in t1_args)


def is_list_or_contains_list(t):
    t_args = get_args(t)

    # If the type is a List
    if get_origin(t) is list:
        return True

    # If the type is a Union
    elif t_args:
        # Check if any of the types in the Union is a List
        for arg in t_args:
            if get_origin(arg) is list:
                return True
    return False


def is_any(t: Any) -> bool:
    return t == Any or Any in get_args(t)


def are_connection_types_compatible(from_type: Any, to_type: Any) -> bool:
    if not from_type:
        return False
    if not to_type:
        return False

    # TODO: this is pretty forgiving on generic types. Clean that up (need to handle optionals and such)
    if from_type and to_type:
        # Ports are compatible
        if from_type == to_type or is_any(from_type) or is_any(to_type):
            return True

        if from_type in get_args(to_type):
            return True

        if to_type in get_args(from_type):
            return True

        # allow int -> float, pydantic will cast for us
        if from_type is int and to_type is float:
            return True

        # allow int|float -> str, pydantic will cast for us
        if (from_type is int or from_type is float) and to_type is str:
            return True

        # if not issubclass(from_type, to_type):
        if not is_union_subtype(from_type, to_type):
            return False
    else:
        return False

    return True


def are_connections_compatible(
    from_node: BaseInvocation, from_field: str, to_node: BaseInvocation, to_field: str
) -> bool:
    """Determines if a connection between fields of two nodes is compatible."""

    # TODO: handle iterators and collectors
    from_type = get_output_field_type(from_node, from_field)
    to_type = get_input_field_type(to_node, to_field)

    return are_connection_types_compatible(from_type, to_type)


T = TypeVar("T")


def copydeep(obj: T) -> T:
    """Deep-copies an object. If it is a pydantic model, use the model's copy method."""
    if isinstance(obj, BaseModel):
        return obj.model_copy(deep=True)
    return copy.deepcopy(obj)


class NodeAlreadyInGraphError(ValueError):
    pass


class InvalidEdgeError(ValueError):
    pass


class NodeNotFoundError(ValueError):
    pass


class NodeAlreadyExecutedError(ValueError):
    pass


class DuplicateNodeIdError(ValueError):
    pass


class NodeFieldNotFoundError(ValueError):
    pass


class NodeIdMismatchError(ValueError):
    pass


class CyclicalGraphError(ValueError):
    pass


class UnknownGraphValidationError(ValueError):
    pass


class NodeInputError(ValueError):
    """Raised when a node fails preparation. This occurs when a node's inputs are being set from its incomers, but an
    input fails validation.

    Attributes:
        node: The node that failed preparation. Note: only successfully set fields will be accurate. Review the error to
            determine which field caused the failure.
    """

    def __init__(self, node: BaseInvocation, e: ValidationError):
        self.original_error = e
        self.node = node
        # When preparing a node, we set each input one-at-a-time. We may thus safely assume that the first error
        # represents the first input that failed.
        self.failed_input = loc_to_dot_sep(e.errors()[0]["loc"])
        super().__init__(f"Node {node.id} has invalid incoming input for {self.failed_input}")


def loc_to_dot_sep(loc: tuple[Union[str, int], ...]) -> str:
    """Helper to pretty-print pydantic error locations as dot-separated strings.
    Taken from https://docs.pydantic.dev/latest/errors/errors/#customize-error-messages
    """
    path = ""
    for i, x in enumerate(loc):
        if isinstance(x, str):
            if i > 0:
                path += "."
            path += x
        else:
            path += f"[{x}]"
    return path


@invocation_output("iterate_output")
class IterateInvocationOutput(BaseInvocationOutput):
    """Used to connect iteration outputs. Will be expanded to a specific output."""

    item: Any = OutputField(
        description="The item being iterated over", title="Collection Item", ui_type=UIType._CollectionItem
    )
    index: int = OutputField(description="The index of the item", title="Index")
    total: int = OutputField(description="The total number of items", title="Total")


# TODO: Fill this out and move to invocations
@invocation("iterate", version="1.1.0")
class IterateInvocation(BaseInvocation):
    """Iterates over a list of items"""

    collection: list[Any] = InputField(
        description="The list of items to iterate over", default=[], ui_type=UIType._Collection
    )
    index: int = InputField(description="The index, will be provided on executed iterators", default=0, ui_hidden=True)

    def invoke(self, context: InvocationContext) -> IterateInvocationOutput:
        """Produces the outputs as values"""
        return IterateInvocationOutput(item=self.collection[self.index], index=self.index, total=len(self.collection))


@invocation_output("collect_output")
class CollectInvocationOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(
        description="The collection of input items", title="Collection", ui_type=UIType._Collection
    )


@invocation("collect", version="1.0.0")
class CollectInvocation(BaseInvocation):
    """Collects values into a collection"""

    item: Optional[Any] = InputField(
        default=None,
        description="The item to collect (all inputs must be of the same type)",
        ui_type=UIType._CollectionItem,
        title="Collection Item",
        input=Input.Connection,
    )
    collection: list[Any] = InputField(
        description="The collection, will be provided on execution", default=[], ui_hidden=True
    )

    def invoke(self, context: InvocationContext) -> CollectInvocationOutput:
        """Invoke with provided services and return outputs."""
        return CollectInvocationOutput(collection=copy.copy(self.collection))


class AnyInvocation(BaseInvocation):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        def validate_invocation(v: Any) -> "AnyInvocation":
            return BaseInvocation.get_typeadapter().validate_python(v)

        return core_schema.no_info_plain_validator_function(validate_invocation)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Nodes are too powerful, we have to make our own OpenAPI schema manually
        # No but really, because the schema is dynamic depending on loaded nodes, we need to generate it manually
        oneOf: list[dict[str, str]] = []
        names = [i.__name__ for i in BaseInvocation.get_invocations()]
        for name in sorted(names):
            oneOf.append({"$ref": f"#/components/schemas/{name}"})
        return {"oneOf": oneOf}


class AnyInvocationOutput(BaseInvocationOutput):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler):
        def validate_invocation_output(v: Any) -> "AnyInvocationOutput":
            return BaseInvocationOutput.get_typeadapter().validate_python(v)

        return core_schema.no_info_plain_validator_function(validate_invocation_output)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Nodes are too powerful, we have to make our own OpenAPI schema manually
        # No but really, because the schema is dynamic depending on loaded nodes, we need to generate it manually

        oneOf: list[dict[str, str]] = []
        names = [i.__name__ for i in BaseInvocationOutput.get_outputs()]
        for name in sorted(names):
            oneOf.append({"$ref": f"#/components/schemas/{name}"})
        return {"oneOf": oneOf}


class Graph(BaseModel):
    id: str = Field(description="The id of this graph", default_factory=uuid_string)
    # TODO: use a list (and never use dict in a BaseModel) because pydantic/fastapi hates me
    nodes: dict[str, AnyInvocation] = Field(description="The nodes in this graph", default_factory=dict)
    edges: list[Edge] = Field(
        description="The connections between nodes and their fields in this graph",
        default_factory=list,
    )

    def add_node(self, node: BaseInvocation) -> None:
        """Adds a node to a graph

        :raises NodeAlreadyInGraphError: the node is already present in the graph.
        """

        if node.id in self.nodes:
            raise NodeAlreadyInGraphError()

        self.nodes[node.id] = node

    def delete_node(self, node_id: str) -> None:
        """Deletes a node from a graph"""

        try:
            # Delete edges for this node
            input_edges = self._get_input_edges(node_id)
            output_edges = self._get_output_edges(node_id)

            for edge in input_edges:
                self.delete_edge(edge)

            for edge in output_edges:
                self.delete_edge(edge)

            del self.nodes[node_id]

        except NodeNotFoundError:
            pass  # Ignore, not doesn't exist (should this throw?)

    def add_edge(self, edge: Edge) -> None:
        """Adds an edge to a graph

        :raises InvalidEdgeError: the provided edge is invalid.
        """

        self._validate_edge(edge)
        if edge not in self.edges:
            self.edges.append(edge)
        else:
            raise InvalidEdgeError()

    def delete_edge(self, edge: Edge) -> None:
        """Deletes an edge from a graph"""

        try:
            self.edges.remove(edge)
        except KeyError:
            pass

    def validate_self(self) -> None:
        """
        Validates the graph.

        Raises an exception if the graph is invalid:
        - `DuplicateNodeIdError`
        - `NodeIdMismatchError`
        - `InvalidSubGraphError`
        - `NodeNotFoundError`
        - `NodeFieldNotFoundError`
        - `CyclicalGraphError`
        - `InvalidEdgeError`
        """

        # Validate that all node ids are unique
        node_ids = [n.id for n in self.nodes.values()]
        duplicate_node_ids = {node_id for node_id in node_ids if node_ids.count(node_id) >= 2}
        if duplicate_node_ids:
            raise DuplicateNodeIdError(f"Node ids must be unique, found duplicates {duplicate_node_ids}")

        # Validate that all node ids match the keys in the nodes dict
        for k, v in self.nodes.items():
            if k != v.id:
                raise NodeIdMismatchError(f"Node ids must match, got {k} and {v.id}")

        # Validate that all edges match nodes and fields in the graph
        for edge in self.edges:
            source_node = self.nodes.get(edge.source.node_id, None)
            if source_node is None:
                raise NodeNotFoundError(f"Edge source node {edge.source.node_id} does not exist in the graph")

            destination_node = self.nodes.get(edge.destination.node_id, None)
            if destination_node is None:
                raise NodeNotFoundError(f"Edge destination node {edge.destination.node_id} does not exist in the graph")

            # output fields are not on the node object directly, they are on the output type
            if edge.source.field not in source_node.get_output_annotation().model_fields:
                raise NodeFieldNotFoundError(
                    f"Edge source field {edge.source.field} does not exist in node {edge.source.node_id}"
                )

            # input fields are on the node
            if edge.destination.field not in destination_node.model_fields:
                raise NodeFieldNotFoundError(
                    f"Edge destination field {edge.destination.field} does not exist in node {edge.destination.node_id}"
                )

        # Validate there are no cycles
        g = self.nx_graph_flat()
        if not nx.is_directed_acyclic_graph(g):
            raise CyclicalGraphError("Graph contains cycles")

        # Validate all edge connections are valid
        for edge in self.edges:
            if not are_connections_compatible(
                self.get_node(edge.source.node_id),
                edge.source.field,
                self.get_node(edge.destination.node_id),
                edge.destination.field,
            ):
                raise InvalidEdgeError(f"Edge source and target types do not match ({edge})")

        # Validate all iterators & collectors
        # TODO: may need to validate all iterators & collectors in subgraphs so edge connections in parent graphs will be available
        for node in self.nodes.values():
            if isinstance(node, IterateInvocation):
                err = self._is_iterator_connection_valid(node.id)
                if err is not None:
                    raise InvalidEdgeError(f"Invalid iterator node ({node.id}): {err}")
            if isinstance(node, CollectInvocation):
                err = self._is_collector_connection_valid(node.id)
                if err is not None:
                    raise InvalidEdgeError(f"Invalid collector node ({node.id}): {err}")

        return None

    def is_valid(self) -> bool:
        """
        Checks if the graph is valid.

        Raises `UnknownGraphValidationError` if there is a problem validating the graph (not a validation error).
        """
        try:
            self.validate_self()
            return True
        except (
            DuplicateNodeIdError,
            NodeIdMismatchError,
            NodeNotFoundError,
            NodeFieldNotFoundError,
            CyclicalGraphError,
            InvalidEdgeError,
        ):
            return False
        except Exception as e:
            raise UnknownGraphValidationError(f"Problem validating graph {e}") from e

    def _is_destination_field_Any(self, edge: Edge) -> bool:
        """Checks if the destination field for an edge is of type typing.Any"""
        return get_input_field_type(self.get_node(edge.destination.node_id), edge.destination.field) == Any

    def _is_destination_field_list_of_Any(self, edge: Edge) -> bool:
        """Checks if the destination field for an edge is of type typing.Any"""
        return get_input_field_type(self.get_node(edge.destination.node_id), edge.destination.field) == list[Any]

    def _validate_edge(self, edge: Edge):
        """Validates that a new edge doesn't create a cycle in the graph"""

        # Validate that the nodes exist
        try:
            from_node = self.get_node(edge.source.node_id)
            to_node = self.get_node(edge.destination.node_id)
        except NodeNotFoundError:
            raise InvalidEdgeError(f"One or both nodes don't exist ({edge})")

        # Validate that an edge to this node+field doesn't already exist
        input_edges = self._get_input_edges(edge.destination.node_id, edge.destination.field)
        if len(input_edges) > 0 and not isinstance(to_node, CollectInvocation):
            raise InvalidEdgeError(f"Edge already exists ({edge})")

        # Validate that no cycles would be created
        g = self.nx_graph_flat()
        g.add_edge(edge.source.node_id, edge.destination.node_id)
        if not nx.is_directed_acyclic_graph(g):
            raise InvalidEdgeError(f"Edge creates a cycle in the graph ({edge})")

        # Validate that the field types are compatible
        if not are_connections_compatible(from_node, edge.source.field, to_node, edge.destination.field):
            raise InvalidEdgeError(f"Field types are incompatible ({edge})")

        # Validate if iterator output type matches iterator input type (if this edge results in both being set)
        if isinstance(to_node, IterateInvocation) and edge.destination.field == "collection":
            err = self._is_iterator_connection_valid(edge.destination.node_id, new_input=edge.source)
            if err is not None:
                raise InvalidEdgeError(f"Iterator input type does not match iterator output type ({edge}): {err}")

        # Validate if iterator input type matches output type (if this edge results in both being set)
        if isinstance(from_node, IterateInvocation) and edge.source.field == "item":
            err = self._is_iterator_connection_valid(edge.source.node_id, new_output=edge.destination)
            if err is not None:
                raise InvalidEdgeError(f"Iterator output type does not match iterator input type ({edge}): {err}")

        # Validate if collector input type matches output type (if this edge results in both being set)
        if isinstance(to_node, CollectInvocation) and edge.destination.field == "item":
            err = self._is_collector_connection_valid(edge.destination.node_id, new_input=edge.source)
            if err is not None:
                raise InvalidEdgeError(f"Collector output type does not match collector input type ({edge}): {err}")

        # Validate if collector output type matches input type (if this edge results in both being set) - skip if the destination field is not Any or list[Any]
        if (
            isinstance(from_node, CollectInvocation)
            and edge.source.field == "collection"
            and not self._is_destination_field_list_of_Any(edge)
            and not self._is_destination_field_Any(edge)
        ):
            err = self._is_collector_connection_valid(edge.source.node_id, new_output=edge.destination)
            if err is not None:
                raise InvalidEdgeError(f"Collector input type does not match collector output type ({edge}): {err}")

    def has_node(self, node_id: str) -> bool:
        """Determines whether or not a node exists in the graph."""
        try:
            _ = self.get_node(node_id)
            return True
        except NodeNotFoundError:
            return False

    def get_node(self, node_id: str) -> BaseInvocation:
        """Gets a node from the graph."""
        try:
            return self.nodes[node_id]
        except KeyError as e:
            raise NodeNotFoundError(f"Node {node_id} not found in graph") from e

    def update_node(self, node_id: str, new_node: BaseInvocation) -> None:
        """Updates a node in the graph."""
        node = self.nodes[node_id]

        # Ensure the node type matches the new node
        if type(node) is not type(new_node):
            raise TypeError(f"Node {node_id} is type {type(node)} but new node is type {type(new_node)}")

        # Ensure the new id is either the same or is not in the graph
        if new_node.id != node.id and self.has_node(new_node.id):
            raise NodeAlreadyInGraphError(f"Node with id {new_node.id} already exists in graph")

        # Set the new node in the graph
        self.nodes[new_node.id] = new_node
        if new_node.id != node.id:
            input_edges = self._get_input_edges(node_id)
            output_edges = self._get_output_edges(node_id)

            # Delete node and all edges
            self.delete_node(node_id)

            # Create new edges for each input and output
            for edge in input_edges:
                self.add_edge(
                    Edge(
                        source=edge.source,
                        destination=EdgeConnection(node_id=new_node.id, field=edge.destination.field),
                    )
                )

            for edge in output_edges:
                self.add_edge(
                    Edge(
                        source=EdgeConnection(node_id=new_node.id, field=edge.source.field),
                        destination=edge.destination,
                    )
                )

    def _get_input_edges(self, node_id: str, field: Optional[str] = None) -> list[Edge]:
        """Gets all input edges for a node. If field is provided, only edges to that field are returned."""

        edges = [e for e in self.edges if e.destination.node_id == node_id]

        if field is None:
            return edges

        filtered_edges = [e for e in edges if e.destination.field == field]

        return filtered_edges

    def _get_output_edges(self, node_id: str, field: Optional[str] = None) -> list[Edge]:
        """Gets all output edges for a node. If field is provided, only edges from that field are returned."""
        edges = [e for e in self.edges if e.source.node_id == node_id]

        if field is None:
            return edges

        filtered_edges = [e for e in edges if e.source.field == field]

        return filtered_edges

    def _is_iterator_connection_valid(
        self,
        node_id: str,
        new_input: Optional[EdgeConnection] = None,
        new_output: Optional[EdgeConnection] = None,
    ) -> str | None:
        inputs = [e.source for e in self._get_input_edges(node_id, "collection")]
        outputs = [e.destination for e in self._get_output_edges(node_id, "item")]

        if new_input is not None:
            inputs.append(new_input)
        if new_output is not None:
            outputs.append(new_output)

        # Only one input is allowed for iterators
        if len(inputs) > 1:
            return "Iterator may only have one input edge"

        input_node = self.get_node(inputs[0].node_id)

        # Get input and output fields (the fields linked to the iterator's input/output)
        input_field_type = get_output_field_type(input_node, inputs[0].field)
        output_field_types = [get_input_field_type(self.get_node(e.node_id), e.field) for e in outputs]

        # Input type must be a list
        if get_origin(input_field_type) is not list:
            return "Iterator input must be a collection"

        # Validate that all outputs match the input type
        input_field_item_type = get_args(input_field_type)[0]
        if not all((are_connection_types_compatible(input_field_item_type, t) for t in output_field_types)):
            return "Iterator outputs must connect to an input with a matching type"

        # Collector input type must match all iterator output types
        if isinstance(input_node, CollectInvocation):
            # Traverse the graph to find the first collector input edge. Collectors validate that their collection
            # inputs are all of the same type, so we can use the first input edge to determine the collector's type
            first_collector_input_edge = self._get_input_edges(input_node.id, "item")[0]
            first_collector_input_type = get_output_field_type(
                self.get_node(first_collector_input_edge.source.node_id), first_collector_input_edge.source.field
            )
            resolved_collector_type = (
                first_collector_input_type
                if get_origin(first_collector_input_type) is None
                else get_args(first_collector_input_type)
            )
            if not all((are_connection_types_compatible(resolved_collector_type, t) for t in output_field_types)):
                return "Iterator collection type must match all iterator output types"

        return None

    def _is_collector_connection_valid(
        self,
        node_id: str,
        new_input: Optional[EdgeConnection] = None,
        new_output: Optional[EdgeConnection] = None,
    ) -> str | None:
        inputs = [e.source for e in self._get_input_edges(node_id, "item")]
        outputs = [e.destination for e in self._get_output_edges(node_id, "collection")]

        if new_input is not None:
            inputs.append(new_input)
        if new_output is not None:
            outputs.append(new_output)

        # Get input and output fields (the fields linked to the iterator's input/output)
        input_field_types = [get_output_field_type(self.get_node(e.node_id), e.field) for e in inputs]
        output_field_types = [get_input_field_type(self.get_node(e.node_id), e.field) for e in outputs]

        # Validate that all inputs are derived from or match a single type
        input_field_types = {
            resolved_type
            for input_field_type in input_field_types
            for resolved_type in (
                [input_field_type] if get_origin(input_field_type) is None else get_args(input_field_type)
            )
            if resolved_type != NoneType
        }  # Get unique types
        type_tree = nx.DiGraph()
        type_tree.add_nodes_from(input_field_types)
        type_tree.add_edges_from([e for e in itertools.permutations(input_field_types, 2) if issubclass(e[1], e[0])])
        type_degrees = type_tree.in_degree(type_tree.nodes)
        if sum((t[1] == 0 for t in type_degrees)) != 1:  # type: ignore
            return "Collector input collection items must be of a single type"

        # Get the input root type
        input_root_type = next(t[0] for t in type_degrees if t[1] == 0)  # type: ignore

        # Verify that all outputs are lists
        if not all(is_list_or_contains_list(t) or is_any(t) for t in output_field_types):
            return "Collector output must connect to a collection input"

        # Verify that all outputs match the input type (are a base class or the same class)
        if not all(
            is_any(t)
            or is_union_subtype(input_root_type, get_args(t)[0])
            or issubclass(input_root_type, get_args(t)[0])
            for t in output_field_types
        ):
            return "Collector outputs must connect to a collection input with a matching type"

        return None

    def nx_graph(self) -> nx.DiGraph:
        """Returns a NetworkX DiGraph representing the layout of this graph"""
        # TODO: Cache this?
        g = nx.DiGraph()
        g.add_nodes_from(list(self.nodes.keys()))
        g.add_edges_from({(e.source.node_id, e.destination.node_id) for e in self.edges})
        return g

    def nx_graph_with_data(self) -> nx.DiGraph:
        """Returns a NetworkX DiGraph representing the data and layout of this graph"""
        g = nx.DiGraph()
        g.add_nodes_from(list(self.nodes.items()))
        g.add_edges_from({(e.source.node_id, e.destination.node_id) for e in self.edges})
        return g

    def nx_graph_flat(self, nx_graph: Optional[nx.DiGraph] = None) -> nx.DiGraph:
        """Returns a flattened NetworkX DiGraph, including all subgraphs (but not with iterations expanded)"""
        g = nx_graph or nx.DiGraph()

        # Add all nodes from this graph except graph/iteration nodes
        g.add_nodes_from([n.id for n in self.nodes.values() if not isinstance(n, IterateInvocation)])

        # TODO: figure out if iteration nodes need to be expanded

        unique_edges = {(e.source.node_id, e.destination.node_id) for e in self.edges}
        g.add_edges_from([(e[0], e[1]) for e in unique_edges])
        return g


class GraphExecutionState(BaseModel):
    """Tracks the state of a graph execution"""

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

    @field_validator("graph")
    def graph_is_valid(cls, v: Graph):
        """Validates that the graph is valid"""
        v.validate_self()
        return v

    def next(self) -> Optional[BaseInvocation]:
        """Gets the next node ready to execute."""

        # TODO: enable multiple nodes to execute simultaneously by tracking currently executing nodes
        #       possibly with a timeout?

        # If there are no prepared nodes, prepare some nodes
        next_node = self._get_next_node()
        if next_node is None:
            prepared_id = self._prepare()

            # Prepare as many nodes as we can
            while prepared_id is not None:
                prepared_id = self._prepare()
                next_node = self._get_next_node()

        # Get values from edges
        if next_node is not None:
            try:
                self._prepare_inputs(next_node)
            except ValidationError as e:
                raise NodeInputError(next_node, e)

        # If next is still none, there's no next node, return None
        return next_node

    def complete(self, node_id: str, output: BaseInvocationOutput) -> None:
        """Marks a node as complete"""

        if node_id not in self.execution_graph.nodes:
            return  # TODO: log error?

        # Mark node as executed
        self.executed.add(node_id)
        self.results[node_id] = output

        # Check if source node is complete (all prepared nodes are complete)
        source_node = self.prepared_source_mapping[node_id]
        prepared_nodes = self.source_prepared_mapping[source_node]

        if all(n in self.executed for n in prepared_nodes):
            self.executed.add(source_node)
            self.executed_history.append(source_node)

    def set_node_error(self, node_id: str, error: str):
        """Marks a node as errored"""
        self.errors[node_id] = error

    def is_complete(self) -> bool:
        """Returns true if the graph is complete"""
        node_ids = set(self.graph.nx_graph_flat().nodes)
        return self.has_error() or all((k in self.executed for k in node_ids))

    def has_error(self) -> bool:
        """Returns true if the graph has any errors"""
        return len(self.errors) > 0

    def _create_execution_node(self, node_id: str, iteration_node_map: list[tuple[str, str]]) -> list[str]:
        """Prepares an iteration node and connects all edges, returning the new node id"""

        node = self.graph.get_node(node_id)

        self_iteration_count = -1

        # If this is an iterator node, we must create a copy for each iteration
        if isinstance(node, IterateInvocation):
            # Get input collection edge (should error if there are no inputs)
            input_collection_edge = next(iter(self.graph._get_input_edges(node_id, "collection")))
            input_collection_prepared_node_id = next(
                n[1] for n in iteration_node_map if n[0] == input_collection_edge.source.node_id
            )
            input_collection_prepared_node_output = self.results[input_collection_prepared_node_id]
            input_collection = getattr(input_collection_prepared_node_output, input_collection_edge.source.field)
            self_iteration_count = len(input_collection)

        new_nodes: list[str] = []
        if self_iteration_count == 0:
            # TODO: should this raise a warning? It might just happen if an empty collection is input, and should be valid.
            return new_nodes

        # Get all input edges
        input_edges = self.graph._get_input_edges(node_id)

        # Create new edges for this iteration
        # For collect nodes, this may contain multiple inputs to the same field
        new_edges: list[Edge] = []
        for edge in input_edges:
            for input_node_id in (n[1] for n in iteration_node_map if n[0] == edge.source.node_id):
                new_edge = Edge(
                    source=EdgeConnection(node_id=input_node_id, field=edge.source.field),
                    destination=EdgeConnection(node_id="", field=edge.destination.field),
                )
                new_edges.append(new_edge)

        # Create a new node (or one for each iteration of this iterator)
        for i in range(self_iteration_count) if self_iteration_count > 0 else [-1]:
            # Create a new node
            new_node = copy.deepcopy(node)

            # Create the node id (use a random uuid)
            new_node.id = uuid_string()

            # Set the iteration index for iteration invocations
            if isinstance(new_node, IterateInvocation):
                new_node.index = i

            # Add to execution graph
            self.execution_graph.add_node(new_node)
            self.prepared_source_mapping[new_node.id] = node_id
            if node_id not in self.source_prepared_mapping:
                self.source_prepared_mapping[node_id] = set()
            self.source_prepared_mapping[node_id].add(new_node.id)

            # Add new edges to execution graph
            for edge in new_edges:
                new_edge = Edge(
                    source=edge.source,
                    destination=EdgeConnection(node_id=new_node.id, field=edge.destination.field),
                )
                self.execution_graph.add_edge(new_edge)

            new_nodes.append(new_node.id)

        return new_nodes

    def _iterator_graph(self) -> nx.DiGraph:
        """Gets a DiGraph with edges to collectors removed so an ancestor search produces all active iterators for any node"""
        g = self.graph.nx_graph_flat()
        collectors = (n for n in self.graph.nodes if isinstance(self.graph.get_node(n), CollectInvocation))
        for c in collectors:
            g.remove_edges_from(list(g.in_edges(c)))
        return g

    def _get_node_iterators(self, node_id: str) -> list[str]:
        """Gets iterators for a node"""
        g = self._iterator_graph()
        iterators = [n for n in nx.ancestors(g, node_id) if isinstance(self.graph.get_node(n), IterateInvocation)]
        return iterators

    def _prepare(self) -> Optional[str]:
        # Get flattened source graph
        g = self.graph.nx_graph_flat()

        # Find next node that:
        # - was not already prepared
        # - is not an iterate node whose inputs have not been executed
        # - does not have an unexecuted iterate ancestor
        sorted_nodes = nx.topological_sort(g)
        next_node_id = next(
            (
                n
                for n in sorted_nodes
                # exclude nodes that have already been prepared
                if n not in self.source_prepared_mapping
                # exclude iterate nodes whose inputs have not been executed
                and not (
                    isinstance(self.graph.get_node(n), IterateInvocation)  # `n` is an iterate node...
                    and not all((e[0] in self.executed for e in g.in_edges(n)))  # ...that has unexecuted inputs
                )
                # exclude nodes who have unexecuted iterate ancestors
                and not any(
                    (
                        isinstance(self.graph.get_node(a), IterateInvocation)  # `a` is an iterate ancestor of `n`...
                        and a not in self.executed  # ...that is not executed
                        for a in nx.ancestors(g, n)  # for all ancestors `a` of node `n`
                    )
                )
            ),
            None,
        )

        if next_node_id is None:
            return None

        # Get all parents of the next node
        next_node_parents = [e[0] for e in g.in_edges(next_node_id)]

        # Create execution nodes
        next_node = self.graph.get_node(next_node_id)
        new_node_ids = []
        if isinstance(next_node, CollectInvocation):
            # Collapse all iterator input mappings and create a single execution node for the collect invocation
            all_iteration_mappings = list(
                itertools.chain(*(((s, p) for p in self.source_prepared_mapping[s]) for s in next_node_parents))
            )
            # all_iteration_mappings = list(set(itertools.chain(*prepared_parent_mappings)))
            create_results = self._create_execution_node(next_node_id, all_iteration_mappings)
            if create_results is not None:
                new_node_ids.extend(create_results)
        else:  # Iterators or normal nodes
            # Get all iterator combinations for this node
            # Will produce a list of lists of prepared iterator nodes, from which results can be iterated
            iterator_nodes = self._get_node_iterators(next_node_id)
            iterator_nodes_prepared = [list(self.source_prepared_mapping[n]) for n in iterator_nodes]
            iterator_node_prepared_combinations = list(itertools.product(*iterator_nodes_prepared))

            # Select the correct prepared parents for each iteration
            # For every iterator, the parent must either not be a child of that iterator, or must match the prepared iteration for that iterator
            # TODO: Handle a node mapping to none
            eg = self.execution_graph.nx_graph_flat()
            prepared_parent_mappings = [
                [(n, self._get_iteration_node(n, g, eg, it)) for n in next_node_parents]
                for it in iterator_node_prepared_combinations
            ]  # type: ignore

            # Create execution node for each iteration
            for iteration_mappings in prepared_parent_mappings:
                create_results = self._create_execution_node(next_node_id, iteration_mappings)  # type: ignore
                if create_results is not None:
                    new_node_ids.extend(create_results)

        return next(iter(new_node_ids), None)

    def _get_iteration_node(
        self,
        source_node_id: str,
        graph: nx.DiGraph,
        execution_graph: nx.DiGraph,
        prepared_iterator_nodes: list[str],
    ) -> Optional[str]:
        """Gets the prepared version of the specified source node that matches every iteration specified"""
        prepared_nodes = self.source_prepared_mapping[source_node_id]
        if len(prepared_nodes) == 1:
            return next(iter(prepared_nodes))

        # Check if the requested node is an iterator
        prepared_iterator = next((n for n in prepared_nodes if n in prepared_iterator_nodes), None)
        if prepared_iterator is not None:
            return prepared_iterator

        # Filter to only iterator nodes that are a parent of the specified node, in tuple format (prepared, source)
        iterator_source_node_mapping = [(n, self.prepared_source_mapping[n]) for n in prepared_iterator_nodes]
        parent_iterators = [itn for itn in iterator_source_node_mapping if nx.has_path(graph, itn[1], source_node_id)]

        return next(
            (n for n in prepared_nodes if all(nx.has_path(execution_graph, pit[0], n) for pit in parent_iterators)),
            None,
        )

    def _get_next_node(self) -> Optional[BaseInvocation]:
        """Gets the deepest node that is ready to be executed"""
        g = self.execution_graph.nx_graph()

        # Perform a topological sort using depth-first search
        topo_order = list(nx.dfs_postorder_nodes(g))

        # Get all IterateInvocation nodes
        iterate_nodes = [n for n in topo_order if isinstance(self.execution_graph.nodes[n], IterateInvocation)]

        # Sort the IterateInvocation nodes based on their index attribute
        iterate_nodes.sort(key=lambda x: self.execution_graph.nodes[x].index)

        # Prioritize IterateInvocation nodes and their children
        for iterate_node in iterate_nodes:
            if iterate_node not in self.executed and all((e[0] in self.executed for e in g.in_edges(iterate_node))):
                return self.execution_graph.nodes[iterate_node]

            # Check the children of the IterateInvocation node
            for child_node in nx.dfs_postorder_nodes(g, iterate_node):
                if child_node not in self.executed and all((e[0] in self.executed for e in g.in_edges(child_node))):
                    return self.execution_graph.nodes[child_node]

        # If no IterateInvocation node or its children are ready, return the first ready node in the topological order
        for node in topo_order:
            if node not in self.executed and all((e[0] in self.executed for e in g.in_edges(node))):
                return self.execution_graph.nodes[node]

        # If no node is found, return None
        return None

    def _prepare_inputs(self, node: BaseInvocation):
        input_edges = [e for e in self.execution_graph.edges if e.destination.node_id == node.id]
        # Inputs must be deep-copied, else if a node mutates the object, other nodes that get the same input
        # will see the mutation.
        if isinstance(node, CollectInvocation):
            output_collection = [
                copydeep(getattr(self.results[edge.source.node_id], edge.source.field))
                for edge in input_edges
                if edge.destination.field == "item"
            ]
            node.collection = output_collection
        else:
            for edge in input_edges:
                setattr(
                    node,
                    edge.destination.field,
                    copydeep(getattr(self.results[edge.source.node_id], edge.source.field)),
                )

    # TODO: Add API for modifying underlying graph that checks if the change will be valid given the current execution state
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

    def update_node(self, node_id: str, new_node: BaseInvocation) -> None:
        if not self._is_node_updatable(node_id):
            raise NodeAlreadyExecutedError(
                f"Node {node_id} has already been prepared or executed and cannot be updated"
            )
        self.graph.update_node(node_id, new_node)

    def delete_node(self, node_id: str) -> None:
        if not self._is_node_updatable(node_id):
            raise NodeAlreadyExecutedError(
                f"Node {node_id} has already been prepared or executed and cannot be deleted"
            )
        self.graph.delete_node(node_id)

    def add_edge(self, edge: Edge) -> None:
        if not self._is_node_updatable(edge.destination.node_id):
            raise NodeAlreadyExecutedError(
                f"Destination node {edge.destination.node_id} has already been prepared or executed and cannot be linked to"
            )
        self.graph.add_edge(edge)

    def delete_edge(self, edge: Edge) -> None:
        if not self._is_node_updatable(edge.destination.node_id):
            raise NodeAlreadyExecutedError(
                f"Destination node {edge.destination.node_id} has already been prepared or executed and cannot have a source edge deleted"
            )
        self.graph.delete_edge(edge)
