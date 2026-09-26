# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import copy
import itertools
import sys
import weakref
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Iterable,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler, PrivateAttr, ValidationError
from pydantic.fields import Field
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from invokeai.app.invocations import load_all_modules
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationRegistry,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.call_saved_workflow import (
    CallSavedWorkflowInvocation,
    is_call_saved_workflow_dynamic_input,
)
from invokeai.app.invocations.fields import Input, InputField, OutputField, OutputScope, UIType
from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import (
    LOOP_LINKAGE_FIELD,
    ForInvocation,
    ForReturnInvocation,
)
from invokeai.app.services.shared.graph_models import Edge, EdgeConnection
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import uuid_string

load_all_modules()

if TYPE_CHECKING:
    import networkx as nx
else:
    _FACADE_MODULE_NAME = "invokeai.app.services.shared.graph"

    class _LazyNetworkX:
        _module: Any | None = None

        def _load(self) -> Any:
            if self._module is None:
                import networkx

                self._module = networkx
            return self._module

        def __getattr__(self, name: str) -> Any:
            facade = sys.modules.get(_FACADE_MODULE_NAME)
            if facade is not None:
                facade_nx = vars(facade).get("nx")
                if facade_nx is not None and facade_nx is not self:
                    return getattr(facade_nx, name)
            return getattr(self._load(), name)

    nx = _LazyNetworkX()


NoneType = type(None)

_ACTIVE_FACADE_OVERRIDES: ContextVar[frozenset[str]] = ContextVar("active_facade_overrides", default=frozenset())

# Port name constants
ITEM_FIELD = "item"
COLLECTION_FIELD = "collection"
_RESERVED_EFFECT_PORTS = frozenset({"type", "output_meta", LOOP_LINKAGE_FIELD})


def _get_facade_override(name: str, implementation: Callable[..., Any]) -> Callable[..., Any]:
    """Preserve graph.py monkeypatch seams after moving implementations here."""

    facade = sys.modules.get("invokeai.app.services.shared.graph")
    if facade is None:
        return implementation
    override = vars(facade).get(name)
    if override is None or override is implementation:
        return implementation

    # A compatibility wrapper commonly saves the facade function and calls it from the override. In that case the
    # saved function must execute its canonical body instead of rediscovering the wrapper. The extra frame check keeps
    # the wrapper's side effects single-shot; the context guard covers callable overrides without a Python code object.
    caller = sys._getframe(1)
    override_code = getattr(override, "__code__", None)
    if caller.f_back is not None and override_code is not None and caller.f_back.f_code is override_code:
        return implementation
    if name in _ACTIVE_FACADE_OVERRIDES.get():
        return implementation

    @wraps(override)
    def dispatch(*args: Any, **kwargs: Any) -> Any:
        token = _ACTIVE_FACADE_OVERRIDES.set(_ACTIVE_FACADE_OVERRIDES.get() | {name})
        try:
            return override(*args, **kwargs)
        finally:
            _ACTIVE_FACADE_OVERRIDES.reset(token)

    return dispatch


@dataclass(frozen=True)
class _SupportedNestedForBody:
    body_path_nodes: frozenset[str]
    outer_return_id: str
    inner_for_ids: tuple[str, ...]
    continuation_nodes: frozenset[str]


@dataclass(frozen=True)
class _SupportedNestedIterateBody:
    body_path_nodes: set[str]
    return_node_id: str
    iterate_node_id: str
    collect_node_id: str


@dataclass(frozen=True)
class _SupportedNestedIterateChain:
    body_path_nodes: frozenset[str]
    return_node_id: str
    iterate_node_ids: tuple[str, str]
    collect_node_id: str


@dataclass(frozen=True)
class _SupportedNestedIterateSequence:
    body_path_nodes: frozenset[str]
    source_node_id: str
    iterate_node_ids: tuple[str, ...]
    preparation_node_ids: tuple[str, ...]
    body_node_id: str


def get_output_field_type(node: BaseInvocation, field: str) -> Any:
    implementation = _get_facade_override("get_output_field_type", get_output_field_type)
    if implementation is not get_output_field_type:
        return implementation(node, field)

    # TODO(psyche): This is awkward - if field_info is None, it means the field is not defined in the output, which
    # really should raise. The consumers of this utility expect it to never raise, and return None instead. Fixing this
    # would require some fairly significant changes and I don't want risk breaking anything.
    try:
        invocation_class = type(node)
        invocation_output_class = invocation_class.get_output_annotation()
        field_info = invocation_output_class.model_fields.get(field)
        assert field_info is not None, f"Output field '{field}' not found in {invocation_output_class.get_type()}"
        output_field_type = field_info.annotation
        return output_field_type
    except Exception:
        return None


def get_output_field_scope(node: BaseInvocation, field: str) -> OutputScope | None:
    implementation = _get_facade_override("get_output_field_scope", get_output_field_scope)
    if implementation is not get_output_field_scope:
        return implementation(node, field)

    try:
        invocation_class = type(node)
        invocation_output_class = invocation_class.get_output_annotation()
        field_info = invocation_output_class.model_fields.get(field)
        assert field_info is not None, f"Output field '{field}' not found in {invocation_output_class.get_type()}"
        json_schema_extra = field_info.json_schema_extra
        if not isinstance(json_schema_extra, dict):
            return None
        output_scope = json_schema_extra.get("output_scope")
        if output_scope is None:
            return None
        return OutputScope(output_scope)
    except Exception:
        return None


def get_input_field_type(node: BaseInvocation, field: str) -> Any:
    implementation = _get_facade_override("get_input_field_type", get_input_field_type)
    if implementation is not get_input_field_type:
        return implementation(node, field)

    # TODO(psyche): This is awkward - if field_info is None, it means the field is not defined in the output, which
    # really should raise. The consumers of this utility expect it to never raise, and return None instead. Fixing this
    # would require some fairly significant changes and I don't want risk breaking anything.
    try:
        invocation_class = type(node)
        field_info = invocation_class.model_fields.get(field)
        assert field_info is not None, f"Input field '{field}' not found in {invocation_class.get_type()}"
        input_field_type = field_info.annotation
        return input_field_type
    except Exception:
        return None


def is_union_subtype(t1, t2):
    implementation = _get_facade_override("is_union_subtype", is_union_subtype)
    if implementation is not is_union_subtype:
        return implementation(t1, t2)

    t1_args = get_args(t1)
    t2_args = get_args(t2)
    if not t1_args:
        # t1 is a single type
        return t1 in t2_args
    else:
        # t1 is a Union, check that all of its types are in t2_args
        return all(arg in t2_args for arg in t1_args)


def is_list_or_contains_list(t):
    implementation = _get_facade_override("is_list_or_contains_list", is_list_or_contains_list)
    if implementation is not is_list_or_contains_list:
        return implementation(t)

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
    implementation = _get_facade_override("is_any", is_any)
    if implementation is not is_any:
        return implementation(t)

    return t == Any or Any in get_args(t)


def extract_collection_item_types(t: Any) -> set[Any]:
    implementation = _get_facade_override("extract_collection_item_types", extract_collection_item_types)
    if implementation is not extract_collection_item_types:
        return implementation(t)

    """Extracts list item types from a collection annotation, including unions containing list branches."""
    if is_any(t):
        return {Any}

    if get_origin(t) is list:
        return {arg for arg in get_args(t) if arg != NoneType}

    item_types: set[Any] = set()
    for arg in get_args(t):
        if is_any(arg):
            item_types.add(Any)
        elif get_origin(arg) is list:
            item_types.update(item_arg for item_arg in get_args(arg) if item_arg != NoneType)
    return item_types


def are_connection_types_compatible(from_type: Any, to_type: Any) -> bool:
    implementation = _get_facade_override("are_connection_types_compatible", are_connection_types_compatible)
    if implementation is not are_connection_types_compatible:
        return implementation(from_type, to_type)

    if not from_type or not to_type:
        return False

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

    # Prefer issubclass when both are real classes
    try:
        if isinstance(from_type, type) and isinstance(to_type, type):
            return issubclass(from_type, to_type)
    except TypeError:
        pass

    # Union-to-Union (or Union-to-non-Union) handling
    return is_union_subtype(from_type, to_type)


def are_connections_compatible(
    from_node: BaseInvocation, from_field: str, to_node: BaseInvocation, to_field: str
) -> bool:
    """Determines if a connection between fields of two nodes is compatible."""

    implementation = _get_facade_override("are_connections_compatible", are_connections_compatible)
    if implementation is not are_connections_compatible:
        return implementation(from_node, from_field, to_node, to_field)

    # TODO: handle iterators and collectors
    from_type = get_output_field_type(from_node, from_field)
    to_type = get_input_field_type(to_node, to_field)

    return are_connection_types_compatible(from_type, to_type)


T = TypeVar("T")


def copydeep(obj: T) -> T:
    """Deep-copies an object. If it is a pydantic model, use the model's copy method."""
    implementation = _get_facade_override("copydeep", copydeep)
    if implementation is not copydeep:
        return implementation(obj)

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
    implementation = _get_facade_override("loc_to_dot_sep", loc_to_dot_sep)
    if implementation is not loc_to_dot_sep:
        return implementation(loc)

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
@invocation("iterate", version="1.1.0", use_cache=False)
class IterateInvocation(BaseInvocation):
    """Iterates over a list of items"""

    execution_effects_enabled = True

    collection: list[Any] = InputField(
        description="The list of items to iterate over", default=[], ui_type=UIType._Collection
    )
    index: int = InputField(description="The index, will be provided on executed iterators", default=0, ui_hidden=True)

    def invoke(self, context: InvocationContext) -> IterateInvocationOutput:
        """Produces the outputs as values"""
        item = self.collection[self.index]
        execution = getattr(context, "execution", None)
        if execution is not None:
            execution.emit("item", item, sequence=self.index)
            if self.index + 1 >= len(self.collection):
                execution.close_stream("item")
        return IterateInvocationOutput(item=item, index=self.index, total=len(self.collection))

    def get_event_invocation(self) -> "IterateInvocation":
        event_invocation = self.model_copy()
        event_invocation.collection = []
        return event_invocation


@invocation_output("collect_output")
class CollectInvocationOutput(BaseInvocationOutput):
    collection: list[Any] = OutputField(
        description="The collection of input items", title="Collection", ui_type=UIType._Collection
    )


@invocation("collect", version="1.1.0", use_cache=False)
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
        description="An optional collection to append to",
        default=[],
        ui_type=UIType._Collection,
        input=Input.Connection,
    )

    def invoke(self, context: InvocationContext) -> CollectInvocationOutput:
        """Invoke with provided services and return outputs."""
        return CollectInvocationOutput(collection=copy.copy(self.collection))

    def get_event_invocation(self) -> "CollectInvocation":
        event_invocation = self.model_copy()
        event_invocation.collection = []
        return event_invocation


class AnyInvocation(BaseInvocation):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        def validate_invocation(v: Any) -> "AnyInvocation":
            return InvocationRegistry.get_invocation_typeadapter().validate_python(v)

        return core_schema.no_info_plain_validator_function(validate_invocation)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Nodes are too powerful, we have to make our own OpenAPI schema manually
        # No but really, because the schema is dynamic depending on loaded nodes, we need to generate it manually
        oneOf: list[dict[str, str]] = []
        names = [i.__name__ for i in InvocationRegistry.get_invocation_classes()]
        for name in sorted(names):
            oneOf.append({"$ref": f"#/components/schemas/{name}"})
        return {"oneOf": oneOf}


class AnyInvocationOutput(BaseInvocationOutput):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler):
        def validate_invocation_output(v: Any) -> "AnyInvocationOutput":
            return InvocationRegistry.get_output_typeadapter().validate_python(v)

        return core_schema.no_info_plain_validator_function(validate_invocation_output)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Nodes are too powerful, we have to make our own OpenAPI schema manually
        # No but really, because the schema is dynamic depending on loaded nodes, we need to generate it manually

        oneOf: list[dict[str, str]] = []
        names = [i.__name__ for i in InvocationRegistry.get_output_classes()]
        for name in sorted(names):
            oneOf.append({"$ref": f"#/components/schemas/{name}"})
        return {"oneOf": oneOf}


_EdgeListMutationParams = ParamSpec("_EdgeListMutationParams")
_EdgeListMutationResult = TypeVar("_EdgeListMutationResult")


def _invalidates_edge_indexes(
    method: Callable[Concatenate["_EdgeList", _EdgeListMutationParams], _EdgeListMutationResult],
) -> Callable[Concatenate["_EdgeList", _EdgeListMutationParams], _EdgeListMutationResult]:
    @wraps(method)
    def wrapped(
        self: "_EdgeList", *args: _EdgeListMutationParams.args, **kwargs: _EdgeListMutationParams.kwargs
    ) -> _EdgeListMutationResult:
        try:
            return method(self, *args, **kwargs)
        finally:
            self._invalidate_indexes()

    return wrapped


class _EdgeList(list[Edge]):
    """A graph-owned edge list that invalidates adjacency indexes after direct mutation."""

    def __init__(self, edges: Iterable[Edge], owner: "Graph") -> None:
        super().__init__(edges)
        self._owner_ref = weakref.ref(owner)

    def _invalidate_indexes(self) -> None:
        owner_ref = getattr(self, "_owner_ref", None)
        owner = owner_ref() if owner_ref is not None else None
        if owner is not None:
            owner._invalidate_edge_indexes()

    def __getstate__(self) -> dict[str, Any]:
        return {}

    @_invalidates_edge_indexes
    def append(self, edge: Edge) -> None:
        super().append(edge)

    @_invalidates_edge_indexes
    def extend(self, edges: Iterable[Edge]) -> None:
        super().extend(edges)

    @_invalidates_edge_indexes
    def insert(self, index: int, edge: Edge) -> None:
        super().insert(index, edge)

    @_invalidates_edge_indexes
    def __setitem__(self, index: Any, value: Any) -> None:
        super().__setitem__(index, value)

    @_invalidates_edge_indexes
    def __delitem__(self, index: Any) -> None:
        super().__delitem__(index)

    @_invalidates_edge_indexes
    def __iadd__(self, edges: Iterable[Edge]):
        return super().__iadd__(edges)

    @_invalidates_edge_indexes
    def __imul__(self, value: int):
        return super().__imul__(value)

    @_invalidates_edge_indexes
    def clear(self) -> None:
        super().clear()

    @_invalidates_edge_indexes
    def pop(self, index: int = -1) -> Edge:
        return super().pop(index)

    @_invalidates_edge_indexes
    def remove(self, edge: Edge) -> None:
        super().remove(edge)

    @_invalidates_edge_indexes
    def reverse(self) -> None:
        super().reverse()

    @_invalidates_edge_indexes
    def sort(self, *, key=None, reverse: bool = False) -> None:
        super().sort(key=key, reverse=reverse)


class _GraphEdgeIndexes:
    """Process-local adjacency indexes kept out of pydantic's attribute lookup path."""

    __slots__ = ("input_edges_by_node", "output_edges_by_node")

    def __init__(self) -> None:
        self.input_edges_by_node: Optional[dict[str, list[Edge]]] = None
        self.output_edges_by_node: Optional[dict[str, list[Edge]]] = None


class Graph(BaseModel):
    """A validated invocation graph made of nodes and typed edges."""

    id: str = Field(description="The id of this graph", default_factory=uuid_string)
    # TODO: use a list (and never use dict in a BaseModel) because pydantic/fastapi hates me
    nodes: dict[str, AnyInvocation] = Field(description="The nodes in this graph", default_factory=dict)
    edges: list[Edge] = Field(
        description="The connections between nodes and their fields in this graph",
        default_factory=list,
    )
    _edge_indexes: _GraphEdgeIndexes = PrivateAttr(default_factory=_GraphEdgeIndexes)

    def _rebind_edge_list(self) -> None:
        object.__setattr__(self, "edges", _EdgeList(self.edges, self))
        self._invalidate_edge_indexes()

    def model_post_init(self, __context: Any) -> None:
        self._rebind_edge_list()

    def model_copy(self, *, update: Optional[dict[str, Any]] = None, deep: bool = False) -> "Graph":
        copied = super().model_copy(update=update, deep=deep)
        copied.__pydantic_private__["_edge_indexes"] = _GraphEdgeIndexes()
        copied._rebind_edge_list()
        return copied

    def __copy__(self) -> "Graph":
        copied = super().__copy__()
        copied.__pydantic_private__["_edge_indexes"] = _GraphEdgeIndexes()
        copied._rebind_edge_list()
        return copied

    def __deepcopy__(self, memo: Optional[dict[int, Any]] = None) -> "Graph":
        copied = super().__deepcopy__(memo)
        copied.__pydantic_private__["_edge_indexes"] = _GraphEdgeIndexes()
        copied._rebind_edge_list()
        return copied

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._rebind_edge_list()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "edges":
            value = _EdgeList(value, self)
            super().__setattr__(name, value)
            self._invalidate_edge_indexes()
            return
        super().__setattr__(name, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Graph):
            return NotImplemented
        return self.id == other.id and self.nodes == other.nodes and self.edges == other.edges

    def _invalidate_edge_indexes(self) -> None:
        indexes = self.__pydantic_private__["_edge_indexes"]
        indexes.input_edges_by_node = None
        indexes.output_edges_by_node = None

    def _ensure_edge_indexes(self) -> None:
        indexes = self.__pydantic_private__["_edge_indexes"]
        if indexes.input_edges_by_node is not None and indexes.output_edges_by_node is not None:
            return

        input_edges_by_node: dict[str, list[Edge]] = {}
        output_edges_by_node: dict[str, list[Edge]] = {}
        for edge in self.edges:
            input_edges_by_node.setdefault(edge.destination.node_id, []).append(edge)
            output_edges_by_node.setdefault(edge.source.node_id, []).append(edge)
        indexes.input_edges_by_node = input_edges_by_node
        indexes.output_edges_by_node = output_edges_by_node

    def _add_edge_to_indexes(self, edge: Edge) -> None:
        indexes = self.__pydantic_private__["_edge_indexes"]
        if indexes.input_edges_by_node is not None:
            indexes.input_edges_by_node.setdefault(edge.destination.node_id, []).append(edge)
        if indexes.output_edges_by_node is not None:
            indexes.output_edges_by_node.setdefault(edge.source.node_id, []).append(edge)

    def _remove_edge_from_indexes(self, edge: Edge) -> None:
        indexes = self.__pydantic_private__["_edge_indexes"]
        if indexes.input_edges_by_node is not None:
            input_edges = indexes.input_edges_by_node.get(edge.destination.node_id)
            if input_edges is not None:
                input_edges.remove(edge)
        if indexes.output_edges_by_node is not None:
            output_edges = indexes.output_edges_by_node.get(edge.source.node_id)
            if output_edges is not None:
                output_edges.remove(edge)

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
            input_edges = self._get_input_edges(node_id, include_loop_linkage=True)
            output_edges = self._get_output_edges(node_id, include_loop_linkage=True)

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

        self._add_edge(edge, allow_inputless_source_collector=False)

    def _add_edge(self, edge: Edge, allow_inputless_source_collector: bool) -> None:
        destination_node = self._validate_edge(edge, allow_inputless_source_collector)
        if edge not in self.edges:
            list.append(self.edges, edge)
            self._add_edge_to_indexes(edge)
            try:
                self._validate_if_branch_input_dependents(edge, destination_node)
            except Exception:
                self.delete_edge(edge)
                raise
        else:
            raise InvalidEdgeError()

    def _validate_if_branch_input_dependents(self, edge: Edge, node: BaseInvocation) -> None:
        """Revalidate type-dependent paths through Ifs and collectors after adding an If branch."""
        if not isinstance(node, IfInvocation) or edge.destination.field not in ("true_input", "false_input"):
            return

        pending_nodes: list[IfInvocation | CollectInvocation | IterateInvocation] = [node]
        validated_node_ids: set[str] = set()
        while pending_nodes:
            source_node = pending_nodes.pop()
            if source_node.id in validated_node_ids:
                continue
            validated_node_ids.add(source_node.id)

            if isinstance(source_node, IterateInvocation):
                err = self._is_iterator_connection_valid(source_node.id)
                if err is not None:
                    raise InvalidEdgeError(
                        f"Iterator input type does not match iterator output type ({source_node.id}): {err}"
                    )
                continue

            if isinstance(source_node, CollectInvocation):
                err = self._is_collector_connection_valid(source_node.id)
                if err is not None:
                    raise InvalidEdgeError(
                        f"Collector output type does not match collector input type ({source_node.id}): {err}"
                    )
                output_field = COLLECTION_FIELD
            else:
                if self._get_effective_output_connections(source_node.id, "value") is None:
                    continue
                output_field = "value"

            for output_edge in self._get_output_edges(source_node.id, output_field):
                destination_node = self.get_node(output_edge.destination.node_id)
                self._validate_edge_field_compatibility(output_edge, source_node, destination_node)

                if isinstance(destination_node, IfInvocation) and output_edge.destination.field in (
                    "true_input",
                    "false_input",
                ):
                    pending_nodes.append(destination_node)
                elif isinstance(destination_node, CollectInvocation) and output_edge.destination.field in (
                    ITEM_FIELD,
                    COLLECTION_FIELD,
                ):
                    pending_nodes.append(destination_node)
                elif (
                    isinstance(destination_node, IterateInvocation)
                    and output_edge.destination.field == COLLECTION_FIELD
                ):
                    pending_nodes.append(destination_node)

    def _extend_edges_unchecked(self, edges: Iterable[Edge]) -> None:
        """Adds trusted runtime edges without author-time graph validation.

        This is only for execution edges derived from an already-validated source graph. Runtime materialization
        preserves the source graph's direction and field connections, so repeating cycle, uniqueness, and type checks
        for every expanded edge is redundant and prohibitively expensive for large iterator collections.
        """
        new_edges = list(edges)
        list.extend(self.edges, new_edges)
        for edge in new_edges:
            self._add_edge_to_indexes(edge)

    def delete_edge(self, edge: Edge) -> None:
        """Deletes an edge from a graph"""

        try:
            list.remove(self.edges, edge)
            self._remove_edge_from_indexes(edge)
        except ValueError:
            pass

    def _validate_unique_node_ids(self) -> None:
        node_ids = [n.id for n in self.nodes.values()]
        seen = set()
        duplicate_node_ids = {nid for nid in node_ids if (nid in seen) or seen.add(nid)}
        if duplicate_node_ids:
            raise DuplicateNodeIdError(f"Node ids must be unique, found duplicates {duplicate_node_ids}")

    def _validate_node_id_mapping(self) -> None:
        for node_dict_id, node in self.nodes.items():
            if node_dict_id != node.id:
                raise NodeIdMismatchError(f"Node ids must match, got {node_dict_id} and {node.id}")

    def _validate_edge_nodes_and_fields(self) -> None:
        for edge in self.edges:
            self._validate_reserved_edge_fields(edge)
            source_node = self.nodes.get(edge.source.node_id, None)
            if source_node is None:
                raise NodeNotFoundError(f"Edge source node {edge.source.node_id} does not exist in the graph")

            destination_node = self.nodes.get(edge.destination.node_id, None)
            if destination_node is None:
                raise NodeNotFoundError(f"Edge destination node {edge.destination.node_id} does not exist in the graph")

            if edge.source.field not in source_node.get_output_annotation().model_fields:
                raise NodeFieldNotFoundError(
                    f"Edge source field {edge.source.field} does not exist in node {edge.source.node_id}"
                )

            if edge.destination.field not in type(destination_node).model_fields:
                if isinstance(destination_node, CallSavedWorkflowInvocation) and is_call_saved_workflow_dynamic_input(
                    edge.destination.field
                ):
                    continue
                raise NodeFieldNotFoundError(
                    f"Edge destination field {edge.destination.field} does not exist in node {edge.destination.node_id}"
                )

    def _validate_graph_is_acyclic(self) -> None:
        graph = self.nx_graph_flat()
        if not nx.is_directed_acyclic_graph(graph):
            raise CyclicalGraphError("Graph contains cycles")

    def _validate_edge_type_compatibility(self) -> None:
        for edge in self.edges:
            destination_node = self.get_node(edge.destination.node_id)
            if isinstance(destination_node, CallSavedWorkflowInvocation) and is_call_saved_workflow_dynamic_input(
                edge.destination.field
            ):
                continue
            self._validate_edge_not_to_direct_input(edge, destination_node)
            source_node = self.get_node(edge.source.node_id)
            if not self._are_effective_connections_compatible(
                source_node, edge.source.field, destination_node, edge.destination.field
            ):
                raise InvalidEdgeError(f"Edge source and target types do not match ({edge})")

    def _validate_special_nodes(self) -> None:
        # TODO: may need to validate all iterators & collectors in subgraphs so edge connections in parent graphs will be available
        self._validate_for_loop_linkages()
        for node in self.nodes.values():
            if isinstance(node, IterateInvocation):
                err = self._is_iterator_connection_valid(node.id)
                if err is not None:
                    raise InvalidEdgeError(f"Invalid iterator node ({node.id}): {err}")
            if isinstance(node, CollectInvocation):
                err = self._is_collector_connection_valid(node.id)
                if err is not None:
                    raise InvalidEdgeError(f"Invalid collector node ({node.id}): {err}")
            if isinstance(node, ForInvocation):
                err = self._is_for_connection_valid(node.id)
                if err is not None:
                    raise InvalidEdgeError(f"Invalid For node ({node.id}): {err}")
            if isinstance(node, ForReturnInvocation):
                err = self._is_for_return_connection_valid(node.id)
                if err is not None:
                    raise InvalidEdgeError(f"Invalid ForReturn node ({node.id}): {err}")

    def _validate_for_loop_linkages(self) -> None:
        """Validates the required non-data association between each For and its ForReturn."""
        for_nodes = [node for node in self.nodes.values() if isinstance(node, ForInvocation)]
        return_nodes = [node for node in self.nodes.values() if isinstance(node, ForReturnInvocation)]
        linkage_edges = self._get_loop_linkage_edges()

        for edge in linkage_edges:
            source_node = self.nodes.get(edge.source.node_id)
            destination_node = self.nodes.get(edge.destination.node_id)
            if (
                not isinstance(source_node, ForInvocation)
                or not isinstance(destination_node, ForReturnInvocation)
                or edge.source.field != LOOP_LINKAGE_FIELD
                or edge.destination.field != LOOP_LINKAGE_FIELD
            ):
                raise InvalidEdgeError(f"Invalid loop linkage ({edge})")

        for node in for_nodes:
            matching_edges = [edge for edge in linkage_edges if edge.source.node_id == node.id]
            if len(matching_edges) != 1:
                raise InvalidEdgeError(f"For '{node.id}' must have exactly one loop linkage")
        for node in return_nodes:
            matching_edges = [edge for edge in linkage_edges if edge.destination.node_id == node.id]
            if len(matching_edges) != 1:
                raise InvalidEdgeError(f"ForReturn '{node.id}' must have exactly one loop linkage")

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

        self._validate_unique_node_ids()
        self._validate_node_id_mapping()
        self._validate_edge_nodes_and_fields()
        self._validate_graph_is_acyclic()
        self._validate_edge_type_compatibility()
        self._validate_special_nodes()
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

    def _get_edge_nodes(self, edge: Edge) -> tuple[BaseInvocation, BaseInvocation]:
        try:
            return self.get_node(edge.source.node_id), self.get_node(edge.destination.node_id)
        except NodeNotFoundError:
            raise InvalidEdgeError(f"One or both nodes don't exist ({edge})")

    def _validate_edge_destination_uniqueness(self, edge: Edge, destination_node: BaseInvocation) -> None:
        input_edges = self._get_input_edges(edge.destination.node_id, edge.destination.field)
        if len(input_edges) > 0 and (
            not isinstance(destination_node, CollectInvocation) or edge.destination.field != ITEM_FIELD
        ):
            raise InvalidEdgeError(f"Edge already exists ({edge})")

    def _validate_edge_would_not_create_cycle(self, edge: Edge) -> None:
        graph = self.nx_graph_flat()
        graph.add_edge(edge.source.node_id, edge.destination.node_id)
        if not nx.is_directed_acyclic_graph(graph):
            raise InvalidEdgeError(f"Edge creates a cycle in the graph ({edge})")

    def _validate_edge_field_compatibility(
        self, edge: Edge, source_node: BaseInvocation, destination_node: BaseInvocation
    ) -> None:
        if isinstance(destination_node, CallSavedWorkflowInvocation) and is_call_saved_workflow_dynamic_input(
            edge.destination.field
        ):
            return
        self._validate_edge_not_to_direct_input(edge, destination_node)
        if not self._are_effective_connections_compatible(
            source_node, edge.source.field, destination_node, edge.destination.field
        ):
            raise InvalidEdgeError(f"Field types are incompatible ({edge})")

    def _validate_edge_not_to_direct_input(self, edge: Edge, destination_node: BaseInvocation) -> None:
        destination_field = type(destination_node).model_fields.get(edge.destination.field)
        if destination_field is not None:
            json_schema_extra = destination_field.json_schema_extra
            if isinstance(json_schema_extra, dict) and json_schema_extra.get("input") == Input.Direct:
                raise InvalidEdgeError(f"Cannot connect to direct input ({edge})")

    def _validate_reserved_edge_fields(self, edge: Edge) -> None:
        if edge.type == "default" and (
            edge.source.field == LOOP_LINKAGE_FIELD or edge.destination.field == LOOP_LINKAGE_FIELD
        ):
            raise InvalidEdgeError(f"The loop_linkage field must use a loop_linkage edge ({edge})")

    def _validate_loop_linkage_edge(
        self, edge: Edge, source_node: BaseInvocation, destination_node: BaseInvocation
    ) -> None:
        if (
            not isinstance(source_node, ForInvocation)
            or not isinstance(destination_node, ForReturnInvocation)
            or edge.source.field != LOOP_LINKAGE_FIELD
            or edge.destination.field != LOOP_LINKAGE_FIELD
        ):
            raise InvalidEdgeError(f"Invalid loop linkage ({edge})")

        if any(existing_edge.source.node_id == source_node.id for existing_edge in self._get_loop_linkage_edges()):
            raise InvalidEdgeError(f"For node already has a loop linkage ({edge})")
        if any(
            existing_edge.destination.node_id == destination_node.id for existing_edge in self._get_loop_linkage_edges()
        ):
            raise InvalidEdgeError(f"ForReturn node already has a loop linkage ({edge})")

    def _validate_iterator_edge_rules(
        self, edge: Edge, source_node: BaseInvocation, destination_node: BaseInvocation
    ) -> None:
        if isinstance(destination_node, IterateInvocation) and edge.destination.field == COLLECTION_FIELD:
            err = self._is_iterator_connection_valid(edge.destination.node_id, new_input=edge.source)
            if err is not None:
                raise InvalidEdgeError(f"Iterator input type does not match iterator output type ({edge}): {err}")

        if isinstance(source_node, IterateInvocation) and edge.source.field == ITEM_FIELD:
            err = self._is_iterator_connection_valid(edge.source.node_id, new_output=edge.destination)
            if err is not None:
                raise InvalidEdgeError(f"Iterator output type does not match iterator input type ({edge}): {err}")

    def _validate_collector_edge_rules(
        self,
        edge: Edge,
        source_node: BaseInvocation,
        destination_node: BaseInvocation,
        allow_inputless_source_collector: bool,
    ) -> None:
        if isinstance(destination_node, CollectInvocation) and edge.destination.field in (ITEM_FIELD, COLLECTION_FIELD):
            err = self._is_collector_connection_valid(
                edge.destination.node_id, new_input=edge.source, new_input_field=edge.destination.field
            )
            if err is not None:
                raise InvalidEdgeError(f"Collector output type does not match collector input type ({edge}): {err}")

        if (
            isinstance(source_node, CollectInvocation)
            and edge.source.field == COLLECTION_FIELD
            and not self._is_destination_field_list_of_Any(edge)
            and not self._is_destination_field_Any(edge)
        ):
            if allow_inputless_source_collector and not any(
                edge.destination.node_id == source_node.id for edge in self.edges
            ):
                return
            err = self._is_collector_connection_valid(edge.source.node_id, new_output=edge.destination)
            if err is not None:
                raise InvalidEdgeError(f"Collector input type does not match collector output type ({edge}): {err}")

    def _validate_edge(self, edge: Edge, allow_inputless_source_collector: bool = False) -> BaseInvocation:
        """Validate a new edge and return its destination for post-insertion validation."""
        self._validate_reserved_edge_fields(edge)
        source_node, destination_node = self._get_edge_nodes(edge)
        if edge.type == "loop_linkage":
            self._validate_loop_linkage_edge(edge, source_node, destination_node)
            return destination_node
        self._validate_edge_destination_uniqueness(edge, destination_node)
        self._validate_edge_would_not_create_cycle(edge)
        self._validate_edge_field_compatibility(edge, source_node, destination_node)
        self._validate_iterator_edge_rules(edge, source_node, destination_node)
        self._validate_collector_edge_rules(edge, source_node, destination_node, allow_inputless_source_collector)
        return destination_node

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
            input_edges = self._get_input_edges(node_id, include_loop_linkage=True)
            output_edges = self._get_output_edges(node_id, include_loop_linkage=True)

            # Delete node and all edges
            self.delete_node(node_id)

            # Create new edges for each input and output
            for edge in input_edges:
                self.add_edge(
                    Edge(
                        type=edge.type,
                        source=edge.source,
                        destination=EdgeConnection(node_id=new_node.id, field=edge.destination.field),
                    )
                )

            for edge in output_edges:
                self.add_edge(
                    Edge(
                        type=edge.type,
                        source=EdgeConnection(node_id=new_node.id, field=edge.source.field),
                        destination=edge.destination,
                    )
                )

    def _get_input_edges(
        self, node_id: str, field: Optional[str] = None, *, include_loop_linkage: bool = False
    ) -> list[Edge]:
        """Gets all input edges for a node. If field is provided, only edges to that field are returned."""

        self._ensure_edge_indexes()
        indexes = self.__pydantic_private__["_edge_indexes"]
        assert indexes.input_edges_by_node is not None
        edges = indexes.input_edges_by_node.get(node_id, [])
        if not include_loop_linkage:
            edges = [edge for edge in edges if edge.type == "default"]

        if field is None:
            return list(edges)

        filtered_edges = [e for e in edges if e.destination.field == field]

        return filtered_edges

    def _get_output_edges(
        self, node_id: str, field: Optional[str] = None, *, include_loop_linkage: bool = False
    ) -> list[Edge]:
        """Gets all output edges for a node. If field is provided, only edges from that field are returned."""
        self._ensure_edge_indexes()
        indexes = self.__pydantic_private__["_edge_indexes"]
        assert indexes.output_edges_by_node is not None
        edges = indexes.output_edges_by_node.get(node_id, [])
        if not include_loop_linkage:
            edges = [edge for edge in edges if edge.type == "default"]

        if field is None:
            return list(edges)

        filtered_edges = [e for e in edges if e.source.field == field]

        return filtered_edges

    def _get_loop_linkage_edges(self, node_id: str | None = None) -> list[Edge]:
        edges = [edge for edge in self.edges if edge.type == "loop_linkage"]
        if node_id is None:
            return edges
        return [edge for edge in edges if edge.source.node_id == node_id or edge.destination.node_id == node_id]

    def _get_linked_for_return_id(self, for_node_id: str) -> str | None:
        linkage_edges = [
            edge for edge in self._get_loop_linkage_edges(for_node_id) if edge.source.node_id == for_node_id
        ]
        if len(linkage_edges) != 1:
            return None
        return linkage_edges[0].destination.node_id

    def _get_for_iteration_output_edges(self, node_id: str) -> list[Edge]:
        node = self.get_node(node_id)
        return [
            edge
            for edge in self._get_output_edges(node_id)
            if get_output_field_scope(node, edge.source.field) == OutputScope.Iteration
        ]

    def _get_for_final_output_edges(self, node_id: str) -> list[Edge]:
        node = self.get_node(node_id)
        return [
            edge
            for edge in self._get_output_edges(node_id)
            if get_output_field_scope(node, edge.source.field) == OutputScope.Final
        ]

    def _get_for_reachable_body_nodes(self, iteration_edges: list[Edge], graph: "nx.DiGraph") -> set[str]:
        body_nodes: set[str] = set()
        for edge in iteration_edges:
            body_nodes.add(edge.destination.node_id)
            body_nodes.update(nx.descendants(graph, edge.destination.node_id))
        return body_nodes

    def _get_for_body_path_nodes(
        self, reachable_body_nodes: set[str], return_node_id: str, graph: "nx.DiGraph"
    ) -> set[str]:
        return (reachable_body_nodes & nx.ancestors(graph, return_node_id)) | {return_node_id}

    def _get_for_body_path_to_return(self, node_id: str, graph: "nx.DiGraph") -> tuple[set[str], str] | None:
        """Resolve the runtime body path to its owning ForReturn.

        The loop linkage identifies the return endpoint. The ordinary body graph still determines whether that return
        is reachable from an iteration output and which nodes belong to the body.
        """
        iteration_edges = self._get_for_iteration_output_edges(node_id)
        if len(iteration_edges) == 0:
            return None

        reachable_body_nodes = self._get_for_reachable_body_nodes(iteration_edges, graph)
        return_node_id = self._get_linked_for_return_id(node_id)
        if return_node_id is None or return_node_id not in reachable_body_nodes:
            return None

        return self._get_for_body_path_nodes(reachable_body_nodes, return_node_id, graph), return_node_id

    def _get_supported_for_nested_iterate_body(
        self, node_id: str, graph: "nx.DiGraph"
    ) -> _SupportedNestedIterateBody | None:
        """Return the bounded internal Iterate body contract, if this For uses it.

        Supported shape::

            For.item -> Iterate.item -> body -> Collect.item
                                             Collect.collection -> ForReturn.output

        The Iterate and Collect nodes are scheduler-managed, so no other branch or final For output may escape this
        body contract.
        """
        body_path_to_return = self._get_for_body_path_to_return(node_id, graph)
        if body_path_to_return is None:
            return None

        body_path_nodes, return_node_id = body_path_to_return
        iterate_node_ids = [
            body_node_id
            for body_node_id in body_path_nodes
            if isinstance(self.get_node(body_node_id), IterateInvocation)
        ]
        collect_node_ids = [
            body_node_id
            for body_node_id in body_path_nodes
            if isinstance(self.get_node(body_node_id), CollectInvocation)
        ]
        if len(iterate_node_ids) != 1 or len(collect_node_ids) != 1:
            return None

        iterate_node_id = iterate_node_ids[0]
        collect_node_id = collect_node_ids[0]
        if not nx.has_path(graph, iterate_node_id, collect_node_id):
            return None
        iterate_input_edges = self._get_input_edges(iterate_node_id, COLLECTION_FIELD)
        if len(iterate_input_edges) != 1:
            return None
        iterate_input_source_id = iterate_input_edges[0].source.node_id
        if iterate_input_source_id != node_id and iterate_input_source_id not in body_path_nodes:
            return None

        return_output_edges = self._get_input_edges(return_node_id, "output")
        if len(return_output_edges) != 1 or (
            return_output_edges[0].source.node_id != collect_node_id
            or return_output_edges[0].source.field != COLLECTION_FIELD
        ):
            return None
        if any(
            edge.destination.field != "output"
            and edge.destination.field != "continue_condition"
            and (edge.destination.field != "state" or edge.source.node_id != node_id or edge.source.field != "state")
            for edge in self._get_input_edges(return_node_id)
        ):
            return None

        if self._get_input_edges(collect_node_id, COLLECTION_FIELD):
            return None
        collect_item_edges = self._get_input_edges(collect_node_id, ITEM_FIELD)
        if len(collect_item_edges) != 1:
            return None
        collect_item_source_id = collect_item_edges[0].source.node_id
        if not nx.has_path(graph, iterate_node_id, collect_item_source_id):
            return None

        for body_node_id in body_path_nodes:
            if body_node_id in {iterate_node_id, collect_node_id, return_node_id}:
                continue
            if not nx.has_path(graph, body_node_id, collect_node_id):
                return None
            if not (
                nx.has_path(graph, body_node_id, iterate_node_id) or nx.has_path(graph, iterate_node_id, body_node_id)
            ):
                return None

        return _SupportedNestedIterateBody(
            body_path_nodes=body_path_nodes,
            return_node_id=return_node_id,
            iterate_node_id=iterate_node_id,
            collect_node_id=collect_node_id,
        )

    def _get_supported_for_serial_nested_iterate_chain(
        self, node_id: str, graph: "nx.DiGraph"
    ) -> _SupportedNestedIterateChain | None:
        """Return the exact two-level serial Iterate contract, if present.

        Supported shape::

            For.item -> prep1 -> Iterate1.item -> prep2 -> Iterate2.item -> body -> Collect.item
                                                                                         Collect.collection -> ForReturn.output

        This intentionally excludes sibling iterators, fan-in, and additional control-flow nodes. Those shapes must
        remain on the compatibility path until separately proven equivalent.
        """
        body_path_to_return = self._get_for_body_path_to_return(node_id, graph)
        if body_path_to_return is None:
            return None
        body_path_nodes, return_node_id = body_path_to_return
        iterate_node_ids = tuple(
            body_node_id
            for body_node_id in body_path_nodes
            if isinstance(self.get_node(body_node_id), IterateInvocation)
        )
        collect_node_ids = tuple(
            body_node_id
            for body_node_id in body_path_nodes
            if isinstance(self.get_node(body_node_id), CollectInvocation)
        )
        if len(iterate_node_ids) != 2 or len(collect_node_ids) != 1:
            return None

        collect_node_id = collect_node_ids[0]
        if len(body_path_nodes) != 7:
            return None
        return_output_edges = self._get_input_edges(return_node_id, "output")
        if len(return_output_edges) != 1 or (
            return_output_edges[0].source.node_id != collect_node_id
            or return_output_edges[0].source.field != COLLECTION_FIELD
        ):
            return None
        if self._get_input_edges(collect_node_id, COLLECTION_FIELD):
            return None
        collect_item_edges = self._get_input_edges(collect_node_id, ITEM_FIELD)
        if len(collect_item_edges) != 1:
            return None

        first_iterate_id, second_iterate_id = sorted(
            iterate_node_ids, key=lambda iterate_id: nx.shortest_path_length(graph, node_id, iterate_id)
        )
        if not nx.has_path(graph, first_iterate_id, second_iterate_id) or not nx.has_path(
            graph, second_iterate_id, collect_node_id
        ):
            return None
        first_input_edges = self._get_input_edges(first_iterate_id, COLLECTION_FIELD)
        second_input_edges = self._get_input_edges(second_iterate_id, COLLECTION_FIELD)
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

        control_types = (
            CallSavedWorkflowInvocation,
            IfInvocation,
            ForInvocation,
            ForReturnInvocation,
            IterateInvocation,
            CollectInvocation,
        )
        ordinary_nodes = {first_preparation_id, second_preparation_id, body_node_id}
        if any(isinstance(self.get_node(candidate_id), control_types) for candidate_id in ordinary_nodes):
            return None
        if (
            first_input_edges[0].source.node_id != first_preparation_id
            or first_input_edges[0].source.field != COLLECTION_FIELD
            or second_input_edges[0].source.node_id != second_preparation_id
            or second_input_edges[0].source.field != COLLECTION_FIELD
        ):
            return None

        first_preparation_inputs = self._get_input_edges(first_preparation_id)
        second_preparation_inputs = self._get_input_edges(second_preparation_id)
        body_inputs = self._get_input_edges(body_node_id)
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
        if self._get_output_edges(node_id, ITEM_FIELD) != first_preparation_inputs:
            return None
        if self._get_output_edges(first_iterate_id, ITEM_FIELD) != second_preparation_inputs:
            return None
        if self._get_output_edges(second_iterate_id, ITEM_FIELD) != body_inputs:
            return None
        if self._get_output_edges(body_node_id) != collect_item_edges:
            return None

        return _SupportedNestedIterateChain(
            body_path_nodes=frozenset(body_path_nodes),
            return_node_id=return_node_id,
            iterate_node_ids=(first_iterate_id, second_iterate_id),
            collect_node_id=collect_node_id,
        )

    def _get_supported_nested_iterate_sequence(
        self, graph: "nx.DiGraph", *, iterate_count: int = 2
    ) -> _SupportedNestedIterateSequence | None:
        """Return exact serial nested-Iterate chain of requested depth, if present."""
        if iterate_count < 2 or len(self.nodes) != iterate_count * 2 + 1 or len(self.edges) != iterate_count * 2:
            return None
        iterate_node_ids = tuple(node_id for node_id, node in self.nodes.items() if isinstance(node, IterateInvocation))
        if len(iterate_node_ids) != iterate_count:
            return None
        outer_candidates = [
            node_id
            for node_id in iterate_node_ids
            if self._get_input_edges(node_id, COLLECTION_FIELD)
            and not self._get_input_edges(self._get_input_edges(node_id, COLLECTION_FIELD)[0].source.node_id)
        ]
        if len(outer_candidates) != 1:
            return None

        ordered_iterate_ids = [outer_candidates[0]]
        preparation_node_ids: list[str] = []
        source_node_id = self._get_input_edges(ordered_iterate_ids[0], COLLECTION_FIELD)[0].source.node_id
        remaining_iterate_ids = set(iterate_node_ids) - set(ordered_iterate_ids)
        for _ in range(iterate_count - 1):
            candidates: list[tuple[str, str]] = []
            for candidate_id in remaining_iterate_ids:
                collection_edges = self._get_input_edges(candidate_id, COLLECTION_FIELD)
                if len(collection_edges) != 1:
                    continue
                preparation_id = collection_edges[0].source.node_id
                preparation_inputs = self._get_input_edges(preparation_id)
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
            for node_id in self.nodes
            if node_id not in set(ordered_iterate_ids) | set(preparation_node_ids) | {source_node_id}
        ]
        if len(body_candidates) != 1:
            return None
        body_node_id = body_candidates[0]
        control_types = (
            CallSavedWorkflowInvocation,
            IfInvocation,
            ForInvocation,
            ForReturnInvocation,
            CollectInvocation,
            IterateInvocation,
        )
        if any(
            isinstance(self.get_node(node_id), control_types)
            for node_id in {source_node_id, *preparation_node_ids, body_node_id}
        ):
            return None
        outer_collection_edges = self._get_input_edges(ordered_iterate_ids[0], COLLECTION_FIELD)
        body_inputs = self._get_input_edges(body_node_id)
        if self._get_input_edges(source_node_id) or len(outer_collection_edges) != 1 or len(body_inputs) != 1:
            return None
        if self._get_output_edges(source_node_id) != outer_collection_edges:
            return None
        for index, preparation_id in enumerate(preparation_node_ids):
            preparation_inputs = self._get_input_edges(preparation_id)
            next_collection_edges = self._get_input_edges(ordered_iterate_ids[index + 1], COLLECTION_FIELD)
            if (
                len(preparation_inputs) != 1
                or preparation_inputs[0].source.node_id != ordered_iterate_ids[index]
                or preparation_inputs[0].source.field != ITEM_FIELD
                or len(next_collection_edges) != 1
                or next_collection_edges[0].source.node_id != preparation_id
                or self._get_output_edges(ordered_iterate_ids[index], ITEM_FIELD) != preparation_inputs
                or self._get_output_edges(preparation_id) != next_collection_edges
            ):
                return None
        if (
            body_inputs[0].source.node_id != ordered_iterate_ids[-1]
            or body_inputs[0].source.field != ITEM_FIELD
            or self._get_output_edges(ordered_iterate_ids[-1], ITEM_FIELD) != body_inputs
            or self._get_output_edges(body_node_id)
        ):
            return None
        return _SupportedNestedIterateSequence(
            body_path_nodes=frozenset(self.nodes),
            source_node_id=source_node_id,
            iterate_node_ids=tuple(ordered_iterate_ids),
            preparation_node_ids=tuple(preparation_node_ids),
            body_node_id=body_node_id,
        )

    def _get_supported_for_nested_for_body(self, node_id: str, graph: "nx.DiGraph") -> _SupportedNestedForBody | None:
        """Returns the supported recursive nested For contract, if this For uses it.

        Each direct child loop has its own ForReturn. A single child may close the parent directly or through a
        continuation. Multiple independent child loops must all feed an ordinary parent-scoped continuation, which acts
        as an explicit fan-in barrier after every child has finalized for the current parent iteration.

        Accepted shape (the child body can recursively contain this same shape)::

            outer For -> preparation -> inner For(s) -> continuation -> outer ForReturn
                                           |    ^
                                           v    |
                                         child body -> inner ForReturn

        Preparation, child bodies, and continuation partition the reachable outer body. Only finalized child outputs
        may cross into the continuation; iteration outputs stay inside their owning child body.
        """
        outer_node = self.get_node(node_id)
        if not isinstance(outer_node, ForInvocation):
            return None

        iteration_edges = self._get_for_iteration_output_edges(node_id)
        if len(iteration_edges) == 0:
            return None
        reachable_body_nodes = self._get_for_reachable_body_nodes(iteration_edges, graph)
        reachable_return_ids = [
            body_node_id
            for body_node_id in reachable_body_nodes
            if isinstance(self.get_node(body_node_id), ForReturnInvocation)
        ]
        outer_return_id = self._get_linked_for_return_id(node_id)
        if outer_return_id is None or outer_return_id not in reachable_return_ids:
            return None

        nested_for_ids = [
            body_node_id
            for body_node_id in reachable_body_nodes
            if isinstance(self.get_node(body_node_id), ForInvocation) and body_node_id != node_id
        ]
        direct_nested_for_ids = [
            nested_for_id
            for nested_for_id in nested_for_ids
            if not any(
                other_nested_for_id != nested_for_id and nx.has_path(graph, other_nested_for_id, nested_for_id)
                for other_nested_for_id in nested_for_ids
            )
        ]
        if not direct_nested_for_ids:
            return None
        direct_nested_for_ids = tuple(
            nested_for_id for nested_for_id in nx.topological_sort(graph) if nested_for_id in direct_nested_for_ids
        )

        inner_body_path_nodes: set[str] = set()
        for inner_for_id in direct_nested_for_ids:
            inner_for = self.get_node(inner_for_id)
            assert isinstance(inner_for, ForInvocation)
            inner_return_id = self._get_linked_for_return_id(inner_for_id)
            if inner_return_id is None:
                return None

            inner_body_path_to_return = self._get_for_body_path_to_return(inner_for_id, graph)
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
                if isinstance(self.get_node(body_node_id), ForInvocation)
            ]
            if any(
                isinstance(self.get_node(body_node_id), IterateInvocation) for body_node_id in child_body_path_nodes
            ):
                return None
            inner_nested_body = (
                self._get_supported_for_nested_for_body(inner_for_id, graph) if inner_nested_for_ids else None
            )
            if inner_nested_for_ids and inner_nested_body is None:
                return None
            if inner_nested_body is not None:
                child_body_path_nodes = child_body_path_nodes | inner_nested_body.body_path_nodes
            if any(
                edge.destination.field == "state"
                and edge.source.node_id != inner_for_id
                and edge.source.node_id not in child_body_path_nodes
                for edge in self._get_input_edges(inner_return_id)
            ):
                return None

            inner_collection_edges = self._get_input_edges(inner_for_id, COLLECTION_FIELD)
            if len(inner_collection_edges) != 1:
                return None
            inner_collection_source_id = inner_collection_edges[0].source.node_id
            if inner_collection_source_id != node_id and inner_collection_source_id not in reachable_body_nodes:
                return None

            inner_body_path_nodes.update(child_body_path_nodes)

        if set(reachable_return_ids) - inner_body_path_nodes != {outer_return_id}:
            return None

        outer_output_edges = self._get_input_edges(outer_return_id, "output")
        if len(outer_output_edges) != 1:
            return None
        if any(
            edge.destination.field != "output"
            and edge.destination.field != "continue_condition"
            and (edge.destination.field != "state" or edge.source.node_id != node_id or edge.source.field != "state")
            for edge in self._get_input_edges(outer_return_id)
        ):
            return None

        outer_preparation_nodes = {
            body_node_id
            for inner_for_id in direct_nested_for_ids
            for body_node_id in reachable_body_nodes & nx.ancestors(graph, inner_for_id)
        } | set(direct_nested_for_ids)
        inner_final_descendants: set[str] = set()
        for inner_for_id in direct_nested_for_ids:
            for edge in self._get_for_final_output_edges(inner_for_id):
                inner_final_descendants.add(edge.destination.node_id)
                inner_final_descendants.update(nx.descendants(graph, edge.destination.node_id))
        continuation_nodes = reachable_body_nodes - outer_preparation_nodes - inner_body_path_nodes - {outer_return_id}
        if any(
            edge.destination.field == "continue_condition"
            and edge.source.node_id != node_id
            and edge.source.node_id not in continuation_nodes
            and not (
                edge.source.node_id in direct_nested_for_ids
                and edge.source.field in {"output_collection", "final_state"}
            )
            for edge in self._get_input_edges(outer_return_id)
        ):
            return None
        if not continuation_nodes <= inner_final_descendants:
            return None
        if any(not nx.has_path(graph, body_node_id, outer_return_id) for body_node_id in continuation_nodes):
            return None
        if any(
            isinstance(self.get_node(body_node_id), (ForInvocation, IterateInvocation, ForReturnInvocation))
            for body_node_id in continuation_nodes
        ):
            return None
        if any(
            edge.source.node_id in inner_body_path_nodes
            or (edge.source.node_id in direct_nested_for_ids and edge.source.field != "output_collection")
            for body_node_id in continuation_nodes
            for edge in self._get_input_edges(body_node_id)
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
                for edge in self._get_for_final_output_edges(inner_for_id)
            )
            for inner_for_id in direct_nested_for_ids
        ):
            return None

        allowed_body_nodes = outer_preparation_nodes | inner_body_path_nodes | continuation_nodes | {outer_return_id}
        if reachable_body_nodes != allowed_body_nodes:
            return None

        if any(
            isinstance(self.get_node(body_node_id), (ForInvocation, IterateInvocation))
            for body_node_id in outer_preparation_nodes
            if body_node_id not in direct_nested_for_ids
        ):
            return None

        for body_node_id in allowed_body_nodes - {outer_return_id, *direct_nested_for_ids}:
            if body_node_id in inner_body_path_nodes or body_node_id in continuation_nodes:
                continue
            if not any(nx.has_path(graph, body_node_id, inner_for_id) for inner_for_id in direct_nested_for_ids):
                return None
        return _SupportedNestedForBody(
            body_path_nodes=frozenset(allowed_body_nodes),
            outer_return_id=outer_return_id,
            inner_for_ids=direct_nested_for_ids,
            continuation_nodes=frozenset(continuation_nodes),
        )

    def _get_for_nested_for_continuation_nodes(self, nested_body: _SupportedNestedForBody) -> set[str]:
        return set(nested_body.continuation_nodes)

    def _is_for_connection_valid(self, node_id: str) -> str | None:
        if len(self._get_input_edges(node_id, COLLECTION_FIELD)) > 1:
            return "For loop may have only one collection input edge"
        if len(self._get_input_edges(node_id, "state")) > 1:
            return "For loop may have only one state input edge"

        iteration_edges = self._get_for_iteration_output_edges(node_id)
        if len(iteration_edges) == 0:
            return "For loop must have at least one iteration output edge"

        graph = self.nx_graph_flat()
        reachable_body_nodes = self._get_for_reachable_body_nodes(iteration_edges, graph)

        nested_for_node_ids = [
            body_node_id
            for body_node_id in reachable_body_nodes
            if body_node_id != node_id
            and isinstance(self.get_node(body_node_id), ForInvocation)
            and not any(
                other_body_node_id != body_node_id
                and isinstance(self.get_node(other_body_node_id), ForInvocation)
                and nx.has_path(graph, other_body_node_id, body_node_id)
                for other_body_node_id in reachable_body_nodes
            )
        ]
        nested_body = self._get_supported_for_nested_for_body(node_id, graph) if nested_for_node_ids else None
        if nested_for_node_ids and nested_body is None:
            return "Nested For loops require one linked inner For with a matching ForReturn"

        if nested_body is not None:
            body_path_nodes = nested_body.body_path_nodes
            return_node_id = nested_body.outer_return_id
        else:
            return_node_id = self._get_linked_for_return_id(node_id)
            if return_node_id is None or return_node_id not in reachable_body_nodes:
                return "For loop body must expose exactly one matching ForReturn"
            body_path_nodes = self._get_for_body_path_nodes(reachable_body_nodes, return_node_id, graph)

        unterminated_body_nodes = reachable_body_nodes - body_path_nodes
        if len(unterminated_body_nodes) > 0:
            return "For loop body paths must terminate at the matching ForReturn and not escape the loop body"

        if any(isinstance(self.get_node(body_node_id), IterateInvocation) for body_node_id in body_path_nodes):
            if (
                self._get_supported_for_nested_iterate_body(node_id, graph) is None
                and self._get_supported_for_serial_nested_iterate_chain(node_id, graph) is None
            ):
                return "Iterate nodes inside For loop bodies are unsupported"

        for body_node_id in body_path_nodes:
            for edge in self._get_input_edges(body_node_id):
                source_node_id = edge.source.node_id
                if source_node_id == node_id or source_node_id in body_path_nodes:
                    continue
                active_source_scope = nx.ancestors(graph, source_node_id) | {source_node_id}
                if any(isinstance(self.get_node(source_id), IterateInvocation) for source_id in active_source_scope):
                    return "For loop body does not support iterator-derived external inputs"

        for edge in self._get_for_final_output_edges(node_id):
            if edge.destination.node_id in body_path_nodes or nx.has_path(
                graph, edge.destination.node_id, return_node_id
            ):
                return "final-scoped For outputs cannot feed the loop body"

        for body_node_id in body_path_nodes:
            if body_node_id == return_node_id:
                continue
            for edge in self._get_output_edges(body_node_id):
                if edge.destination.node_id not in body_path_nodes:
                    return "For loop body paths must not escape before the matching ForReturn"

        return None

    def _is_for_return_connection_valid(self, node_id: str) -> str | None:
        graph = self.nx_graph_flat()
        matching_for_node_ids = []
        for loop_node_id, loop_node in self.nodes.items():
            if not isinstance(loop_node, ForInvocation):
                continue
            body_path_to_return = self._get_for_body_path_to_return(loop_node_id, graph)
            if body_path_to_return is None:
                continue
            body_path_nodes, return_node_id = body_path_to_return
            if node_id == return_node_id and node_id in body_path_nodes:
                matching_for_node_ids.append(loop_node_id)

        if len(matching_for_node_ids) != 1:
            return "ForReturn must belong to exactly one matching For"

        if (
            len(self._get_input_edges(node_id, "output")) > 1
            or len(self._get_input_edges(node_id, "state")) > 1
            or len(self._get_input_edges(node_id, "continue_condition")) > 1
        ):
            return "ForReturn may have only one input edge per field"
        return None

    def _is_iterator_connection_valid(
        self,
        node_id: str,
        new_input: Optional[EdgeConnection] = None,
        new_output: Optional[EdgeConnection] = None,
    ) -> str | None:
        inputs = [e.source for e in self._get_input_edges(node_id, COLLECTION_FIELD)]
        outputs = [e.destination for e in self._get_output_edges(node_id, ITEM_FIELD)]

        if new_input is not None:
            inputs.append(new_input)
        if new_output is not None:
            outputs.append(new_output)

        return self._validate_iterator_connections(inputs, outputs)

    def _validate_iterator_connections(self, inputs: list[EdgeConnection], outputs: list[EdgeConnection]) -> str | None:
        presence_error = self._validate_iterator_input_presence(inputs)
        if presence_error is not None:
            return presence_error

        input_node = self.get_node(inputs[0].node_id)
        input_sources = self._get_effective_output_connections(inputs[0].node_id, inputs[0].field)
        input_field_types = (
            [get_output_field_type(self.get_node(source.node_id), source.field) for source in input_sources]
            if input_sources is not None
            else [get_output_field_type(input_node, inputs[0].field)]
        )
        output_field_types = self._get_iterator_output_field_types(outputs)

        for input_field_type in input_field_types:
            input_type_error = self._validate_iterator_input_type(input_field_type)
            if input_type_error is not None:
                return input_type_error

            output_type_error = self._validate_iterator_output_types(input_field_type, output_field_types)
            if output_type_error is not None:
                return output_type_error

        if input_sources is None:
            return self._validate_iterator_collector_input(input_node, output_field_types)

        for source in input_sources:
            collector_input_error = self._validate_iterator_collector_input(
                self.get_node(source.node_id), output_field_types
            )
            if collector_input_error is not None:
                return collector_input_error
        return None

    def _validate_iterator_input_presence(self, inputs: list[EdgeConnection]) -> str | None:
        if len(inputs) == 0:
            return "Iterator must have a collection input edge"
        if len(inputs) > 1:
            return "Iterator may only have one input edge"
        return None

    def _get_iterator_output_field_types(self, outputs: list[EdgeConnection]) -> list[Any]:
        return [get_input_field_type(self.get_node(e.node_id), e.field) for e in outputs]

    def _validate_iterator_input_type(self, input_field_type: Any) -> str | None:
        if get_origin(input_field_type) is not list:
            return "Iterator input must be a collection"
        return None

    def _validate_iterator_output_types(self, input_field_type: Any, output_field_types: list[Any]) -> str | None:
        input_field_item_type = get_args(input_field_type)[0]
        if not all(are_connection_types_compatible(input_field_item_type, t) for t in output_field_types):
            return "Iterator outputs must connect to an input with a matching type"
        return None

    def _validate_iterator_collector_input(
        self, input_node: BaseInvocation, output_field_types: list[Any]
    ) -> str | None:
        if not isinstance(input_node, CollectInvocation):
            return None

        input_root_type = self._get_collector_input_root_type(input_node.id)
        if input_root_type is None:
            return "Iterator input collector must have at least one item or collection input edge"
        if not all(are_connection_types_compatible(input_root_type, t) for t in output_field_types):
            return "Iterator collection type must match all iterator output types"
        return None

    def _get_effective_output_connections(self, node_id: str, field: str) -> list[EdgeConnection] | None:
        """Resolve If.value to its unique branch sources, or None when a branch is unresolved.

        Callers validate acyclicity before resolving connections.
        """
        pending = [(node_id, field)]
        visited: set[tuple[str, str]] = set()
        sources: list[EdgeConnection] = []
        while pending:
            source_node_id, source_field = pending.pop()
            source = (source_node_id, source_field)
            if source in visited:
                continue
            visited.add(source)

            node = self.get_node(source_node_id)
            if not isinstance(node, IfInvocation) or source_field != "value":
                sources.append(EdgeConnection(node_id=source_node_id, field=source_field))
                continue

            branch_sources: list[tuple[str, str]] = []
            for branch_field in ("true_input", "false_input"):
                branch_edges = self._get_input_edges(source_node_id, branch_field)
                if len(branch_edges) != 1:
                    return None
                branch_edge = branch_edges[0]
                branch_sources.append((branch_edge.source.node_id, branch_edge.source.field))

            pending.extend(reversed(branch_sources))

        return sources

    def _get_effective_output_field_types(self, node_id: str, field: str) -> set[Any]:
        node = self.get_node(node_id)
        if not isinstance(node, IfInvocation) or field != "value":
            return {get_output_field_type(node, field)}

        sources = self._get_effective_output_connections(node_id, field)
        if sources is None:
            return {Any}

        source_types = {get_output_field_type(self.get_node(source.node_id), source.field) for source in sources}
        if not source_types or any(is_any(source_type) for source_type in source_types):
            return {Any}
        return source_types

    def _are_effective_connections_compatible(
        self,
        source_node: BaseInvocation,
        source_field: str,
        destination_node: BaseInvocation,
        destination_field: str,
    ) -> bool:
        """Checks each known branch of an If output against a target while preserving compatibility hooks."""
        if not isinstance(source_node, IfInvocation) or source_field != "value":
            return are_connections_compatible(source_node, source_field, destination_node, destination_field)
        sources = self._get_effective_output_connections(source_node.id, source_field)
        if sources is None:
            return are_connections_compatible(source_node, source_field, destination_node, destination_field)
        return all(
            are_connections_compatible(self.get_node(source.node_id), source.field, destination_node, destination_field)
            for source in sources
        )

    def _resolve_collector_input_types(self, node_id: str, visited: Optional[set[str]] = None) -> set[Any]:
        """Resolves possible item types for a collector's inputs, recursively following chained collectors."""
        visited = visited or set()
        if node_id in visited:
            return set()
        visited.add(node_id)

        input_types: set[Any] = set()

        for edge in self._get_input_edges(node_id, ITEM_FIELD):
            input_field_types = list(self._get_effective_output_field_types(edge.source.node_id, edge.source.field))
            input_types.update(self._resolve_item_input_types(input_field_types))

        for edge in self._get_input_edges(node_id, COLLECTION_FIELD):
            source_node = self.get_node(edge.source.node_id)
            if isinstance(source_node, CollectInvocation) and edge.source.field == COLLECTION_FIELD:
                input_types.update(self._resolve_collector_input_types(source_node.id, visited.copy()))
                continue

            for input_field_type in self._get_effective_output_field_types(edge.source.node_id, edge.source.field):
                input_types.update(extract_collection_item_types(input_field_type))

        return input_types

    def _get_type_tree_root_types(self, input_types: set[Any]) -> list[Any]:
        type_tree = nx.DiGraph()
        type_tree.add_nodes_from(input_types)
        type_tree.add_edges_from([e for e in itertools.permutations(input_types, 2) if issubclass(e[1], e[0])])
        type_degrees = type_tree.in_degree(type_tree.nodes)
        return [t[0] for t in type_degrees if t[1] == 0]  # type: ignore

    def _get_collector_input_root_type(self, node_id: str) -> Any | None:
        input_types = self._resolve_collector_input_types(node_id)
        has_multiple_root_types, input_root_type = self._get_collector_input_root_type_from_resolved_types(input_types)
        if has_multiple_root_types:
            return Any
        if input_root_type is not None:
            return input_root_type
        if Any in input_types:
            return Any
        return None

    def _get_collector_connections(
        self,
        node_id: str,
        new_input: Optional[EdgeConnection] = None,
        new_input_field: Optional[str] = None,
        new_output: Optional[EdgeConnection] = None,
    ) -> tuple[list[EdgeConnection], list[EdgeConnection], list[EdgeConnection]]:
        item_inputs = [e.source for e in self._get_input_edges(node_id, ITEM_FIELD)]
        collection_inputs = [e.source for e in self._get_input_edges(node_id, COLLECTION_FIELD)]
        outputs = [e.destination for e in self._get_output_edges(node_id, COLLECTION_FIELD)]

        if new_input is not None:
            field = new_input_field or ITEM_FIELD
            if field == ITEM_FIELD:
                item_inputs.append(new_input)
            elif field == COLLECTION_FIELD:
                collection_inputs.append(new_input)

        if new_output is not None:
            outputs.append(new_output)

        return item_inputs, collection_inputs, outputs

    def _get_collector_port_types(
        self,
        item_inputs: list[EdgeConnection],
        collection_inputs: list[EdgeConnection],
        outputs: list[EdgeConnection],
    ) -> tuple[list[Any], list[Any], list[Any]]:
        item_input_field_types = [
            field_type
            for edge in item_inputs
            for field_type in self._get_effective_output_field_types(edge.node_id, edge.field)
        ]
        collection_input_field_types = [
            field_type
            for edge in collection_inputs
            for field_type in self._get_effective_output_field_types(edge.node_id, edge.field)
        ]
        output_field_types = [get_input_field_type(self.get_node(e.node_id), e.field) for e in outputs]
        return item_input_field_types, collection_input_field_types, output_field_types

    def _resolve_item_input_types(self, item_input_field_types: list[Any]) -> set[Any]:
        return {
            resolved_type
            for input_field_type in item_input_field_types
            for resolved_type in (
                [input_field_type] if get_origin(input_field_type) is None else get_args(input_field_type)
            )
            if resolved_type != NoneType
        }

    def _resolve_collection_input_types(self, collection_inputs: list[EdgeConnection]) -> set[Any]:
        input_field_types: set[Any] = set()
        for input_conn in collection_inputs:
            source_node = self.get_node(input_conn.node_id)
            if isinstance(source_node, CollectInvocation) and input_conn.field == COLLECTION_FIELD:
                input_field_types.update(self._resolve_collector_input_types(source_node.id))
                continue
            for input_field_type in self._get_effective_output_field_types(input_conn.node_id, input_conn.field):
                input_field_types.update(extract_collection_item_types(input_field_type))
        return input_field_types

    def _validate_collector_collection_inputs(self, collection_input_field_types: list[Any]) -> str | None:
        if not all((is_list_or_contains_list(t) or is_any(t) for t in collection_input_field_types)):
            return "Collector collection input must be a collection"
        return None

    def _get_collector_input_root_type_from_resolved_types(
        self, input_field_types: set[Any]
    ) -> tuple[bool, Any | None]:
        non_any_input_field_types = {t for t in input_field_types if t != Any}
        if non_any_input_field_types == {int, float}:
            return False, float

        root_types = self._get_type_tree_root_types(non_any_input_field_types)
        if len(root_types) > 1:
            return True, None
        return False, root_types[0] if len(root_types) == 1 else None

    def _validate_collector_output_types(
        self, output_field_types: list[Any], input_root_type: Any | None
    ) -> str | None:
        if not all(is_list_or_contains_list(t) or is_any(t) for t in output_field_types):
            return "Collector output must connect to a collection input"

        if input_root_type is not None:
            if not all(
                is_any(t)
                or is_union_subtype(input_root_type, get_args(t)[0])
                or issubclass(input_root_type, get_args(t)[0])
                for t in output_field_types
            ):
                return "Collector outputs must connect to a collection input with a matching type"
        elif any(not is_any(t) and get_args(t)[0] != Any for t in output_field_types):
            return "Collector outputs must connect to a collection input with a matching type"

        return None

    def _validate_downstream_collector_outputs(
        self, outputs: list[EdgeConnection], input_root_type: Any | None
    ) -> str | None:
        for output in outputs:
            output_node = self.get_node(output.node_id)
            if not isinstance(output_node, CollectInvocation) or output.field != COLLECTION_FIELD:
                continue
            output_root_type = self._get_collector_input_root_type(output_node.id)
            if output_root_type is None:
                continue
            if input_root_type is None:
                if output_root_type != Any:
                    return "Collector outputs must connect to a collection input with a matching type"
                continue
            if not are_connection_types_compatible(input_root_type, output_root_type):
                return "Collector outputs must connect to a collection input with a matching type"
        return None

    def _is_collector_connection_valid(
        self,
        node_id: str,
        new_input: Optional[EdgeConnection] = None,
        new_input_field: Optional[str] = None,
        new_output: Optional[EdgeConnection] = None,
    ) -> str | None:
        item_inputs, collection_inputs, outputs = self._get_collector_connections(
            node_id, new_input=new_input, new_input_field=new_input_field, new_output=new_output
        )

        if len(item_inputs) == 0 and len(collection_inputs) == 0:
            return "Collector must have at least one item or collection input edge"

        item_input_field_types, collection_input_field_types, output_field_types = self._get_collector_port_types(
            item_inputs, collection_inputs, outputs
        )

        collection_input_error = self._validate_collector_collection_inputs(collection_input_field_types)
        if collection_input_error is not None:
            return collection_input_error

        input_field_types = self._resolve_item_input_types(item_input_field_types)
        input_field_types.update(self._resolve_collection_input_types(collection_inputs))

        has_multiple_root_types, input_root_type = self._get_collector_input_root_type_from_resolved_types(
            input_field_types
        )
        if has_multiple_root_types:
            return "Collector input collection items must be of a single type"

        output_type_error = self._validate_collector_output_types(output_field_types, input_root_type)
        if output_type_error is not None:
            return output_type_error

        downstream_output_error = self._validate_downstream_collector_outputs(outputs, input_root_type)
        if downstream_output_error is not None:
            return downstream_output_error

        return None

    def nx_graph(self) -> "nx.DiGraph":
        """Returns a NetworkX DiGraph representing the layout of this graph"""
        # TODO: Cache this?
        g = nx.DiGraph()
        g.add_nodes_from(list(self.nodes.keys()))
        edges = dict.fromkeys((e.source.node_id, e.destination.node_id) for e in self.edges if e.type == "default")
        g.add_edges_from(edges)
        return g

    def nx_graph_flat(self, nx_graph: Optional["nx.DiGraph"] = None) -> "nx.DiGraph":
        """Returns a flattened NetworkX DiGraph, including all subgraphs (but not with iterations expanded)"""
        g = nx_graph or nx.DiGraph()

        # Add all nodes from this graph except graph/iteration nodes
        g.add_nodes_from([n.id for n in self.nodes.values()])

        edges = dict.fromkeys((e.source.node_id, e.destination.node_id) for e in self.edges if e.type == "default")
        g.add_edges_from(edges)
        return g


def _edge_index_property(name: str) -> property:
    def getter(graph: Graph) -> Optional[dict[str, list[Edge]]]:
        return getattr(graph.__pydantic_private__["_edge_indexes"], name)

    def setter(graph: Graph, value: Optional[dict[str, list[Edge]]]) -> None:
        setattr(graph.__pydantic_private__["_edge_indexes"], name, value)

    return property(getter, setter)


Graph._input_edges_by_node = _edge_index_property("input_edges_by_node")
Graph._output_edges_by_node = _edge_index_property("output_edges_by_node")
