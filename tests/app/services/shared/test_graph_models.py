import subprocess
import sys
import textwrap

import pytest
from fastapi import FastAPI
from pydantic import TypeAdapter, model_validator
from pydantic.json_schema import models_json_schema

from invokeai.app.invocations.logic import IfInvocation
from invokeai.app.invocations.loops import ForInvocation, ForReturnInvocation
from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.shared import graph as graph_facade
from invokeai.app.services.shared import graph_models, graph_validation
from invokeai.app.util.custom_openapi import get_openapi_func


def test_queue_openapi_excludes_runtime_ledgers_and_internal_components() -> None:
    app = FastAPI(separate_input_output_schemas=False)
    app.get("/queue-item", response_model=SessionQueueItem)(lambda: None)
    app.openapi = get_openapi_func(app)

    schemas = app.openapi()["components"]["schemas"]
    assert {
        "execution_refs",
        "execution_tokens",
        "execution_effects",
        "execution_child_dependencies",
    }.isdisjoint(schemas["GraphExecutionState"]["properties"])
    assert {
        "ExecutionFrame",
        "ExecutionReference",
        "ExecutionToken",
        "ChildCompletion",
        "ChildDependencyRecord",
        "ChildExecutionRecord",
    }.isdisjoint(schemas)


def test_graph_facade_reexports_authoring_graph_boundary() -> None:
    exported_names = (
        "Graph",
        "_EdgeList",
        "AnyInvocation",
        "AnyInvocationOutput",
        "IterateInvocation",
        "IterateInvocationOutput",
        "CollectInvocation",
        "CollectInvocationOutput",
        "ITEM_FIELD",
        "COLLECTION_FIELD",
        "NoneType",
        "get_output_field_type",
        "get_output_field_scope",
        "get_input_field_type",
        "copydeep",
        "is_any",
        "is_list_or_contains_list",
        "is_union_subtype",
        "extract_collection_item_types",
        "are_connection_types_compatible",
        "are_connections_compatible",
        "NodeAlreadyInGraphError",
        "InvalidEdgeError",
        "NodeNotFoundError",
        "NodeAlreadyExecutedError",
        "DuplicateNodeIdError",
        "NodeFieldNotFoundError",
        "NodeIdMismatchError",
        "CyclicalGraphError",
        "UnknownGraphValidationError",
        "NodeInputError",
        "loc_to_dot_sep",
        "_SupportedNestedForBody",
        "_SupportedNestedIterateBody",
        "nx",
    )

    for name in exported_names:
        assert getattr(graph_facade, name) is getattr(graph_validation, name)

    assert graph_facade.model_validator is model_validator


def test_graph_facade_preserves_authoring_helper_patch_points(monkeypatch: pytest.MonkeyPatch) -> None:
    source = AddInvocation(id="source", a=1, b=2)
    destination = AddInvocation(id="destination", a=3, b=4)
    edge = graph_facade.Edge(
        source=graph_facade.EdgeConnection(node_id="source", field="value"),
        destination=graph_facade.EdgeConnection(node_id="destination", field="a"),
    )
    graph = graph_facade.Graph(nodes={source.id: source, destination.id: destination})

    monkeypatch.setattr(graph_facade, "are_connections_compatible", lambda *_args: False)
    with pytest.raises(graph_facade.InvalidEdgeError):
        graph.add_edge(edge)

    monkeypatch.setattr(graph_facade, "get_input_field_type", lambda *_args: None)
    assert not graph_facade.are_connections_compatible(source, "value", destination, "a")


def test_graph_facade_override_can_delegate_to_saved_original(monkeypatch: pytest.MonkeyPatch) -> None:
    source = AddInvocation(id="source", a=1, b=2)
    destination = AddInvocation(id="destination", a=3, b=4)
    original = graph_facade.are_connections_compatible
    calls = 0

    def wrapped(*args: object, **kwargs: object) -> bool:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_facade, "are_connections_compatible", wrapped)

    assert graph_facade.are_connections_compatible(source, "value", destination, "a")
    assert calls == 1


def test_graph_facade_reexports_graph_models() -> None:
    exported_names = (
        "EdgeConnection",
        "Edge",
        "WorkflowCallFrame",
        "WorkflowCallExecution",
        "WorkflowCallParentRef",
        "ExecutionFrame",
        "ExecutionReference",
        "ExecutionToken",
        "PreparedExecState",
        "WorkflowCallStatus",
        "ExecutionRef",
        "PreparedExecutionRef",
    )

    for name in exported_names:
        assert getattr(graph_facade, name) is getattr(graph_models, name)

    assert graph_facade.model_validator is model_validator


def test_graph_models_preserve_graph_and_execution_state_serialization() -> None:
    source_graph = graph_facade.Graph()
    source_graph.add_node(AddInvocation(id="source", a=1, b=2))
    source_graph.add_node(AddInvocation(id="destination", a=3, b=4))
    edge = graph_facade.Edge(
        source=graph_facade.EdgeConnection(node_id="source", field="value"),
        destination=graph_facade.EdgeConnection(node_id="destination", field="a"),
    )
    source_graph.add_edge(edge)

    execution_ref = graph_facade.ExecutionReference(
        reference_id="state-id:source",
        state_id="state-id",
        exec_node_id="source",
        source_node_id="source",
        frame=graph_facade.ExecutionFrame(state_id="state-id", frame_id="frame-id"),
    )
    execution_token = graph_facade.ExecutionToken(
        token_id="token-id",
        reference_id=execution_ref.reference_id,
        owner_node_id="source",
        port="value",
        frame=execution_ref.frame,
        value=5,
    )
    state = graph_facade.GraphExecutionState(
        id="state-id",
        graph=source_graph,
        execution_graph=source_graph.model_copy(deep=True),
        execution_refs={execution_ref.reference_id: execution_ref},
        execution_tokens={execution_token.token_id: execution_token},
    )

    graph_json = source_graph.model_dump_json()
    restored_graph = TypeAdapter(graph_facade.Graph).validate_json(graph_json)
    assert restored_graph.model_dump(mode="json", warnings=False) == source_graph.model_dump(
        mode="json", warnings=False
    )

    state_json = state.model_dump_json(exclude_none=True)
    restored_state = graph_facade.GraphExecutionState.model_validate_json(state_json)
    assert restored_state.model_dump(mode="json", warnings=False, exclude_none=True) == state.model_dump(
        mode="json", warnings=False, exclude_none=True
    )

    schema_models = [
        (graph_facade.Graph, "serialization"),
        (graph_facade.GraphExecutionState, "serialization"),
        (graph_facade.Edge, "serialization"),
        (graph_facade.EdgeConnection, "serialization"),
        (graph_facade.WorkflowCallFrame, "serialization"),
        (graph_facade.WorkflowCallExecution, "serialization"),
        (graph_facade.WorkflowCallParentRef, "serialization"),
        (graph_facade.ExecutionFrame, "serialization"),
        (graph_facade.ExecutionReference, "serialization"),
        (graph_facade.ExecutionToken, "serialization"),
    ]
    _, schema = models_json_schema(schema_models)
    definitions = schema["$defs"]
    assert {
        "Graph",
        "GraphExecutionState",
        "Edge",
        "EdgeConnection",
        "WorkflowCallFrame",
        "WorkflowCallExecution",
        "WorkflowCallParentRef",
        "ExecutionFrame",
        "ExecutionReference",
        "ExecutionToken",
    } <= definitions.keys()
    assert {"id", "nodes", "edges"} <= definitions["Graph"]["properties"].keys()
    assert {"id", "graph", "execution_graph"} <= definitions["GraphExecutionState"]["properties"].keys()
    assert {"execution_refs", "execution_tokens", "execution_effects"}.isdisjoint(
        definitions["GraphExecutionState"]["properties"].keys()
    )
    assert {"source", "destination", "type"} <= definitions["Edge"]["properties"].keys()
    assert {"reference_id", "exec_node_id", "frame"} <= definitions["ExecutionReference"]["properties"].keys()
    assert {"token_id", "reference_id", "owner_node_id", "value"} <= definitions["ExecutionToken"]["properties"].keys()

    _, queue_schema = models_json_schema([(SessionQueueItem, "serialization")])
    queue_graph_schema = queue_schema["$defs"]["GraphExecutionState"]
    assert {"execution_refs", "execution_tokens", "execution_effects"}.isdisjoint(
        queue_graph_schema["properties"].keys()
    )

    node_refs = {
        reference["$ref"] for reference in definitions["Graph"]["properties"]["nodes"]["additionalProperties"]["oneOf"]
    }
    assert {
        "#/components/schemas/AddInvocation",
        "#/components/schemas/IfInvocation",
        "#/components/schemas/IterateInvocation",
        "#/components/schemas/CollectInvocation",
    } <= node_refs

    output_schemas, _ = models_json_schema([(graph_facade.AnyInvocationOutput, "serialization")])
    output_refs = {
        reference["$ref"] for reference in output_schemas[(graph_facade.AnyInvocationOutput, "serialization")]["oneOf"]
    }
    assert {
        "#/components/schemas/IfInvocationOutput",
        "#/components/schemas/IterateInvocationOutput",
        "#/components/schemas/CollectInvocationOutput",
    } <= output_refs


def test_graph_models_preserve_heterogeneous_control_flow_roundtrip() -> None:
    source_graph = graph_facade.Graph()
    nodes = {
        "if": IfInvocation(id="if", condition=True),
        "iterate": graph_facade.IterateInvocation(id="iterate", collection=[1]),
        "collect": graph_facade.CollectInvocation(id="collect"),
        "for": ForInvocation(id="for", collection=[1]),
        "return": ForReturnInvocation(id="return"),
    }
    for node in nodes.values():
        source_graph.add_node(node)
    source_graph.add_edge(
        graph_facade.Edge(
            type="loop_linkage",
            source=graph_facade.EdgeConnection(node_id="for", field="loop_linkage"),
            destination=graph_facade.EdgeConnection(node_id="return", field="loop_linkage"),
        )
    )
    source_graph.add_edge(
        graph_facade.Edge(
            source=graph_facade.EdgeConnection(node_id="for", field="item"),
            destination=graph_facade.EdgeConnection(node_id="return", field="output"),
        )
    )

    restored_graph = TypeAdapter(graph_facade.Graph).validate_json(source_graph.model_dump_json())
    assert isinstance(restored_graph.nodes["if"], IfInvocation)
    assert isinstance(restored_graph.nodes["iterate"], graph_facade.IterateInvocation)
    assert isinstance(restored_graph.nodes["collect"], graph_facade.CollectInvocation)
    assert isinstance(restored_graph.nodes["for"], ForInvocation)
    assert isinstance(restored_graph.nodes["return"], ForReturnInvocation)
    assert restored_graph.edges[0].type == "loop_linkage"

    state_graph = graph_facade.Graph()
    state_graph.add_node(nodes["for"])
    state_graph.add_node(nodes["return"])
    state_graph.add_edge(source_graph.edges[0])
    state_graph.add_edge(source_graph.edges[1])
    state = graph_facade.GraphExecutionState(
        graph=state_graph,
        execution_graph=state_graph.model_copy(deep=True),
    )
    restored_state = graph_facade.GraphExecutionState.model_validate_json(state.model_dump_json(exclude_none=True))
    assert isinstance(restored_state.graph.nodes["for"], ForInvocation)
    assert isinstance(restored_state.execution_graph.nodes["return"], ForReturnInvocation)


def test_graph_models_preserve_lazy_networkx_and_registry_behavior() -> None:
    script = """
    import builtins
    import sys

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "networkx" or name.startswith("networkx."):
            raise ModuleNotFoundError("No module named 'networkx'")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = blocked_import
    from invokeai.app.invocations.baseinvocation import InvocationRegistry

    import invokeai.app.services.shared.graph_models  # noqa: F401

    from invokeai.app.services.shared.graph import *  # noqa: F401 F403

    registered_classes = set(InvocationRegistry.get_invocation_classes())
    assert any(cls.__name__ == "AddInvocation" and cls.get_type() == "add" for cls in registered_classes)
    assert any(cls.__name__ == "IfInvocation" and cls.get_type() == "if" for cls in registered_classes)
    assert {"iterate", "collect"} <= {cls.get_type() for cls in registered_classes}
    registered_outputs = set(InvocationRegistry.get_output_classes())
    assert {"iterate_output", "collect_output"} <= {cls.get_type() for cls in registered_outputs}

    assert "networkx" not in sys.modules
    print("GRAPH_MODELS_COMPATIBLE")
    """

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "GRAPH_MODELS_COMPATIBLE" in result.stdout
