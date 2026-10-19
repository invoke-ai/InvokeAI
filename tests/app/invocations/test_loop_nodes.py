import pytest

from invokeai.app.invocations.fields import OutputScope
from invokeai.app.invocations.loops import (
    ForInvocation,
    ForReturnInvocation,
    ForReturnInvocationOutput,
    LoopState,
    LoopStateValueOutput,
    StateEmptyInvocation,
    StateGetInvocation,
    StateMergeInvocation,
    StateSetInvocation,
)
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.shared.graph import get_output_field_scope


def test_loop_state_defaults_to_empty_values() -> None:
    assert LoopState().values == {}


def test_for_invocation_outputs_have_iteration_and_final_scopes() -> None:
    node = ForInvocation(id="for")

    assert get_output_field_scope(node, "item") == OutputScope.Iteration
    assert get_output_field_scope(node, "index") == OutputScope.Iteration
    assert get_output_field_scope(node, "total") == OutputScope.Iteration
    assert get_output_field_scope(node, "state") == OutputScope.Iteration
    assert get_output_field_scope(node, "output_collection") == OutputScope.Final
    assert get_output_field_scope(node, "final_state") == OutputScope.Final


def test_for_invocation_is_not_directly_executable() -> None:
    node = ForInvocation(id="for")

    with pytest.raises(NotImplementedError, match="must be executed as part of a workflow graph"):
        node.invoke(None)  # type: ignore[arg-type]


def test_for_return_loop_linkage_is_the_first_input() -> None:
    input_names = [
        name for name in ForReturnInvocation.model_fields if name not in {"id", "is_intermediate", "use_cache", "type"}
    ]
    assert input_names == ["loop_linkage", "output", "state", "continue_condition"]


def test_for_return_scheduler_outputs_are_hidden_from_ui_schema() -> None:
    schema = ForReturnInvocationOutput.model_json_schema()
    invocation_schema = ForReturnInvocation.model_json_schema()

    assert schema["properties"]["output"]["ui_hidden"] is True
    assert schema["properties"]["state"]["ui_hidden"] is True
    assert invocation_schema["properties"]["output"].get("ui_hidden", False) is False
    assert invocation_schema["properties"]["state"].get("ui_hidden", False) is False


def test_state_set_invocation_schema_exposes_any_value_input() -> None:
    schema = StateSetInvocation.model_json_schema()

    assert schema["properties"]["value"]["ui_type"] == "AnyField"


def test_state_get_invocation_schema_exposes_any_default_input() -> None:
    schema = StateGetInvocation.model_json_schema()

    assert schema["properties"]["default"]["ui_type"] == "AnyField"


def test_state_get_output_schema_exposes_any_value_output() -> None:
    schema = LoopStateValueOutput.model_json_schema()

    assert schema["properties"]["value"]["ui_type"] == "AnyField"


def test_state_get_output_schema_describes_configured_default() -> None:
    schema = LoopStateValueOutput.model_json_schema()

    assert StateGetInvocation.UIConfig.version == "1.0.3"
    assert "configured default" in schema["properties"]["value"]["description"]


def test_state_merge_invocation_schema_exposes_any_values_input() -> None:
    schema = StateMergeInvocation.model_json_schema()

    assert schema["properties"]["values"]["ui_type"] == "AnyField"


def test_for_return_invocation_returns_body_output_and_state() -> None:
    state = LoopState(values={"count": 1})
    node = ForReturnInvocation(id="return", output="value", state=state)

    output = node.invoke(None)  # type: ignore[arg-type]

    assert output.output == "value"
    assert output.state == state


def test_state_empty_invocation_returns_empty_loop_state() -> None:
    node = StateEmptyInvocation(id="state_empty")

    output = node.invoke(None)  # type: ignore[arg-type]

    assert output.state == LoopState()


def test_state_get_invocation_returns_value_for_key() -> None:
    state = LoopState(values={"count": 2})
    node = StateGetInvocation(id="state_get", state=state, key="count")

    output = node.invoke(None)  # type: ignore[arg-type]

    assert output.value == 2


def test_state_get_invocation_returns_default_for_missing_key() -> None:
    state = LoopState(values={"count": 2})
    node = StateGetInvocation(id="state_get", state=state, key="missing", default="fallback")

    output = node.invoke(None)  # type: ignore[arg-type]

    assert output.value == "fallback"


def test_state_get_invocation_deep_copies_model_values() -> None:
    nested_state = LoopState(values={"items": ["alpha"]})
    state = LoopState(values={"nested": nested_state})
    node = StateGetInvocation(id="state_get", state=state, key="nested")

    output = node.invoke(None)  # type: ignore[arg-type]
    assert isinstance(output.value, LoopState)
    output.value.values["items"].append("beta")

    assert nested_state == LoopState(values={"items": ["alpha"]})


def test_state_set_invocation_returns_new_state_with_value() -> None:
    state = LoopState(values={"count": 2})
    node = StateSetInvocation(id="state_set", state=state, key="count", value=3)

    output = node.invoke(None)  # type: ignore[arg-type]

    assert output.state == LoopState(values={"count": 3})
    assert state == LoopState(values={"count": 2})


def test_state_set_invocation_defaults_to_empty_input_state() -> None:
    node = StateSetInvocation(id="state_set", key="count", value=1)

    output = node.invoke(None)  # type: ignore[arg-type]

    assert output.state == LoopState(values={"count": 1})


def test_state_merge_invocation_returns_new_state_with_updates() -> None:
    state = LoopState(values={"count": 2, "name": "old"})
    node = StateMergeInvocation(id="state_merge", state=state, values={"name": "new", "done": True})

    output = node.invoke(None)  # type: ignore[arg-type]

    assert output.state == LoopState(values={"count": 2, "name": "new", "done": True})
    assert state == LoopState(values={"count": 2, "name": "old"})


def test_state_merge_invocation_default_values_are_not_shared() -> None:
    first = StateMergeInvocation(id="first")
    second = StateMergeInvocation(id="second")
    first.values["count"] = 1

    assert second.values == {}


def test_loop_body_cache_key_ignores_rematerialized_node_id_when_inputs_match() -> None:
    first = StateGetInvocation(id="state_get_0", state=LoopState(values={"count": 1}), key="count")
    second = StateGetInvocation(id="state_get_1", state=LoopState(values={"count": 1}), key="count")

    assert MemoryInvocationCache.create_key(first) == MemoryInvocationCache.create_key(second)


def test_loop_body_cache_key_includes_loop_state_input() -> None:
    first = StateGetInvocation(id="state_get_0", state=LoopState(), key="last_item", default=None)
    second = StateGetInvocation(
        id="state_get_1", state=LoopState(values={"last_item": "alpha"}), key="last_item", default=None
    )

    assert MemoryInvocationCache.create_key(first) != MemoryInvocationCache.create_key(second)


def test_loop_body_cache_key_includes_loop_item_input() -> None:
    first = StateSetInvocation(id="state_set_0", state=LoopState(), key="last_item", value="alpha")
    second = StateSetInvocation(id="state_set_1", state=LoopState(), key="last_item", value="beta")

    assert MemoryInvocationCache.create_key(first) != MemoryInvocationCache.create_key(second)
