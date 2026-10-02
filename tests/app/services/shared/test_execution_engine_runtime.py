from invokeai.app.services.shared.execution_engine.primitives import ExecutionFrame
from invokeai.app.services.shared.execution_engine.runtime import ExecutionEngineRuntime


def _frame(path: tuple[int | str, ...] = ()) -> ExecutionFrame:
    return ExecutionFrame(
        state_id="state",
        frame_id=f"state:{path}",
        iteration_path=path,
        workflow_call_depth=0,
    )


def test_runtime_keeps_gate_stream_and_continuation_scoped_by_full_frame() -> None:
    runtime = ExecutionEngineRuntime()
    root = _frame()
    child = _frame((0,))

    root_gate = runtime.register_gate("if", "if", root, ("true", "false"))
    child_gate = runtime.register_gate("if:child", "if", child, ("true", "false"))
    assert root_gate is not child_gate
    runtime.resolve_gate("if", "if", root, "true")
    assert runtime.get_gate("if").selected_branch == "true"
    assert runtime.get_gate("if:child").selected_branch is None

    stream = runtime.get_or_create_stream("iterate", "iterate", child)
    stream.append_data("value")
    stream.close()
    assert runtime.get_stream("iterate").values == ("value",)

    continuation = runtime.register_continuation("loop", "for", child, "loop")
    continuation.start()
    continuation.complete({"value": 1})
    assert runtime.get_continuation("loop").result == {"value": 1}


def test_runtime_snapshot_round_trip_preserves_durable_records() -> None:
    runtime = ExecutionEngineRuntime()
    frame = _frame((2, "inner"))
    runtime.register_gate("gate", "if", frame, ("true", "false"))
    runtime.resolve_gate("gate", "if", frame, "false")
    stream = runtime.get_or_create_stream("stream", "iterate", frame)
    stream.append_data(None)
    stream.close()
    continuation = runtime.register_continuation("continuation", "for", frame, "loop")
    continuation.start()

    restored = ExecutionEngineRuntime.model_validate(runtime.model_dump(mode="json"))

    assert restored == runtime
    assert restored.get_gate("gate").is_active("false")
    assert restored.get_stream("stream").values == (None,)
    assert restored.get_continuation("continuation").status == "running"
