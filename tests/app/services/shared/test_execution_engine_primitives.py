from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from invokeai.app.services.shared.execution_engine.primitives import (
    ActivationGate,
    ContinuationRecord,
    ExecutionFrame,
    StreamBuffer,
    StreamData,
    StreamEnd,
)


def _frame(path: tuple[int | str, ...] = ()) -> ExecutionFrame:
    return ExecutionFrame(
        state_id="state",
        frame_id=f"state:{','.join(str(part) for part in path)}",
        iteration_path=path,
        workflow_call_depth=0,
    )


def test_execution_frame_has_stable_full_identity() -> None:
    frame = _frame((2, "inner"))
    same_frame = ExecutionFrame.model_validate_json(frame.model_dump_json())

    assert frame == same_frame
    assert frame.identity == ("state", "state:2,inner", (2, "inner"), 0)
    assert frame.execution_id == "state"
    assert frame.path == (2, "inner")

    with pytest.raises(ValidationError):
        ExecutionFrame(state_id="", frame_id="frame")


def test_activation_gate_resolves_once_per_owner_and_frame() -> None:
    gate = ActivationGate(
        gate_id="gate",
        owner_id="if-node",
        frame=_frame((0,)),
        branches=("true", "false"),
    )

    assert gate.resolve("true") is True
    assert gate.resolve("true") is False
    assert gate.selected_branch == "true"
    assert gate.is_active("true", owner_id="if-node", frame=_frame((0,)))
    assert not gate.is_active("false", owner_id="if-node", frame=_frame((0,)))

    with pytest.raises(ValueError, match="already resolved"):
        gate.resolve("false")
    with pytest.raises(ValueError, match="owner"):
        gate.is_active("true", owner_id="other", frame=_frame((0,)))
    with pytest.raises(ValueError, match="frame"):
        gate.is_active("true", owner_id="if-node", frame=_frame((1,)))


def test_activation_gate_rejects_unknown_branch_without_mutation() -> None:
    gate = ActivationGate(
        gate_id="gate",
        owner_id="if-node",
        frame=_frame(),
        branches=("true", "false"),
    )

    with pytest.raises(ValueError, match="branch"):
        gate.resolve("other")
    assert gate.selected_branch is None
    assert gate.status == "pending"


def test_stream_buffer_accepts_empty_stream() -> None:
    buffer = StreamBuffer[int](stream_id="stream", owner_id="iterate", frame=_frame())

    assert buffer.accept(StreamEnd(sequence=0)) is True
    assert buffer.accept(StreamEnd(sequence=0)) is False
    assert buffer.closed
    assert buffer.values == ()
    assert buffer.expected_sequence == 0


@pytest.mark.parametrize("discriminator", ["token_kind", "type"])
def test_stream_buffer_normalizes_legacy_event_discriminator_before_validation(discriminator: str) -> None:
    buffer = StreamBuffer[int](stream_id="stream", owner_id="iterate", frame=_frame())

    assert buffer.accept({discriminator: "data", "sequence": 0, "value": 7}) is True
    assert buffer.values == (7,)


def test_stream_buffer_accepts_legacy_type_stream_end_event() -> None:
    buffer = StreamBuffer[int](stream_id="stream", owner_id="iterate", frame=_frame())

    assert buffer.accept({"type": "stream_end", "sequence": 0}) is True
    assert buffer.closed


def test_stream_buffer_rejects_conflicting_event_discriminator_aliases_without_mutation() -> None:
    buffer = StreamBuffer[int](stream_id="stream", owner_id="iterate", frame=_frame())

    with pytest.raises(ValueError, match="conflicting stream event kinds"):
        buffer.accept({"kind": "data", "type": "stream_end", "sequence": 0, "value": 7})

    assert buffer.values == ()
    assert not buffer.closed


def test_stream_buffer_preserves_order_and_none_values() -> None:
    buffer = StreamBuffer[int | None](stream_id="stream", owner_id="iterate", frame=_frame((1,)))

    assert buffer.append_data(3) is True
    assert buffer.append_data(None) is True
    assert buffer.close() is True

    assert buffer.values == (3, None)
    assert [event.sequence for event in buffer.events] == [0, 1, 2]
    assert isinstance(buffer.events[-1], StreamEnd)


@pytest.mark.parametrize("value", [[], {}, {"nested": []}, None])
def test_stream_data_keeps_json_values_by_reference(value: Any) -> None:
    event = StreamData(sequence=0, value=value)
    buffer = StreamBuffer[Any](stream_id="stream", owner_id="iterate", frame=_frame())
    buffer.accept(event)

    assert event.value is value
    assert buffer.events[0].value is value


def test_stream_buffer_rejects_conflicting_duplicate_and_out_of_order_events() -> None:
    buffer = StreamBuffer[int](stream_id="stream", owner_id="iterate", frame=_frame())
    buffer.accept(StreamData(sequence=0, value=1))

    assert buffer.accept(StreamData(sequence=0, value=1)) is False
    with pytest.raises(ValueError, match="duplicate"):
        buffer.accept(StreamData(sequence=0, value=2))
    with pytest.raises(ValueError, match="order"):
        buffer.accept(StreamData(sequence=2, value=3))
    assert buffer.values == (1,)
    assert not buffer.closed


def test_stream_buffer_rejects_data_after_close_without_mutation() -> None:
    buffer = StreamBuffer[int](stream_id="stream", owner_id="iterate", frame=_frame())
    buffer.close()

    with pytest.raises(ValueError, match="closed"):
        buffer.append_data(1)
    assert buffer.values == ()
    assert buffer.events == [StreamEnd(sequence=0)]


def test_stream_buffer_validates_event_json_and_round_trips() -> None:
    buffer = StreamBuffer[dict[str, int]](stream_id="stream", owner_id="iterate", frame=_frame())
    buffer.append_data({"value": 1})
    buffer.close()

    restored = StreamBuffer[dict[str, int]].model_validate_json(buffer.model_dump_json())
    assert restored == buffer

    with pytest.raises(ValidationError):
        StreamData(sequence=0, value=object())


def test_stream_buffer_trusted_rehydration_skips_only_value_serialization() -> None:
    value = object()
    event = StreamData.model_construct(kind="data", sequence=0, value=value)
    buffer = StreamBuffer[Any](stream_id="stream", owner_id="iterate", frame=_frame())

    assert buffer.accept(event, trusted=True) is True
    assert buffer.events[0].value is value

    with pytest.raises(ValueError, match="JSON-serializable"):
        StreamBuffer[Any](stream_id="other", owner_id="iterate", frame=_frame()).accept("data", value=value)


def test_continuation_record_has_atomic_idempotent_status_transitions() -> None:
    continuation = ContinuationRecord[dict[str, int]](
        continuation_id="continuation",
        owner_id="owner",
        frame=_frame((2,)),
        kind="child",
        payload={"input": 1},
    )

    assert continuation.start() is True
    assert continuation.start() is False
    assert continuation.complete({"output": 2}) is True
    assert continuation.complete({"output": 2}) is False
    assert continuation.status == "completed"
    assert continuation.result == {"output": 2}

    with pytest.raises(ValueError, match="terminal"):
        continuation.fail("late failure")
    assert continuation.status == "completed"
    assert continuation.error is None


@pytest.mark.parametrize(
    ("terminal_method", "value", "conflicting_value", "error"),
    [
        ("complete", {"output": 2}, {"output": 3}, "completed continuation has conflicting result"),
        ("fail", "first failure", "different failure", "failed continuation has conflicting error"),
        ("cancel", "user requested", "shutdown", "cancelled continuation has conflicting error"),
    ],
)
def test_continuation_record_terminal_replay_rejects_conflicts_without_mutation(
    terminal_method: str,
    value: object,
    conflicting_value: object,
    error: str,
) -> None:
    continuation = ContinuationRecord(
        continuation_id="continuation",
        owner_id="owner",
        frame=_frame(),
        kind="child",
    )
    continuation.start()
    transition = getattr(continuation, terminal_method)

    assert transition(value) is True
    terminal_state = continuation.model_dump(mode="python")

    assert transition(value) is False
    assert continuation.model_dump(mode="python") == terminal_state

    with pytest.raises(ValueError, match=error):
        transition(conflicting_value)
    assert continuation.model_dump(mode="python") == terminal_state


def test_continuation_record_rejects_invalid_transition_without_mutation() -> None:
    continuation = ContinuationRecord(
        continuation_id="continuation",
        owner_id="owner",
        frame=_frame(),
        kind="child",
    )

    with pytest.raises(ValueError, match="transition"):
        continuation.complete("too early")
    assert continuation.status == "pending"
    assert continuation.result is None

    with pytest.raises(ValueError, match="owner"):
        continuation.assert_scope("other", _frame())
    assert continuation.status == "pending"


def test_continuation_record_json_round_trip_preserves_none_result() -> None:
    continuation = ContinuationRecord(
        continuation_id="continuation",
        owner_id="owner",
        frame=_frame(),
        kind="loop",
    )
    continuation.start()
    continuation.complete(None)

    restored = ContinuationRecord.model_validate_json(continuation.model_dump_json())
    assert restored == continuation
    assert restored.status == "completed"
    assert restored.result is None
