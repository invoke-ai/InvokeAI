import json

import pytest
from pydantic import ValidationError

from invokeai.app.services.shared.execution_engine.child import (
    ChildCompletion,
    ChildExecutionCapability,
    ChildTerminalStatus,
)


def _capability(**kwargs: object) -> ChildExecutionCapability:
    values: dict[str, object] = {
        "parent_execution_id": "parent",
        "parent_frame": ("outer", 2),
        "parent_reference_id": "parent-ref",
        "authorization_context": {"subject": "owner"},
    }
    values.update(kwargs)
    return ChildExecutionCapability(**values)


def test_capability_binds_child_to_exact_parent_and_frame() -> None:
    capability = _capability()
    child = capability.create_child("child", child_frame=("outer", 2, 0), graph={"nodes": {}}, inputs={})

    assert child.parent_execution_id == "parent"
    assert child.parent_frame == ("outer", 2)
    assert child.child_frame == ("outer", 2, 0)
    assert child.depth == 1
    assert child.capability_id == capability.capability_id

    with pytest.raises(ValueError, match="parent frame"):
        capability.create_child("other", parent_frame=("wrong",))
    with pytest.raises(ValueError, match="parent"):
        capability.create_child("other", parent_execution_id="other")
    with pytest.raises(ValueError, match="child frame"):
        capability.create_child("other", child_frame=("wrong", 0))


def test_capability_rejects_depth_count_and_capacity_limits() -> None:
    with pytest.raises(ValueError, match="depth"):
        _capability(depth=2, max_depth=2).create_child("child")
    with pytest.raises(ValueError, match="count"):
        _capability(max_children=1).create_dependency(["one", "two"])
    with pytest.raises(ValueError, match="capacity"):
        _capability(capacity=1).create_dependency(["one", "two"])


def test_capability_limits_cumulative_child_creation() -> None:
    capability = _capability(max_children=2, capacity=1)

    child = capability.create_child("one")

    assert child.child_execution_id == "one"
    assert capability.remaining_capacity == 0
    with pytest.raises(ValueError, match="capacity"):
        capability.create_child("two")
    assert capability.remaining_capacity == 0


def test_dependency_creation_does_not_leak_capacity_on_invalid_child() -> None:
    capability = _capability(capacity=2)

    with pytest.raises(ValueError, match="child frame"):
        capability.create_dependency(
            ["first", "second"],
            child_frames=[("outer", 2, 0), ("wrong", 1)],
        )

    assert capability.remaining_capacity == 2


@pytest.mark.parametrize(
    ("terminal_method", "expected_status"),
    [("fail_child", "failed"), ("cancel_child", "canceled")],
)
def test_a_terminal_child_report_ends_the_dependency_before_siblings_finish(
    terminal_method: str, expected_status: ChildTerminalStatus
) -> None:
    dependency = _capability().create_dependency(["first", "second", "third"], dependency_id="dependency")
    report = getattr(dependency, terminal_method)

    update = report("second", "child reported")
    later = dependency.complete_child("third", {"value": 3})
    duplicate = report("second", "child reported")

    assert update.status == expected_status
    assert update.terminal is True
    assert dependency.error_message == "child reported"
    assert later.changed is False
    assert later.status == expected_status
    assert duplicate.changed is False
    assert duplicate.terminal is True


def test_dependency_rejects_duplicate_prebuilt_children() -> None:
    capability = _capability()
    child = capability.create_child("child")

    with pytest.raises(ValueError, match="unique"):
        capability.create_dependency(children=[child, child])


def test_dependency_accepts_out_of_order_completion_but_aggregates_enqueue_order() -> None:
    capability = _capability()
    dependency = capability.create_dependency(["first", "middle", "last"], dependency_id="dependency")

    update = dependency.complete_child("last", {"value": "last"})
    assert update.terminal is False
    update = dependency.complete_child("first", {"value": "first"})
    assert update.terminal is False
    update = dependency.complete_child("middle", {"value": "middle"})

    assert update.status == "completed"
    assert update.aggregated_outputs == {"value": ["first", "middle", "last"]}
    assert dependency.completed_child_execution_ids == ["first", "middle", "last"]


def test_dependency_is_idempotent_and_rejects_conflicting_or_foreign_completion() -> None:
    capability = _capability()
    dependency = capability.create_dependency(["child"], dependency_id="dependency")
    completed = dependency.complete_child("child", {"value": 1})
    receipt = dependency.completions["child"]

    duplicate = dependency.record_completion(receipt)
    assert duplicate.changed is False
    assert duplicate.status == "completed"

    with pytest.raises(ValueError, match="different terminal"):
        dependency.record_completion(
            ChildCompletion(
                child_execution_id="child",
                parent_execution_id="parent",
                parent_frame=("outer", 2),
                parent_reference_id="parent-ref",
                child_frame=("outer", 2, "child"),
                status="failed",
                error_message="late",
            )
        )
    assert completed.terminal is True

    with pytest.raises(ValueError, match="does not belong"):
        dependency.complete_child("other", {"value": 2})


def test_child_records_are_json_safe_and_round_trip() -> None:
    dependency = _capability().create_dependency(["child"], dependency_id="dependency")
    dependency.complete_child("child", {"value": None})

    raw = dependency.model_dump_json()
    restored = type(dependency).model_validate_json(raw)

    assert json.loads(raw)["children"][0]["capability_id"] == dependency.children[0].capability_id
    assert restored == dependency

    with pytest.raises(ValidationError, match="JSON-serializable"):
        _capability(authorization_context={"runtime": object()})


def test_dependency_rejects_incomplete_completed_snapshot() -> None:
    dependency = _capability().create_dependency(["first", "second"], dependency_id="dependency")
    snapshot = dependency.model_dump(mode="python")
    snapshot.update(status="completed", terminal_transition_count=1)

    with pytest.raises(ValidationError, match="completed dependency"):
        type(dependency).model_validate(snapshot)


def test_failed_child_requires_a_nonblank_error_message() -> None:
    dependency = _capability().create_dependency(["child"], dependency_id="dependency")

    with pytest.raises(ValidationError, match="error"):
        dependency.fail_child("child", "")


@pytest.mark.parametrize("status", ["completed", "failed", "canceled"])
def test_terminal_statuses_are_explicit(status: ChildTerminalStatus) -> None:
    assert status in {"completed", "failed", "canceled"}
