from tests.app.services import workflow_call_test_utils as workflow_call_tests


def test_run_node_enters_waiting_state_without_executing_child_inline(monkeypatch) -> None:
    workflow_call_tests.test_run_node_enters_waiting_state_without_executing_child_inline(monkeypatch)


def test_run_persists_waiting_session_without_completing_queue_item(monkeypatch) -> None:
    workflow_call_tests.test_run_persists_waiting_session_without_completing_queue_item(monkeypatch)


def test_workflow_call_coordinator_suspends_parent_and_enqueues_child_queue_item(monkeypatch) -> None:
    workflow_call_tests.test_workflow_call_coordinator_suspends_parent_and_enqueues_child_queue_item(monkeypatch)


def test_workflow_call_queue_lifecycle_leaves_non_call_workflows_on_normal_execution_path(monkeypatch) -> None:
    workflow_call_tests.test_workflow_call_queue_lifecycle_leaves_non_call_workflows_on_normal_execution_path(
        monkeypatch
    )


def test_workflow_call_queue_lifecycle_resumes_parent_from_completed_child(monkeypatch) -> None:
    workflow_call_tests.test_workflow_call_queue_lifecycle_resumes_parent_from_completed_child(monkeypatch)


def test_run_preserves_canceled_child_workflow_chain_without_failing_parent(monkeypatch) -> None:
    workflow_call_tests.test_run_preserves_canceled_child_workflow_chain_without_failing_parent(monkeypatch)


def test_workflow_call_coordinator_builds_child_queue_item_with_relationship_metadata(monkeypatch) -> None:
    workflow_call_tests.test_workflow_call_coordinator_builds_child_queue_item_with_relationship_metadata(monkeypatch)


def test_workflow_call_coordinator_cleans_up_enqueued_children_when_boundary_setup_fails(monkeypatch) -> None:
    workflow_call_tests.test_workflow_call_coordinator_cleans_up_enqueued_children_when_boundary_setup_fails(
        monkeypatch
    )


def test_workflow_call_coordinator_rejects_child_expansion_that_exceeds_remaining_queue_capacity(monkeypatch) -> None:
    workflow_call_tests.test_workflow_call_coordinator_rejects_child_expansion_that_exceeds_remaining_queue_capacity(
        monkeypatch
    )


def test_run_completes_call_saved_workflow_and_runs_downstream_nodes(monkeypatch) -> None:
    workflow_call_tests.test_run_completes_call_saved_workflow_and_runs_downstream_nodes(monkeypatch)


def test_run_node_records_child_execution_state_for_call_saved_workflow(monkeypatch) -> None:
    workflow_call_tests.test_run_node_records_child_execution_state_for_call_saved_workflow(monkeypatch)


def test_run_executes_child_workflow_and_completes_parent_queue_item(monkeypatch) -> None:
    workflow_call_tests.test_run_executes_child_workflow_and_completes_parent_queue_item(monkeypatch)


def test_run_completes_call_saved_workflow_with_child_return_collection(monkeypatch) -> None:
    workflow_call_tests.test_run_completes_call_saved_workflow_with_child_return_collection(monkeypatch)


def test_run_fails_call_saved_workflow_when_child_has_no_workflow_return(monkeypatch) -> None:
    workflow_call_tests.test_run_fails_call_saved_workflow_when_child_has_no_workflow_return(monkeypatch)


def test_run_respects_child_dependency_readiness(monkeypatch) -> None:
    workflow_call_tests.test_run_respects_child_dependency_readiness(monkeypatch)


def test_run_respects_child_if_branching(monkeypatch) -> None:
    workflow_call_tests.test_run_respects_child_if_branching(monkeypatch)


def test_run_supports_nested_call_saved_workflow_execution(monkeypatch) -> None:
    workflow_call_tests.test_run_supports_nested_call_saved_workflow_execution(monkeypatch)


def test_run_cascades_nested_child_workflow_failures_to_all_parents(monkeypatch) -> None:
    workflow_call_tests.test_run_cascades_nested_child_workflow_failures_to_all_parents(monkeypatch)


def test_run_forwards_literal_dynamic_workflow_inputs_to_child_workflow(monkeypatch) -> None:
    workflow_call_tests.test_run_forwards_literal_dynamic_workflow_inputs_to_child_workflow(monkeypatch)


def test_run_forwards_connected_dynamic_workflow_inputs_to_child_workflow(monkeypatch) -> None:
    workflow_call_tests.test_run_forwards_connected_dynamic_workflow_inputs_to_child_workflow(monkeypatch)


def test_run_rejects_non_exposed_dynamic_workflow_inputs(monkeypatch) -> None:
    workflow_call_tests.test_run_rejects_non_exposed_dynamic_workflow_inputs(monkeypatch)


def test_run_fails_call_saved_workflow_when_child_workflow_graph_cannot_be_built(monkeypatch) -> None:
    workflow_call_tests.test_run_fails_call_saved_workflow_when_child_workflow_graph_cannot_be_built(monkeypatch)


def test_run_fails_call_saved_workflow_with_invalid_selection_without_entering_waiting_state(monkeypatch) -> None:
    workflow_call_tests.test_run_fails_call_saved_workflow_with_invalid_selection_without_entering_waiting_state(
        monkeypatch
    )


def test_run_fails_call_saved_workflow_when_depth_limit_is_exceeded(monkeypatch) -> None:
    workflow_call_tests.test_run_fails_call_saved_workflow_when_depth_limit_is_exceeded(monkeypatch)
