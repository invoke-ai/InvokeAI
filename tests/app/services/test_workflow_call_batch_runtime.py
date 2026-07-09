from tests.app.services import workflow_call_test_utils as workflow_call_tests


def test_run_node_fails_cleanly_for_invalid_batch_child_workflow(monkeypatch) -> None:
    workflow_call_tests.test_run_node_fails_cleanly_for_invalid_batch_child_workflow(monkeypatch)


def test_run_completes_call_saved_workflow_with_batched_child_returns(monkeypatch) -> None:
    workflow_call_tests.test_run_completes_call_saved_workflow_with_batched_child_returns(monkeypatch)


def test_run_zips_grouped_batch_children(monkeypatch) -> None:
    workflow_call_tests.test_run_zips_grouped_batch_children(monkeypatch)


def test_run_expands_ungrouped_batch_children_as_cartesian_product(monkeypatch) -> None:
    workflow_call_tests.test_run_expands_ungrouped_batch_children_as_cartesian_product(monkeypatch)


def test_run_fails_batched_child_workflow_and_cancels_remaining_siblings(monkeypatch) -> None:
    workflow_call_tests.test_run_fails_batched_child_workflow_and_cancels_remaining_siblings(monkeypatch)


def test_run_supports_generator_backed_integer_batched_child_workflow(monkeypatch) -> None:
    workflow_call_tests.test_run_supports_generator_backed_integer_batched_child_workflow(monkeypatch)


def test_run_supports_generator_backed_image_batched_child_workflow(monkeypatch) -> None:
    workflow_call_tests.test_run_supports_generator_backed_image_batched_child_workflow(monkeypatch)


def test_run_rejects_non_generator_connected_batched_child_workflow(monkeypatch) -> None:
    workflow_call_tests.test_run_rejects_non_generator_connected_batched_child_workflow(monkeypatch)
