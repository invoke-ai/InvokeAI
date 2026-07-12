from tests.app.services import workflow_call_test_utils as workflow_call_tests


def test_run_node_propagates_keyboard_interrupt(monkeypatch) -> None:
    workflow_call_tests.test_run_node_propagates_keyboard_interrupt(monkeypatch)


def test_run_node_does_not_swallow_sigint_in_subprocess() -> None:
    workflow_call_tests.test_run_node_does_not_swallow_sigint_in_subprocess()


def test_on_after_run_session_does_not_complete_incomplete_session(monkeypatch) -> None:
    workflow_call_tests.test_on_after_run_session_does_not_complete_incomplete_session(monkeypatch)
