from __future__ import annotations

from typing import TYPE_CHECKING, Any

from invokeai.app.invocations.call_saved_workflow import (
    CallSavedWorkflowInvocation,
    is_call_saved_workflow_dynamic_input,
)
from invokeai.app.invocations.workflow_return import WorkflowReturnOutput
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.shared.graph import GraphExecutionState
from invokeai.app.services.shared.workflow_graph_builder import (
    apply_workflow_inputs_to_graph,
    build_graph_from_workflow,
)

if TYPE_CHECKING:
    from invokeai.app.services.session_processor.session_processor_default import DefaultSessionRunner


class WorkflowCallCoordinator:
    """Coordinates call-specific workflow setup."""

    def __init__(self, session_runner: DefaultSessionRunner) -> None:
        self._session_runner = session_runner

    def _collect_call_saved_workflow_inputs(
        self, invocation: CallSavedWorkflowInvocation, queue_item: SessionQueueItem
    ) -> dict[str, Any]:
        workflow_inputs = dict(invocation.workflow_inputs)
        for edge in queue_item.session.execution_graph._get_input_edges(invocation.id):
            if not is_call_saved_workflow_dynamic_input(edge.destination.field):
                continue
            if edge.source.node_id not in queue_item.session.results:
                continue
            workflow_inputs[edge.destination.field] = getattr(
                queue_item.session.results[edge.source.node_id], edge.source.field
            )
        return workflow_inputs

    @staticmethod
    def build_child_queue_item(queue_item: SessionQueueItem, child_session: GraphExecutionState) -> SessionQueueItem:
        workflow_call_execution = queue_item.session.waiting_workflow_call_execution
        if workflow_call_execution is None:
            raise ValueError("Parent queue item is missing active workflow call execution metadata.")
        root_item_id = getattr(queue_item, "root_item_id", None) or queue_item.item_id
        child_updates = {
            "session": child_session,
            "session_id": child_session.id,
            "workflow_call_id": workflow_call_execution.id,
            "parent_item_id": queue_item.item_id,
            "parent_session_id": queue_item.session_id,
            "root_item_id": root_item_id,
            "workflow_call_depth": workflow_call_execution.depth,
        }
        if hasattr(queue_item, "model_copy"):
            return queue_item.model_copy(update=child_updates)

        child_queue_item = type(queue_item).__new__(type(queue_item))
        child_queue_item.__dict__ = {**queue_item.__dict__, **child_updates}
        return child_queue_item

    def begin_workflow_call_boundary(
        self,
        invocation: CallSavedWorkflowInvocation,
        queue_item: SessionQueueItem,
        workflow_record,
    ) -> SessionQueueItem:
        call_frame = queue_item.session.build_workflow_call_frame(invocation.id, invocation.workflow_id)
        child_graph = build_graph_from_workflow(workflow_record.workflow.model_dump())
        apply_workflow_inputs_to_graph(
            child_graph,
            workflow_record.workflow.model_dump(),
            self._collect_call_saved_workflow_inputs(invocation, queue_item),
        )
        child_session = queue_item.session.create_child_workflow_execution_state(child_graph, call_frame)
        queue_item.session.begin_waiting_on_workflow_call(call_frame)
        queue_item.session.attach_waiting_workflow_call_child_session(child_session)
        self._session_runner._services.session_queue.set_queue_item_session(queue_item.item_id, queue_item.session)
        child_queue_item = self._session_runner._services.session_queue.enqueue_workflow_call_child(
            parent_queue_item=queue_item,
            child_session=child_session,
        )
        self._session_runner._services.session_queue.suspend_queue_item(queue_item.item_id)
        queue_item.status = "waiting"
        return child_queue_item


class WorkflowCallQueueLifecycle:
    """Coordinates queue-visible child workflow execution and parent lifecycle transitions."""

    def __init__(self, session_runner: DefaultSessionRunner) -> None:
        self._session_runner = session_runner

    @staticmethod
    def get_waiting_workflow_call_invocation(queue_item: SessionQueueItem) -> CallSavedWorkflowInvocation:
        waiting_frame = queue_item.session.waiting_workflow_call
        if waiting_frame is None:
            raise ValueError("Execution state is not waiting on a workflow call.")
        invocation = queue_item.session.execution_graph.nodes.get(waiting_frame.prepared_call_node_id)
        if not isinstance(invocation, CallSavedWorkflowInvocation):
            raise ValueError("Waiting workflow call frame does not point to a call_saved_workflow invocation.")
        return invocation

    @staticmethod
    def get_child_workflow_return_output(child_session: GraphExecutionState) -> WorkflowReturnOutput:
        workflow_return_node_ids = [
            node_id for node_id, node in child_session.graph.nodes.items() if node.get_type() == "workflow_return"
        ]
        if not workflow_return_node_ids:
            raise ValueError("The selected saved workflow must contain exactly one workflow_return node.")
        if len(workflow_return_node_ids) > 1:
            raise ValueError("The selected saved workflow must not contain more than one workflow_return node.")

        workflow_return_node_id = workflow_return_node_ids[0]
        prepared_return_node_ids = child_session.source_prepared_mapping.get(workflow_return_node_id, set())
        if len(prepared_return_node_ids) != 1:
            raise ValueError(
                "The selected saved workflow produced an unsupported number of workflow_return executions."
            )

        prepared_return_node_id = next(iter(prepared_return_node_ids))
        output = child_session.results.get(prepared_return_node_id)
        if not isinstance(output, WorkflowReturnOutput):
            raise ValueError("The selected saved workflow did not produce a valid workflow_return output.")

        return output

    def resume_waiting_workflow_call(self, queue_item: SessionQueueItem) -> None:
        invocation = self.get_waiting_workflow_call_invocation(queue_item)
        child_session = queue_item.session.waiting_workflow_call_child_session
        if child_session is None:
            raise ValueError("Execution state is waiting on a workflow call but has no attached child session.")
        output = self.get_child_workflow_return_output(child_session)
        queue_item.session.end_waiting_on_workflow_call(status="completed")
        queue_item.session.complete(invocation.id, output)
        self._session_runner._on_after_run_node(invocation, queue_item, output)

    def fail_waiting_workflow_call(self, queue_item: SessionQueueItem, error_message: str) -> None:
        invocation = self.get_waiting_workflow_call_invocation(queue_item)
        queue_item.session.end_waiting_on_workflow_call(status="failed", error_message=error_message)
        self._session_runner._on_node_error(
            invocation=invocation,
            queue_item=queue_item,
            error_type="ValueError",
            error_message=error_message,
            error_traceback=error_message,
        )

    def _get_parent_queue_item(self, child_queue_item: SessionQueueItem) -> SessionQueueItem:
        parent_item_id = child_queue_item.parent_item_id
        if parent_item_id is None:
            raise ValueError("Child workflow queue item is missing parent_item_id metadata.")
        return self._session_runner._services.session_queue.get_queue_item(parent_item_id)

    def _resume_parent_from_completed_child(self, child_queue_item: SessionQueueItem) -> None:
        parent_queue_item = self._get_parent_queue_item(child_queue_item)
        parent_queue_item.session.waiting_workflow_call_child_session = child_queue_item.session
        try:
            self.resume_waiting_workflow_call(parent_queue_item)
        except Exception as e:
            self.fail_waiting_workflow_call(parent_queue_item, str(e))
            parent_queue_item = self._session_runner._services.session_queue.get_queue_item(parent_queue_item.item_id)
            if getattr(parent_queue_item, "parent_item_id", None) is not None:
                self._fail_parent_from_failed_child(parent_queue_item)
            return
        parent_queue_item = self._session_runner._services.session_queue.set_queue_item_session(
            parent_queue_item.item_id, parent_queue_item.session
        )
        if parent_queue_item.session.is_complete():
            parent_queue_item = self._session_runner._services.session_queue.complete_queue_item(
                parent_queue_item.item_id
            )
            if getattr(parent_queue_item, "parent_item_id", None) is not None:
                self._resume_parent_from_completed_child(parent_queue_item)
            return
        self._session_runner._services.session_queue.resume_queue_item(parent_queue_item.item_id)

    def _fail_parent_from_failed_child(self, child_queue_item: SessionQueueItem) -> None:
        parent_queue_item = self._get_parent_queue_item(child_queue_item)
        waiting_frame = parent_queue_item.session.waiting_workflow_call
        if waiting_frame is None:
            raise ValueError("Parent queue item is missing workflow call waiting state.")
        child_error_message = getattr(child_queue_item, "error_message", None) or (
            f"The selected saved workflow '{waiting_frame.workflow_id}' failed during child execution."
        )
        self.fail_waiting_workflow_call(parent_queue_item, child_error_message)
        parent_queue_item = self._session_runner._services.session_queue.get_queue_item(parent_queue_item.item_id)
        if getattr(parent_queue_item, "parent_item_id", None) is not None:
            self._fail_parent_from_failed_child(parent_queue_item)

    def run_queue_item(self, queue_item: SessionQueueItem) -> None:
        self._session_runner.run(queue_item)
        updated_queue_item = self._session_runner._services.session_queue.get_queue_item(queue_item.item_id)
        if getattr(updated_queue_item, "parent_item_id", None) is None:
            return
        if updated_queue_item.status == "completed":
            self._resume_parent_from_completed_child(updated_queue_item)
        elif updated_queue_item.status in ["failed", "canceled"]:
            self._fail_parent_from_failed_child(updated_queue_item)
