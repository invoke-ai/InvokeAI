from __future__ import annotations

from typing import TYPE_CHECKING, Any

from invokeai.app.invocations.call_saved_workflow import (
    CallSavedWorkflowInvocation,
    is_call_saved_workflow_dynamic_input,
)
from invokeai.app.invocations.workflow_return import WorkflowReturnOutput
from invokeai.app.services.session_processor.workflow_call_batch import build_child_workflow_session_results
from invokeai.app.services.session_queue.session_queue_common import (
    NodeFieldValue,
    SessionQueueItem,
    SessionQueueItemNotFoundError,
)
from invokeai.app.services.shared.graph import GraphExecutionState

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
    def build_child_queue_item(
        queue_item: SessionQueueItem,
        child_session: GraphExecutionState,
        field_values: list[NodeFieldValue] | None = None,
    ) -> SessionQueueItem:
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
            "field_values": field_values,
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
        *,
        workflow: Any | None = None,
        workflow_inputs: dict[str, Any] | None = None,
        dependency_id: str | None = None,
    ) -> SessionQueueItem:
        queue_status = self._session_runner._services.session_queue.get_queue_status(queue_item.queue_id)
        remaining_queue_capacity = self._session_runner._services.configuration.max_queue_size - queue_status.pending
        if remaining_queue_capacity <= 0:
            raise ValueError("call_saved_workflow exceeds remaining queue capacity for child workflow executions")

        call_frame = queue_item.session.build_workflow_call_frame(invocation.id, invocation.workflow_id)
        if workflow is None:
            workflow = workflow_record.workflow.model_dump()
        if workflow_inputs is None:
            workflow_inputs = self._collect_call_saved_workflow_inputs(invocation, queue_item)
        child_session_results = build_child_workflow_session_results(
            parent_session=queue_item.session,
            workflow=workflow,
            workflow_inputs=workflow_inputs,
            call_frame=call_frame,
            maximum_children=remaining_queue_capacity,
            services=self._session_runner._services,
            user_id=getattr(queue_item, "user_id", None),
        )
        child_sessions = [child_result.session for child_result in child_session_results]
        if len(child_sessions) > remaining_queue_capacity:
            raise ValueError("call_saved_workflow exceeds remaining queue capacity for child workflow executions")
        if dependency_id is not None and (not isinstance(dependency_id, str) or not dependency_id.strip()):
            raise ValueError("Workflow call dependency id must not be blank.")
        queue_item.session.begin_waiting_on_workflow_call(call_frame)
        if dependency_id is not None:
            if queue_item.session.waiting_workflow_call_execution is None:
                raise ValueError("Execution state is waiting on a workflow call but has no execution metadata.")
            queue_item.session.waiting_workflow_call_execution.id = dependency_id
        queue_item.session.attach_waiting_workflow_call_child_sessions(child_sessions)
        enqueued_child_item_ids: list[int] = []
        try:
            child_queue_items = self._session_runner._services.session_queue.enqueue_workflow_call_children(
                parent_queue_item=queue_item,
                child_sessions=[(result.session, result.field_values) for result in child_session_results],
            )
            enqueued_child_item_ids.extend(child_queue_item.item_id for child_queue_item in child_queue_items)
        except Exception as e:
            if enqueued_child_item_ids:
                self._session_runner._services.session_queue.delete_queue_items_by_id(enqueued_child_item_ids)
            queue_item.session.end_waiting_on_workflow_call(status="failed", error_message=str(e))
            raise
        queue_item.status = "waiting"
        if not child_queue_items:
            raise ValueError("Workflow call did not produce any child executions.")
        return child_queue_items[-1]


class WorkflowCallQueueLifecycle:
    """Coordinates queue-visible child workflow execution and parent lifecycle transitions."""

    def __init__(self, session_runner: DefaultSessionRunner) -> None:
        self._session_runner = session_runner

    @staticmethod
    def _effect_value(effect: Any, name: str, default: Any = None) -> Any:
        if isinstance(effect, dict):
            return effect.get(name, default)
        return getattr(effect, name, default)

    @classmethod
    def _effect_kind(cls, effect: Any) -> Any:
        return next(
            (
                cls._effect_value(effect, name)
                for name in ("kind", "effect_type", "type")
                if cls._effect_value(effect, name) is not None
            ),
            None,
        )

    def apply_execution_effects(
        self,
        *,
        invocation: CallSavedWorkflowInvocation,
        queue_item: SessionQueueItem,
        workflow_record,
        effects,
    ) -> None:
        """Apply call lifecycle effects through durable queue workflow-call semantics."""

        lifecycle_effects = [
            effect for effect in effects if self._effect_kind(effect) in {"spawn_execution", "await", "fail"}
        ]
        spawn_effects = [effect for effect in lifecycle_effects if self._effect_kind(effect) == "spawn_execution"]
        await_effects = [effect for effect in lifecycle_effects if self._effect_kind(effect) == "await"]
        fail_effects = [effect for effect in lifecycle_effects if self._effect_kind(effect) == "fail"]

        if fail_effects:
            if len(fail_effects) != 1 or spawn_effects or await_effects:
                raise ValueError(
                    "Workflow call lifecycle effects must contain either one fail or one spawn+await pair."
                )
            message = self._effect_value(fail_effects[0], "message")
            if not isinstance(message, str):
                raise ValueError("Workflow call fail effect requires an error message.")
            raise ValueError(message)

        if len(spawn_effects) != 1 or len(await_effects) != 1:
            raise ValueError("Workflow call lifecycle effects must contain exactly one spawn and one await effect.")

        spawn_effect = spawn_effects[0]
        await_effect = await_effects[0]
        child_execution_id = self._effect_value(spawn_effect, "child_execution_id")
        if not isinstance(child_execution_id, str) or not child_execution_id.strip():
            raise ValueError("Workflow call spawn effect requires a child execution identity.")

        dependency = self._effect_value(await_effect, "dependency")
        awaited_child_id = next(
            (
                self._effect_value(dependency, name)
                for name in ("execution_node_id", "child_execution_id", "execution_id", "node_id", "child_id")
                if self._effect_value(dependency, name) is not None
            ),
            None,
        )
        if awaited_child_id != child_execution_id:
            raise ValueError("Workflow call await effect does not match its spawn effect.")

        workflow = self._effect_value(spawn_effect, "graph")
        if hasattr(workflow, "model_dump"):
            workflow = workflow.model_dump(mode="json")
        if not isinstance(workflow, dict):
            raise ValueError("Workflow call spawn effect requires a workflow graph.")
        workflow_inputs = self._effect_value(spawn_effect, "inputs")
        if not isinstance(workflow_inputs, dict):
            raise ValueError("Workflow call spawn effect requires workflow inputs.")

        self._session_runner.workflow_call_coordinator.begin_workflow_call_boundary(
            invocation,
            queue_item,
            workflow_record,
            workflow=workflow,
            workflow_inputs=dict(workflow_inputs),
            dependency_id=child_execution_id,
        )

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

    @staticmethod
    def _apply_workflow_call_output(
        queue_item: SessionQueueItem, invocation: CallSavedWorkflowInvocation, output: WorkflowReturnOutput
    ) -> None:
        execution_ref = queue_item.session.get_execution_ref(invocation.id)
        queue_item.session.apply(execution_ref, output)

    def resume_waiting_workflow_call(self, queue_item: SessionQueueItem) -> None:
        invocation = self.get_waiting_workflow_call_invocation(queue_item)
        child_session = queue_item.session.waiting_workflow_call_child_session
        if child_session is None:
            raise ValueError("Execution state is waiting on a workflow call but has no attached child session.")
        output = self.get_child_workflow_return_output(child_session)
        queue_item.session.end_waiting_on_workflow_call(status="completed")
        self._apply_workflow_call_output(queue_item, invocation, output)
        self._session_runner._on_after_run_node(invocation, queue_item, output)

    def fail_waiting_workflow_call(
        self,
        queue_item: SessionQueueItem,
        error_message: str,
    ) -> bool:
        invocation = self.get_waiting_workflow_call_invocation(queue_item)
        queue_item.session.end_waiting_on_workflow_call(status="failed", error_message=error_message)
        return bool(
            self._session_runner._on_node_error(
                invocation=invocation,
                queue_item=queue_item,
                error_type="ValueError",
                error_message=error_message,
                error_traceback=error_message,
                require_active=True,
            )
        )

    def _get_parent_queue_item(self, child_queue_item: SessionQueueItem) -> SessionQueueItem | None:
        parent_item_id = child_queue_item.parent_item_id
        if parent_item_id is None:
            raise ValueError("Child workflow queue item is missing parent_item_id metadata.")
        try:
            return self._session_runner._services.session_queue.get_queue_item(parent_item_id)
        except SessionQueueItemNotFoundError:
            # The parent may have been deleted while the child was running (e.g. the queue was cleared).
            return None

    def _resume_parent_from_completed_child(self, child_queue_item: SessionQueueItem) -> None:
        parent_queue_item = self._get_parent_queue_item(child_queue_item)
        if parent_queue_item is None or parent_queue_item.status in ("completed", "failed", "canceled"):
            return
        try:
            output = self.get_child_workflow_return_output(child_queue_item.session)
            completion = self._session_runner._services.session_queue.record_workflow_call_child_completion(
                parent_item_id=parent_queue_item.item_id,
                child_item_id=child_queue_item.item_id,
                output_values=output.values,
            )
            if completion is None:
                return
            parent_queue_item = completion.parent_queue_item
            should_resume_parent = completion.should_resume
            aggregated_values = completion.aggregated_values
        except Exception as e:
            workflow_call_execution = parent_queue_item.session.waiting_workflow_call_execution
            if workflow_call_execution is not None:
                self._session_runner._services.session_queue.cancel_workflow_call_children(
                    workflow_call_execution.id,
                    exclude_item_ids={child_queue_item.item_id},
                )
            self.fail_waiting_workflow_call(parent_queue_item, str(e))
            try:
                parent_queue_item = self._session_runner._services.session_queue.get_queue_item(
                    parent_queue_item.item_id
                )
            except SessionQueueItemNotFoundError:
                return
            if getattr(parent_queue_item, "parent_item_id", None) is not None:
                self._fail_parent_from_failed_child(parent_queue_item)
            return
        if not should_resume_parent:
            # The queue-owned completion transaction already persisted this parent session. Do not
            # write the returned snapshot again: another child may have committed newer state.
            return
        parent_queue_item.session.waiting_workflow_call_child_session = child_queue_item.session
        waiting_invocation = self.get_waiting_workflow_call_invocation(parent_queue_item)
        parent_queue_item.session.end_waiting_on_workflow_call(status="completed")
        parent_output = WorkflowReturnOutput(values=aggregated_values)
        self._apply_workflow_call_output(parent_queue_item, waiting_invocation, parent_output)
        saved = self._session_runner._save_queue_item_session_if_active(parent_queue_item)
        if not saved:
            return
        self._session_runner._on_after_run_node(waiting_invocation, parent_queue_item, parent_output)
        if parent_queue_item.session.is_complete():
            parent_queue_item = self._session_runner._services.session_queue.complete_queue_item(
                parent_queue_item.item_id, queue_item=parent_queue_item
            )
            if parent_queue_item.status != "completed":
                return
            if getattr(parent_queue_item, "parent_item_id", None) is not None:
                self._resume_parent_from_completed_child(parent_queue_item)
            return
        self._session_runner._services.session_queue.resume_queue_item(
            parent_queue_item.item_id, queue_item=parent_queue_item
        )

    def _fail_parent_from_failed_child(self, child_queue_item: SessionQueueItem) -> None:
        parent_queue_item = self._get_parent_queue_item(child_queue_item)
        if parent_queue_item is None or parent_queue_item.status in ("completed", "failed", "canceled"):
            return
        waiting_frame = parent_queue_item.session.waiting_workflow_call
        if waiting_frame is None:
            raise ValueError("Parent queue item is missing workflow call waiting state.")
        child_error_message = getattr(child_queue_item, "error_message", None) or (
            f"The selected saved workflow '{waiting_frame.workflow_id}' failed during child execution."
        )
        workflow_call_execution = parent_queue_item.session.waiting_workflow_call_execution
        generic_update = None
        if workflow_call_execution is not None:
            generic_update = parent_queue_item.session.fail_generic_child(child_queue_item.item_id, child_error_message)
            if generic_update is not None and generic_update.status != "failed":
                return
        if not self.fail_waiting_workflow_call(parent_queue_item, child_error_message):
            return
        if workflow_call_execution is not None:
            self._session_runner._services.session_queue.cancel_workflow_call_children(
                workflow_call_execution.id,
                exclude_item_ids={child_queue_item.item_id},
            )
        try:
            parent_queue_item = self._session_runner._services.session_queue.get_queue_item(parent_queue_item.item_id)
        except SessionQueueItemNotFoundError:
            return
        if getattr(parent_queue_item, "parent_item_id", None) is not None:
            self._fail_parent_from_failed_child(parent_queue_item)

    def _cancel_parent_from_canceled_child(self, child_queue_item: SessionQueueItem) -> None:
        parent_queue_item = self._get_parent_queue_item(child_queue_item)
        if parent_queue_item is None or parent_queue_item.status == "canceled":
            return
        generic_update = parent_queue_item.session.cancel_generic_child(child_queue_item.item_id, "child canceled")
        if generic_update is not None and generic_update.status != "canceled":
            return
        if generic_update is not None:
            saved = self._session_runner._save_queue_item_session_if_active(parent_queue_item)
            if not saved:
                return
        self._session_runner._services.session_queue.cancel_queue_item(parent_queue_item.item_id)

    def run_queue_item(self, queue_item: SessionQueueItem) -> None:
        self._session_runner.run(queue_item)
        try:
            updated_queue_item = self._session_runner._services.session_queue.get_queue_item(queue_item.item_id)
        except SessionQueueItemNotFoundError:
            # The queue item was deleted while it was running (e.g. the queue was cleared or the current item
            # was deleted). There is no parent lifecycle work to do for a deleted item.
            return
        if getattr(updated_queue_item, "parent_item_id", None) is None:
            return
        try:
            if updated_queue_item.status == "completed":
                self._resume_parent_from_completed_child(updated_queue_item)
            elif updated_queue_item.status == "failed":
                self._fail_parent_from_failed_child(updated_queue_item)
            elif updated_queue_item.status == "canceled":
                self._cancel_parent_from_canceled_child(updated_queue_item)
        except SessionQueueItemNotFoundError:
            # An item in the parent chain was deleted after we looked it up but before we mutated it
            # (the queue mutations re-read the row and raise when it has disappeared). The chain is
            # being torn down; there is nothing left to update.
            return
