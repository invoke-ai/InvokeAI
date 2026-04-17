from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import InputField, UIType
from invokeai.app.invocations.primitives import IntegerOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowCategory, WorkflowNotFoundError


@invocation(
    "call_saved_workflow",
    title="Call Saved Workflow",
    tags=["workflow", "saved", "library"],
    category="workflow",
    version="1.0.0",
    use_cache=False,
    classification=Classification.Beta,
)
class CallSavedWorkflowInvocation(BaseInvocation):
    """Displays and later executes against a selected saved workflow."""

    workflow_id: str = InputField(
        default="",
        description="The selected saved workflow ID, managed by the workflow editor UI.",
        ui_type=UIType.SavedWorkflow,
    )

    def validate_selected_workflow(self, context: InvocationContext):
        if not self.workflow_id:
            raise ValueError("A saved workflow must be selected before executing call_saved_workflow.")

        try:
            workflow_record = context._services.workflow_records.get(self.workflow_id)
        except WorkflowNotFoundError as e:
            raise ValueError(f"The selected saved workflow '{self.workflow_id}' could not be found.") from e

        config = context._services.configuration
        if config.multiuser:
            queue_user_id = context._data.queue_item.user_id
            user = context._services.users.get(queue_user_id)
            is_admin = bool(user and user.is_admin)
            is_owner = workflow_record.user_id == queue_user_id
            is_default = workflow_record.workflow.meta.category is WorkflowCategory.Default
            if not (is_default or is_owner or workflow_record.is_public or is_admin):
                raise ValueError(f"The selected saved workflow '{self.workflow_id}' is not accessible to this user.")

        return workflow_record

    def invoke(self, context: InvocationContext) -> IntegerOutput:
        self.validate_selected_workflow(context)

        return IntegerOutput(value=0)
