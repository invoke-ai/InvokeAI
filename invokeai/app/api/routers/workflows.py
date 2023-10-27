from fastapi import APIRouter, Path

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.invocations.baseinvocation import WorkflowField

workflows_router = APIRouter(prefix="/v1/workflows", tags=["workflows"])


@workflows_router.get(
    "/i/{workflow_id}",
    operation_id="get_workflow",
    responses={
        200: {"model": WorkflowField},
    },
)
async def get_workflow(
    workflow_id: str = Path(description="The workflow to get"),
) -> WorkflowField:
    """Gets a workflow"""
    return ApiDependencies.invoker.services.workflow_records.get(workflow_id)
