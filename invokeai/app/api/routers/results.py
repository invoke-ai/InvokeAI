from fastapi import HTTPException, Path, Query
from fastapi.routing import APIRouter
from invokeai.app.services.results import ResultType, ResultWithSession
from invokeai.app.services.item_storage import PaginatedResults

from ..dependencies import ApiDependencies

results_router = APIRouter(prefix="/v1/results", tags=["results"])


@results_router.get("/{result_type}/{result_name}", operation_id="get_result")
async def get_result(
    result_type: ResultType = Path(description="The type of result to get"),
    result_name: str = Path(description="The name of the result to get"),
) -> ResultWithSession:
    """Gets a result"""

    result = ApiDependencies.invoker.services.results.get(
        result_id=result_name, result_type=result_type
    )

    if result is not None:
        return result
    else:
        raise HTTPException(status_code=404)


@results_router.get(
    "/",
    operation_id="list_results",
    responses={200: {"model": PaginatedResults[ResultWithSession]}},
)
async def list_results(
    result_type: ResultType = Query(description="The type of results to get"),
    page: int = Query(default=0, description="The page of results to get"),
    per_page: int = Query(default=10, description="The number of results per page"),
) -> PaginatedResults[ResultWithSession]:
    """Gets a list of results"""
    results = ApiDependencies.invoker.services.results.get_many(
        result_type=result_type, page=page, per_page=per_page
    )
    return results
