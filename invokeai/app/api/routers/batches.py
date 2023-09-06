# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from fastapi import Body, HTTPException, Path, Response
from fastapi.routing import APIRouter

from invokeai.app.services.batch_manager_storage import BatchSession, BatchSessionNotFoundException

# Importing * is bad karma but needed here for node detection
from ...invocations import *  # noqa: F401 F403
from ...services.batch_manager import Batch, BatchProcessResponse
from ...services.graph import Graph
from ..dependencies import ApiDependencies

batches_router = APIRouter(prefix="/v1/batches", tags=["sessions"])


@batches_router.post(
    "/",
    operation_id="create_batch",
    responses={
        200: {"model": BatchProcessResponse},
        400: {"description": "Invalid json"},
    },
)
async def create_batch(
    graph: Graph = Body(description="The graph to initialize the session with"),
    batch: Batch = Body(description="Batch config to apply to the given graph"),
) -> BatchProcessResponse:
    """Creates a batch process"""
    return ApiDependencies.invoker.services.batch_manager.create_batch_process(batch, graph)


@batches_router.put(
    "/b/{batch_process_id}/invoke",
    operation_id="start_batch",
    responses={
        202: {"description": "Batch process started"},
        404: {"description": "Batch session not found"},
    },
)
async def start_batch(
    batch_process_id: str = Path(description="ID of Batch to start"),
) -> Response:
    """Executes a batch process"""
    try:
        ApiDependencies.invoker.services.batch_manager.run_batch_process(batch_process_id)
        return Response(status_code=202)
    except BatchSessionNotFoundException:
        raise HTTPException(status_code=404, detail="Batch session not found")


@batches_router.delete(
    "/b/{batch_process_id}",
    operation_id="cancel_batch",
    responses={202: {"description": "The batch is canceled"}},
)
async def cancel_batch(
    batch_process_id: str = Path(description="The id of the batch process to cancel"),
) -> Response:
    """Cancels a batch process"""
    ApiDependencies.invoker.services.batch_manager.cancel_batch_process(batch_process_id)
    return Response(status_code=202)


@batches_router.get(
    "/incomplete",
    operation_id="list_incomplete_batches",
    responses={200: {"model": list[BatchProcessResponse]}},
)
async def list_incomplete_batches() -> list[BatchProcessResponse]:
    """Lists incomplete batch processes"""
    return ApiDependencies.invoker.services.batch_manager.get_incomplete_batch_processes()


@batches_router.get(
    "/",
    operation_id="list_batches",
    responses={200: {"model": list[BatchProcessResponse]}},
)
async def list_batches() -> list[BatchProcessResponse]:
    """Lists all batch processes"""
    return ApiDependencies.invoker.services.batch_manager.get_batch_processes()


@batches_router.get(
    "/b/{batch_process_id}",
    operation_id="get_batch",
    responses={200: {"model": BatchProcessResponse}},
)
async def get_batch(
    batch_process_id: str = Path(description="The id of the batch process to get"),
) -> BatchProcessResponse:
    """Gets a Batch Process"""
    return ApiDependencies.invoker.services.batch_manager.get_batch(batch_process_id)


@batches_router.get(
    "/b/{batch_process_id}/sessions",
    operation_id="get_batch_sessions",
    responses={200: {"model": list[BatchSession]}},
)
async def get_batch_sessions(
    batch_process_id: str = Path(description="The id of the batch process to get"),
) -> list[BatchSession]:
    """Gets a list of batch sessions for a given batch process"""
    return ApiDependencies.invoker.services.batch_manager.get_sessions(batch_process_id)
