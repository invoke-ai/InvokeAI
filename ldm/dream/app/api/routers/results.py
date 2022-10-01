# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from fastapi import Path, Query
from fastapi.routing import APIRouter
from fastapi.responses import FileResponse
from ..dependencies import ApiDependencies

results_router = APIRouter(
    prefix = '/v1/results',
    tags = ['results']
)


@results_router.get('/')
async def get_result(
    path: str = Query(description = "Path to the result to get") # TODO: make this a path var
):
    """Gets a result"""
    # TODO: This is not really secure at all. At least make sure only output results are served
    filename = ApiDependencies.invoker.services.images.get_path(path)
    return FileResponse(filename)
