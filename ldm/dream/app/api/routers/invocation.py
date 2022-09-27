# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from fastapi import status
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
from ..dependencies import ApiDependencies
from ...services.invocation_graph import InvocationGraph


invocation_router = APIRouter(
    prefix = '/v1/invocations',
    tags = ['invocation']
)

@invocation_router.post(
    '/',
    status_code=status.HTTP_201_CREATED,
    responses = {
        400: {'description': 'Invalid json'}
    })
async def create_invocation(invocation_graph: InvocationGraph) -> dict:
    context = ApiDependencies.invoker.create_context_from_graph(invocation_graph)
    ApiDependencies.invoker.invoke(context, invoke_all = True)
    return JSONResponse(status_code = status.HTTP_201_CREATED, content={ "id": context.id }) # TODO: add content schema
