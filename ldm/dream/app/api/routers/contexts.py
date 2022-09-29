# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import List, Optional, Union
from typing_extensions import Annotated
from fastapi import Query, Path, Body
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse, Response
from grpc import StatusCode
from pydantic.fields import Field
from ..dependencies import ApiDependencies
from ...services.invocation_context import ContextConflict, InvocationContext, InvocationFieldLink
from ...services.invocation_graph import InvocationGraph
from ...services.context_manager import PaginatedContexts
from ...invocations.baseinvocation import BaseInvocation
from ...invocations import *

context_router = APIRouter(
    prefix = '/v1/contexts',
    tags = ['contexts']
)


@context_router.post('/',
    responses = {
        400: {'description': 'Invalid json'}
    })
async def create_context(
    invocation_graph: Optional[InvocationGraph] = Body(default = None, description = "The invocation graph to initialize the context with")
) -> InvocationContext:
    """Creates a new context, optionally initializing it with an invocation graph"""
    context = ApiDependencies.invoker.create_context_from_graph(invocation_graph)
    return context


@context_router.get('/')
async def list_contexts(
    page: Optional[int]     = Query(default = 0, description = "The page of results to get"),
    per_page: Optional[int] = Query(default = 10, description = "The number of results per page")
) -> PaginatedContexts:
    """Gets a paged list of context ids"""
    result = ApiDependencies.invoker.invoker_services.context_manager.list(page, per_page)
    return result


@context_router.get('/{context_id}',
    responses = {
        404: {'description': 'Context not found'}
    })
async def get_context(
    context_id: str = Path(description = "The id of the context to get")
) -> InvocationContext:
    """Gets a single context"""
    context = ApiDependencies.invoker.invoker_services.context_manager.get(context_id)
    if not context:
        return Response(status_code = 404)
    else:
        return context


@context_router.post('/{context_id}/invocations',
    responses = {
        400: {'description': 'Invalid invocation or link'},
        404: {'description': 'Context not found'}
    }
)
async def append_invocation(
    context_id: str = Path(description = "The id of the context to invoke"),
    invocation: Annotated[Union[BaseInvocation.get_invocations()], Field(discriminator="type")] = Body(description = "The invocation to add"),
    links: List[InvocationFieldLink] = Body(default=list(), description = "Links from previous invocations to the new invocation")
) -> InvocationContext:
    context = ApiDependencies.invoker.invoker_services.context_manager.get(context_id)
    if not context:
        return Response(status_code = 404)

    try:
        context.add_invocation(invocation, links)
        return context
    except ContextConflict:
        return Response(status_code = 400)
    except IndexError:
        return Response(status_code = 400)


@context_router.put('/{context_id}/invoke',
    responses = {
        202: {'description': 'The invocation is queued'},
        400: {'description': 'The context has no invocations ready to invoke'},
        404: {'description': 'Context not found'}
    })
async def invoke_context(
    context_id: str = Path(description = "The id of the context to invoke"),
    all: bool       = Query(default = False, description = "Whether or not to invoke all remaining invocations")
) -> None:
    """Invokes the context"""
    context = ApiDependencies.invoker.invoker_services.context_manager.get(context_id)
    if not context:
        return Response(status_code = 404)
    
    if not context.ready_to_invoke():
        return Response(StatusCode = 400)
    
    ApiDependencies.invoker.invoke(context, invoke_all = all)
    return Response(StatusCode=202)
