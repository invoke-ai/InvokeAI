# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from typing import List, Optional, Union
from typing_extensions import Annotated
from fastapi import Query, Path, Body
from fastapi.routing import APIRouter
from fastapi.responses import Response
from pydantic.fields import Field
from ..dependencies import ApiDependencies
from ...services.invocation_session import SessionConflict, InvocationSession, InvocationFieldLink
from ...services.invocation_graph import InvocationGraph
from ...services.session_manager import PaginatedSession
from ...invocations.baseinvocation import BaseInvocation
from ...invocations import *

session_router = APIRouter(
    prefix = '/v1/sessions',
    tags = ['sessions']
)


@session_router.post('/',
    operation_id = 'create_session',
    responses = {
        400: {'description': 'Invalid json'}
    })
async def create_session(
    invocation_graph: Optional[InvocationGraph] = Body(default = None, description = "The invocation graph to initialize the session with")
) -> InvocationSession:
    """Creates a new sessions, optionally initializing it with an invocation graph"""
    sessions = ApiDependencies.invoker.create_session_from_graph(invocation_graph)
    return sessions


@session_router.get('/', operation_id = 'list_sessions')
async def list_sessions(
    page: Optional[int]     = Query(default = 0, description = "The page of results to get"),
    per_page: Optional[int] = Query(default = 10, description = "The number of results per page")
) -> PaginatedSession:
    """Gets a paged list of sessions ids"""
    result = ApiDependencies.invoker.invoker_services.session_manager.list(page, per_page)
    return result


@session_router.get('/{session_id}',
    operation_id = 'get_session',
    responses = {
        404: {'description': 'Session not found'}
    })
async def get_session(
    session_id: str = Path(description = "The id of the session to get")
) -> InvocationSession:
    """Gets a single session"""
    session = ApiDependencies.invoker.invoker_services.session_manager.get(session_id)
    if not session:
        return Response(status_code = 404)
    else:
        return session


@session_router.post('/{session_id}/invocations',
    operation_id = 'append_invocation',
    responses = {
        400: {'description': 'Invalid invocation or link'},
        404: {'description': 'Session not found'}
    }
)
async def append_invocation(
    session_id: str = Path(description = "The id of the sessions to invoke"),
    invocation: Annotated[Union[BaseInvocation.get_invocations()], Field(discriminator="type")] = Body(description = "The invocation to add"),
    links: List[InvocationFieldLink] = Body(default=list(), description = "Links from previous invocations to the new invocation")
) -> InvocationSession:
    session = ApiDependencies.invoker.invoker_services.session_manager.get(session_id)
    if not session:
        return Response(status_code = 404)

    try:
        session.add_invocation(invocation, links)
        return session
    except SessionConflict:
        return Response(status_code = 400)
    except IndexError:
        return Response(status_code = 400)


@session_router.put('/{session_id}/invoke',
    operation_id = 'invoke_session',
    responses = {
        202: {'description': 'The invocation is queued'},
        400: {'description': 'The session has no invocations ready to invoke'},
        404: {'description': 'Session not found'}
    })
async def invoke_session(
    session_id: str = Path(description = "The id of the session to invoke"),
    all: bool       = Query(default = False, description = "Whether or not to invoke all remaining invocations")
) -> None:
    """Invokes the session"""
    session = ApiDependencies.invoker.invoker_services.session_manager.get(session_id)
    if not session:
        return Response(status_code = 404)
    
    if not session.ready_to_invoke():
        return Response(status_code = 400)
    
    ApiDependencies.invoker.invoke(session, invoke_all = all)
    return Response(status_code=202)
