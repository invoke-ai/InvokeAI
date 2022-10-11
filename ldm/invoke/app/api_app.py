# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import asyncio
from inspect import signature
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from fastapi_events.middleware import EventHandlerASGIMiddleware
from fastapi_events.handlers.local import local_handler
from fastapi.middleware.cors import CORSMiddleware
from pydantic.schema import schema
import uvicorn
from .api.sockets import SocketIO
from .invocations import *
from .invocations.baseinvocation import BaseInvocation
from .api.routers import images, sessions
from .api.dependencies import ApiDependencies
from ..args import Args

origins = []

# Create the app
# TODO: create this all in a method so configuration/etc. can be passed in?
app = FastAPI(
    title     = "Invoke AI",
    docs_url  = None,
    redoc_url = None
)

# Add event handler
event_handler_id: int = id(app)
app.add_middleware(
    EventHandlerASGIMiddleware,
    handlers      = [local_handler], # TODO: consider doing this in services to support different configurations
    middleware_id = event_handler_id)

# Add CORS
# TODO: use configuration for this
origins = []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

socket_io = SocketIO(app)

config = {}

# Add startup event to load dependencies
@app.on_event('startup')
async def startup_event():
    args = Args()
    config = args.parse_args()

    ApiDependencies.initialize(
        config           = config,
        event_handler_id = event_handler_id
    )

# Shut down threads
@app.on_event('shutdown')
async def shutdown_event():
    ApiDependencies.shutdown()

# Include all routers
# TODO: REMOVE
# app.include_router(
#     invocation.invocation_router,
#     prefix = '/api')

app.include_router(
    sessions.session_router,
    prefix = '/api'
)

app.include_router(
    images.images_router,
    prefix = '/api'
)

# Build a custom OpenAPI to include all outputs
# TODO: can outputs be included on metadata of invocation schemas somehow?
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title       = app.title,
        description = "An API for invoking AI image operations",
        version     = "1.0.0",
        routes      = app.routes
    )

    # Add all outputs
    all_invocations = BaseInvocation.get_invocations()
    output_types = set()
    output_type_titles = dict()
    for invoker in all_invocations:
        output_type = signature(invoker.invoke).return_annotation
        output_types.add(output_type)

    output_schemas = schema(output_types, ref_prefix="#/components/schemas/")
    for schema_key, output_schema in output_schemas['definitions'].items():
        openapi_schema["components"]["schemas"][schema_key] = output_schema

        # TODO: note that we assume the schema_key here is the TYPE.__name__
        # This could break in some cases, figure out a better way to do it
        output_type_titles[schema_key] = output_schema['title']

    # Add a reference to the output type to additionalProperties of the invoker schema
    for invoker in all_invocations:
        invoker_name = invoker.__name__
        output_type = signature(invoker.invoke).return_annotation
        output_type_title = output_type_titles[output_type.__name__]
        invoker_schema = openapi_schema["components"]["schemas"][invoker_name]
        outputs_ref = { '$ref': f'#/components/schemas/{output_type_title}' }
        if 'additionalProperties' not in invoker_schema:
            invoker_schema['additionalProperties'] = {}

        invoker_schema['additionalProperties']['outputs'] = outputs_ref
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Override API doc favicons
app.mount('/static', StaticFiles(directory='static/dream_web'), name='static')

@app.get("/docs", include_in_schema=False)
def overridden_swagger():
	return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title,
        swagger_favicon_url="/static/favicon.ico"
    )

@app.get("/redoc", include_in_schema=False)
def overridden_redoc():
	return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title,
        redoc_favicon_url="/static/favicon.ico"
    )

def invoke_api():
    # Start our own event loop for eventing usage
    # TODO: determine if there's a better way to do this
    loop = asyncio.new_event_loop()
    config = uvicorn.Config(
        app = app,
        host = "0.0.0.0",
        port = 9090,
        loop = loop)
        # Use access_log to turn off logging
    
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    invoke_api()
