# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)
import asyncio
from inspect import signature

import uvicorn
from invokeai.backend.util.logging import InvokeAILogger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi_events.handlers.local import local_handler
from fastapi_events.middleware import EventHandlerASGIMiddleware
from pydantic.schema import schema

from .api.dependencies import ApiDependencies
from .api.routers import sessions, models, images
from .api.sockets import SocketIO
from .invocations.baseinvocation import BaseInvocation
from .services.config import InvokeAIAppConfig

logger = InvokeAILogger.getLogger()

# Create the app
# TODO: create this all in a method so configuration/etc. can be passed in?
app = FastAPI(title="Invoke AI", docs_url=None, redoc_url=None)

# Add event handler
event_handler_id: int = id(app)
app.add_middleware(
    EventHandlerASGIMiddleware,
    handlers=[
        local_handler
    ],  # TODO: consider doing this in services to support different configurations
    middleware_id=event_handler_id,
)

socket_io = SocketIO(app)

# initialize config
# this is a module global
app_config = InvokeAIAppConfig()

# Add startup event to load dependencies
@app.on_event("startup")
async def startup_event():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_config.allow_origins,
        allow_credentials=app_config.allow_credentials,
        allow_methods=app_config.allow_methods,
        allow_headers=app_config.allow_headers,
    )

    ApiDependencies.initialize(
        config=app_config, event_handler_id=event_handler_id, logger=logger
    )


# Shut down threads
@app.on_event("shutdown")
async def shutdown_event():
    ApiDependencies.shutdown()


# Include all routers
# TODO: REMOVE
# app.include_router(
#     invocation.invocation_router,
#     prefix = '/api')

app.include_router(sessions.session_router, prefix="/api")

app.include_router(models.models_router, prefix="/api")

app.include_router(images.images_router, prefix="/api")

# Build a custom OpenAPI to include all outputs
# TODO: can outputs be included on metadata of invocation schemas somehow?
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        description="An API for invoking AI image operations",
        version="1.0.0",
        routes=app.routes,
    )

    # Add all outputs
    all_invocations = BaseInvocation.get_invocations()
    output_types = set()
    output_type_titles = dict()
    for invoker in all_invocations:
        output_type = signature(invoker.invoke).return_annotation
        output_types.add(output_type)

    output_schemas = schema(output_types, ref_prefix="#/components/schemas/")
    for schema_key, output_schema in output_schemas["definitions"].items():
        openapi_schema["components"]["schemas"][schema_key] = output_schema

        # TODO: note that we assume the schema_key here is the TYPE.__name__
        # This could break in some cases, figure out a better way to do it
        output_type_titles[schema_key] = output_schema["title"]

    # Add a reference to the output type to additionalProperties of the invoker schema
    for invoker in all_invocations:
        invoker_name = invoker.__name__
        output_type = signature(invoker.invoke).return_annotation
        output_type_title = output_type_titles[output_type.__name__]
        invoker_schema = openapi_schema["components"]["schemas"][invoker_name]
        outputs_ref = {"$ref": f"#/components/schemas/{output_type_title}"}

        invoker_schema["output"] = outputs_ref

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Override API doc favicons
app.mount("/static", StaticFiles(directory="static/dream_web"), name="static")


@app.get("/docs", include_in_schema=False)
def overridden_swagger():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title,
        swagger_favicon_url="/static/favicon.ico",
    )


@app.get("/redoc", include_in_schema=False)
def overridden_redoc():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title,
        redoc_favicon_url="/static/favicon.ico",
    )


# Must mount *after* the other routes else it borks em
app.mount(
    "/", StaticFiles(directory="invokeai/frontend/web/dist", html=True), name="ui"
)


def invoke_api():
    # Start our own event loop for eventing usage
    loop = asyncio.new_event_loop()
    config = uvicorn.Config(app=app, host=app_config.host, port=app_config.port, loop=loop)
    # Use access_log to turn off logging
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    invoke_api()
