# Copyright (c) 2022-2023 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI Team
import asyncio
import sys
from inspect import signature

import logging
import uvicorn
import socket

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi_events.handlers.local import local_handler
from fastapi_events.middleware import EventHandlerASGIMiddleware
from pathlib import Path
from pydantic.schema import schema

# This should come early so that modules can log their initialization properly
from .services.config import InvokeAIAppConfig
from ..backend.util.logging import InvokeAILogger

app_config = InvokeAIAppConfig.get_config()
app_config.parse_args()
logger = InvokeAILogger.getLogger(config=app_config)
from invokeai.version.invokeai_version import __version__

# we call this early so that the message appears before
# other invokeai initialization messages
if app_config.version:
    print(f"InvokeAI version {__version__}")
    sys.exit(0)

import invokeai.frontend.web as web_dir
import mimetypes

from .api.dependencies import ApiDependencies
from .api.routers import sessions, models, images, boards, board_images, app_info
from .api.sockets import SocketIO
from .invocations.baseinvocation import BaseInvocation


import torch
import invokeai.backend.util.hotfixes

if torch.backends.mps.is_available():
    import invokeai.backend.util.mps_fixes

# fix for windows mimetypes registry entries being borked
# see https://github.com/invoke-ai/InvokeAI/discussions/3684#discussioncomment-6391352
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

# Create the app
# TODO: create this all in a method so configuration/etc. can be passed in?
app = FastAPI(title="Invoke AI", docs_url=None, redoc_url=None)

# Add event handler
event_handler_id: int = id(app)
app.add_middleware(
    EventHandlerASGIMiddleware,
    handlers=[local_handler],  # TODO: consider doing this in services to support different configurations
    middleware_id=event_handler_id,
)

socket_io = SocketIO(app)


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

    ApiDependencies.initialize(config=app_config, event_handler_id=event_handler_id, logger=logger)


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

app.include_router(boards.boards_router, prefix="/api")

app.include_router(board_images.board_images_router, prefix="/api")

app.include_router(app_info.app_router, prefix="/api")


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

    from invokeai.backend.model_management.models import get_model_config_enums

    for model_config_format_enum in set(get_model_config_enums()):
        name = model_config_format_enum.__qualname__

        if name in openapi_schema["components"]["schemas"]:
            # print(f"Config with name {name} already defined")
            continue

        # "BaseModelType":{"title":"BaseModelType","description":"An enumeration.","enum":["sd-1","sd-2"],"type":"string"}
        openapi_schema["components"]["schemas"][name] = dict(
            title=name,
            description="An enumeration.",
            type="string",
            enum=list(v.value for v in model_config_format_enum),
        )

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Override API doc favicons
app.mount("/static", StaticFiles(directory=Path(web_dir.__path__[0], "static/dream_web")), name="static")


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
app.mount("/", StaticFiles(directory=Path(web_dir.__path__[0], "dist"), html=True), name="ui")


def invoke_api():
    def find_port(port: int):
        """Find a port not in use starting at given port"""
        # Taken from https://waylonwalker.com/python-find-available-port/, thanks Waylon!
        # https://github.com/WaylonWalker
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) == 0:
                return find_port(port=port + 1)
            else:
                return port

    from invokeai.backend.install.check_root import check_invokeai_root

    check_invokeai_root(app_config)  # note, may exit with an exception if root not set up

    port = find_port(app_config.port)
    if port != app_config.port:
        logger.warn(f"Port {app_config.port} in use, using port {port}")
        
    # Start our own event loop for eventing usage
    loop = asyncio.new_event_loop()
    config = uvicorn.Config(
        app=app,
        host=app_config.host,
        port=port,
        loop=loop,
        log_level=app_config.log_level,
    )
    server = uvicorn.Server(config)

    # replace uvicorn's loggers with InvokeAI's for consistent appearance
    for logname in ["uvicorn.access", "uvicorn"]:
        l = logging.getLogger(logname)
        l.handlers.clear()
        for ch in logger.handlers:
            l.addHandler(ch)
    
    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    invoke_api()
