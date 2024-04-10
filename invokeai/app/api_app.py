import asyncio
import logging
import mimetypes
import socket
from contextlib import asynccontextmanager
from inspect import signature
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse
from fastapi_events.handlers.local import local_handler
from fastapi_events.middleware import EventHandlerASGIMiddleware
from pydantic.json_schema import models_json_schema
from torch.backends.mps import is_available as is_mps_available

# for PyCharm:
# noinspection PyUnresolvedReferences
import invokeai.backend.util.hotfixes  # noqa: F401 (monkeypatching on import)
import invokeai.frontend.web as web_dir
from invokeai.app.api.no_cache_staticfiles import NoCacheStaticFiles
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.config.config_default import get_config
from invokeai.app.services.session_processor.session_processor_common import ProgressImage
from invokeai.backend.util.devices import TorchDeviceSelect

from ..backend.util.logging import InvokeAILogger
from .api.dependencies import ApiDependencies
from .api.routers import (
    app_info,
    board_images,
    boards,
    download_queue,
    images,
    model_manager,
    session_queue,
    utilities,
    workflows,
)
from .api.sockets import SocketIO
from .invocations.baseinvocation import (
    BaseInvocation,
    UIConfigBase,
)
from .invocations.fields import InputFieldJSONSchemaExtra, OutputFieldJSONSchemaExtra

app_config = get_config()


if is_mps_available():
    import invokeai.backend.util.mps_fixes  # noqa: F401 (monkeypatching on import)


logger = InvokeAILogger.get_logger(config=app_config)
# fix for windows mimetypes registry entries being borked
# see https://github.com/invoke-ai/InvokeAI/discussions/3684#discussioncomment-6391352
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

torch_device_name = TorchDeviceSelect.get_torch_device_name()
logger.info(f"Using torch device: {torch_device_name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Add startup event to load dependencies
    ApiDependencies.initialize(config=app_config, event_handler_id=event_handler_id, logger=logger)
    yield
    # Shut down threads
    ApiDependencies.shutdown()


# Create the app
# TODO: create this all in a method so configuration/etc. can be passed in?
app = FastAPI(
    title="Invoke - Community Edition",
    docs_url=None,
    redoc_url=None,
    separate_input_output_schemas=False,
    lifespan=lifespan,
)

# Add event handler
event_handler_id: int = id(app)
app.add_middleware(
    EventHandlerASGIMiddleware,
    handlers=[local_handler],  # TODO: consider doing this in services to support different configurations
    middleware_id=event_handler_id,
)

socket_io = SocketIO(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_config.allow_origins,
    allow_credentials=app_config.allow_credentials,
    allow_methods=app_config.allow_methods,
    allow_headers=app_config.allow_headers,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Include all routers
app.include_router(utilities.utilities_router, prefix="/api")
app.include_router(model_manager.model_manager_router, prefix="/api")
app.include_router(download_queue.download_queue_router, prefix="/api")
app.include_router(images.images_router, prefix="/api")
app.include_router(boards.boards_router, prefix="/api")
app.include_router(board_images.board_images_router, prefix="/api")
app.include_router(app_info.app_router, prefix="/api")
app.include_router(session_queue.session_queue_router, prefix="/api")
app.include_router(workflows.workflows_router, prefix="/api")


# Build a custom OpenAPI to include all outputs
# TODO: can outputs be included on metadata of invocation schemas somehow?
def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        description="An API for invoking AI image operations",
        version="1.0.0",
        routes=app.routes,
        separate_input_output_schemas=False,  # https://fastapi.tiangolo.com/how-to/separate-openapi-schemas/
    )

    # Add all outputs
    all_invocations = BaseInvocation.get_invocations()
    output_types = set()
    output_type_titles = {}
    for invoker in all_invocations:
        output_type = signature(invoker.invoke).return_annotation
        output_types.add(output_type)

    output_schemas = models_json_schema(
        models=[(o, "serialization") for o in output_types], ref_template="#/components/schemas/{model}"
    )
    for schema_key, output_schema in output_schemas[1]["$defs"].items():
        # TODO: note that we assume the schema_key here is the TYPE.__name__
        # This could break in some cases, figure out a better way to do it
        output_type_titles[schema_key] = output_schema["title"]
        openapi_schema["components"]["schemas"][schema_key] = output_schema
        openapi_schema["components"]["schemas"][schema_key]["class"] = "output"

    # Some models don't end up in the schemas as standalone definitions
    additional_schemas = models_json_schema(
        [
            (UIConfigBase, "serialization"),
            (InputFieldJSONSchemaExtra, "serialization"),
            (OutputFieldJSONSchemaExtra, "serialization"),
            (ModelIdentifierField, "serialization"),
            (ProgressImage, "serialization"),
        ],
        ref_template="#/components/schemas/{model}",
    )
    for schema_key, schema_json in additional_schemas[1]["$defs"].items():
        openapi_schema["components"]["schemas"][schema_key] = schema_json

    # Add a reference to the output type to additionalProperties of the invoker schema
    for invoker in all_invocations:
        invoker_name = invoker.__name__  # type: ignore [attr-defined] # this is a valid attribute
        output_type = signature(obj=invoker.invoke).return_annotation
        output_type_title = output_type_titles[output_type.__name__]
        invoker_schema = openapi_schema["components"]["schemas"][f"{invoker_name}"]
        outputs_ref = {"$ref": f"#/components/schemas/{output_type_title}"}
        invoker_schema["output"] = outputs_ref
        invoker_schema["class"] = "invocation"

    # This code no longer seems to be necessary?
    # Leave it here just in case
    #
    # from invokeai.backend.model_manager import get_model_config_formats
    # formats = get_model_config_formats()
    # for model_config_name, enum_set in formats.items():

    #     if model_config_name in openapi_schema["components"]["schemas"]:
    #         # print(f"Config with name {name} already defined")
    #         continue

    #     openapi_schema["components"]["schemas"][model_config_name] = {
    #         "title": model_config_name,
    #         "description": "An enumeration.",
    #         "type": "string",
    #         "enum": [v.value for v in enum_set],
    #     }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore [method-assign] # this is a valid assignment


@app.get("/docs", include_in_schema=False)
def overridden_swagger() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,  # type: ignore [arg-type] # this is always a string
        title=f"{app.title} - Swagger UI",
        swagger_favicon_url="static/docs/invoke-favicon-docs.svg",
    )


@app.get("/redoc", include_in_schema=False)
def overridden_redoc() -> HTMLResponse:
    return get_redoc_html(
        openapi_url=app.openapi_url,  # type: ignore [arg-type] # this is always a string
        title=f"{app.title} - Redoc",
        redoc_favicon_url="static/docs/invoke-favicon-docs.svg",
    )


web_root_path = Path(list(web_dir.__path__)[0])

try:
    app.mount("/", NoCacheStaticFiles(directory=Path(web_root_path, "dist"), html=True), name="ui")
except RuntimeError:
    logger.warn(f"No UI found at {web_root_path}/dist, skipping UI mount")
app.mount(
    "/static", NoCacheStaticFiles(directory=Path(web_root_path, "static/")), name="static"
)  # docs favicon is in here


def check_cudnn(logger: logging.Logger) -> None:
    """Check for cuDNN issues that could be causing degraded performance."""
    if torch.backends.cudnn.is_available():
        try:
            # Note: At the time of writing (torch 2.2.1), torch.backends.cudnn.version() only raises an error the first
            # time it is called. Subsequent calls will return the version number without complaining about a mismatch.
            cudnn_version = torch.backends.cudnn.version()
            logger.info(f"cuDNN version: {cudnn_version}")
        except RuntimeError as e:
            logger.warning(
                "Encountered a cuDNN version issue. This may result in degraded performance. This issue is usually "
                "caused by an incompatible cuDNN version installed in your python environment, or on the host "
                f"system. Full error message:\n{e}"
            )


def invoke_api() -> None:
    def find_port(port: int) -> int:
        """Find a port not in use starting at given port"""
        # Taken from https://waylonwalker.com/python-find-available-port/, thanks Waylon!
        # https://github.com/WaylonWalker
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) == 0:
                return find_port(port=port + 1)
            else:
                return port

    if app_config.dev_reload:
        try:
            import jurigged
        except ImportError as e:
            logger.error(
                'Can\'t start `--dev_reload` because jurigged is not found; `pip install -e ".[dev]"` to include development dependencies.',
                exc_info=e,
            )
        else:
            jurigged.watch(logger=InvokeAILogger.get_logger(name="jurigged").info)

    port = find_port(app_config.port)
    if port != app_config.port:
        logger.warn(f"Port {app_config.port} in use, using port {port}")

    check_cudnn(logger)

    # Start our own event loop for eventing usage
    loop = asyncio.new_event_loop()
    config = uvicorn.Config(
        app=app,
        host=app_config.host,
        port=port,
        loop="asyncio",
        log_level=app_config.log_level,
        ssl_certfile=app_config.ssl_certfile,
        ssl_keyfile=app_config.ssl_keyfile,
    )
    server = uvicorn.Server(config)

    # replace uvicorn's loggers with InvokeAI's for consistent appearance
    for logname in ["uvicorn.access", "uvicorn"]:
        log = InvokeAILogger.get_logger(logname)
        log.handlers.clear()
        for ch in logger.handlers:
            log.addHandler(ch)

    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    invoke_api()
