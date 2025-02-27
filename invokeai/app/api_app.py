import asyncio
import logging
import mimetypes
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi_events.handlers.local import local_handler
from fastapi_events.middleware import EventHandlerASGIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from torch.backends.mps import is_available as is_mps_available

# for PyCharm:
# noinspection PyUnresolvedReferences
import invokeai.backend.util.hotfixes  # noqa: F401 (monkeypatching on import)
import invokeai.frontend.web as web_dir
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.no_cache_staticfiles import NoCacheStaticFiles
from invokeai.app.api.routers import (
    app_info,
    board_images,
    boards,
    download_queue,
    images,
    model_manager,
    session_queue,
    style_presets,
    utilities,
    workflows,
)
from invokeai.app.api.sockets import SocketIO
from invokeai.app.invocations.load_custom_nodes import load_custom_nodes
from invokeai.app.services.config.config_default import get_config
from invokeai.app.util.custom_openapi import get_openapi_func
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

app_config = get_config()


if is_mps_available():
    import invokeai.backend.util.mps_fixes  # noqa: F401 (monkeypatching on import)


logger = InvokeAILogger.get_logger(config=app_config)
# fix for windows mimetypes registry entries being borked
# see https://github.com/invoke-ai/InvokeAI/discussions/3684#discussioncomment-6391352
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

torch_device_name = TorchDevice.get_torch_device_name()
logger.info(f"Using torch device: {torch_device_name}")

loop = asyncio.new_event_loop()

# We may change the port if the default is in use, this global variable is used to store the port so that we can log
# the correct port when the server starts in the lifespan handler.
port = app_config.port

# Load custom nodes. This must be done after importing the Graph class, which itself imports all modules from the
# invocations module. The ordering here is implicit, but important - we want to load custom nodes after all the
# core nodes have been imported so that we can catch when a custom node clobbers a core node.
load_custom_nodes(custom_nodes_path=app_config.custom_nodes_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Add startup event to load dependencies
    ApiDependencies.initialize(config=app_config, event_handler_id=event_handler_id, loop=loop, logger=logger)

    # Log the server address when it starts - in case the network log level is not high enough to see the startup log
    proto = "https" if app_config.ssl_certfile else "http"
    msg = f"Invoke running on {proto}://{app_config.host}:{port} (Press CTRL+C to quit)"

    # Logging this way ignores the logger's log level and _always_ logs the message
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="",
        lno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    logger.handle(record)

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


class RedirectRootWithQueryStringMiddleware(BaseHTTPMiddleware):
    """When a request is made to the root path with a query string, redirect to the root path without the query string.

    For example, to force a Gradio app to use dark mode, users may append `?__theme=dark` to the URL. Their browser may
    have this query string saved in history or a bookmark, so when the user navigates to `http://127.0.0.1:9090/`, the
    browser takes them to `http://127.0.0.1:9090/?__theme=dark`.

    This breaks the static file serving in the UI, so we redirect the user to the root path without the query string.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if request.url.path == "/" and request.url.query:
            return RedirectResponse(url="/")

        response = await call_next(request)
        return response


# Add the middleware
app.add_middleware(RedirectRootWithQueryStringMiddleware)


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
app.include_router(style_presets.style_presets_router, prefix="/api")

app.openapi = get_openapi_func(app)


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
            s.settimeout(1)
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

    global port
    port = find_port(app_config.port)
    if port != app_config.port:
        logger.warn(f"Port {app_config.port} in use, using port {port}")

    check_cudnn(logger)

    config = uvicorn.Config(
        app=app,
        host=app_config.host,
        port=port,
        loop="asyncio",
        log_level=app_config.log_level_network,
        ssl_certfile=app_config.ssl_certfile,
        ssl_keyfile=app_config.ssl_keyfile,
    )
    server = uvicorn.Server(config)

    # replace uvicorn's loggers with InvokeAI's for consistent appearance
    uvicorn_logger = InvokeAILogger.get_logger("uvicorn")
    uvicorn_logger.handlers.clear()
    for hdlr in logger.handlers:
        uvicorn_logger.addHandler(hdlr)

    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    invoke_api()
