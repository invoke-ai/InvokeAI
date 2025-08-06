import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi_events.handlers.local import local_handler
from fastapi_events.middleware import EventHandlerASGIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

import invokeai.frontend.web as web_dir
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.no_cache_staticfiles import NoCacheStaticFiles
from invokeai.app.api.routers import (
    app_info,
    board_images,
    boards,
    client_state,
    download_queue,
    images,
    model_manager,
    model_relationships,
    session_queue,
    style_presets,
    utilities,
    workflows,
)
from invokeai.app.api.sockets import SocketIO
from invokeai.app.services.config.config_default import get_config
from invokeai.app.util.custom_openapi import get_openapi_func
from invokeai.backend.util.logging import InvokeAILogger

app_config = get_config()
logger = InvokeAILogger.get_logger(config=app_config)

loop = asyncio.new_event_loop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Add startup event to load dependencies
    ApiDependencies.initialize(config=app_config, event_handler_id=event_handler_id, loop=loop, logger=logger)

    # Log the server address when it starts - in case the network log level is not high enough to see the startup log
    proto = "https" if app_config.ssl_certfile else "http"
    msg = f"Invoke running on {proto}://{app_config.host}:{app_config.port} (Press CTRL+C to quit)"

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
app.include_router(model_relationships.model_relationships_router, prefix="/api")
app.include_router(app_info.app_router, prefix="/api")
app.include_router(session_queue.session_queue_router, prefix="/api")
app.include_router(workflows.workflows_router, prefix="/api")
app.include_router(style_presets.style_presets_router, prefix="/api")
app.include_router(client_state.client_state_router, prefix="/api")

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

if app_config.unsafe_disable_picklescan:
    logger.warning(
        "The unsafe_disable_picklescan option is enabled. This disables malware scanning while installing and"
        "loading models, which may allow malicious code to be executed. Use at your own risk."
    )

try:
    app.mount("/", NoCacheStaticFiles(directory=Path(web_root_path, "dist"), html=True), name="ui")
except RuntimeError:
    logger.warning(f"No UI found at {web_root_path}/dist, skipping UI mount")
app.mount(
    "/static", NoCacheStaticFiles(directory=Path(web_root_path, "static/")), name="static"
)  # docs favicon is in here
