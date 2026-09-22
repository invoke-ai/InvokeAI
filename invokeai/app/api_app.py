import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security.utils import get_authorization_scheme_param
from fastapi_events.handlers.local import local_handler
from fastapi_events.middleware import EventHandlerASGIMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.gzip import GZipMiddleware, GZipResponder, IdentityResponder
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import invokeai.frontend.web as web_dir
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.no_cache_staticfiles import NoCacheStaticFiles
from invokeai.app.api.routers import (
    app_info,
    auth,
    board_images,
    boards,
    client_state,
    custom_nodes,
    gallery,
    image_moves,
    images,
    model_manager,
    model_relationships,
    recall_parameters,
    session_queue,
    style_presets,
    system_prompts,
    utilities,
    videos,
    virtual_boards,
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

    # Re-derive open sockets' authorization from the database on a timer. This is what
    # catches user changes made by another process — the `invoke-usermod` / `invoke-userdel`
    # CLIs — which cannot raise an in-process event. `socket_io` is created further down
    # this module and is bound by the time the app is served.
    socket_io.start()

    try:
        yield
    finally:
        # In a `finally` so an exception propagating into the generator cannot leave the
        # sweep running against a half-torn-down process, or skip the thread shutdown.
        socket_io.stop()
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


class SlidingWindowTokenMiddleware(BaseHTTPMiddleware):
    """Refresh the JWT token on each authenticated response.

    When a request includes a valid Bearer token, the response includes a
    X-Refreshed-Token header with a new token that has a fresh expiry.
    This implements sliding-window session expiry: the session only expires
    after a period of *inactivity*, not a fixed time after login.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        response = await call_next(request)

        # Only refresh on mutating requests (POST/PUT/PATCH/DELETE) — these indicate
        # genuine user activity. GET requests are often background fetches (RTK Query
        # cache revalidation, refetch-on-focus, etc.) and should not reset the
        # inactivity timer.
        # Behind a sub-path proxy the public path carries the prefix (see SubPathASGIMiddleware),
        # so strip `root_path` before matching the auth-route exclusions — otherwise a proxied
        # logout would mint a replacement token/cookie right after the route deleted them.
        route_path = request.url.path.removeprefix(request.scope.get("root_path", ""))
        if (
            response.status_code < 400
            and request.method in ("POST", "PUT", "PATCH", "DELETE")
            and route_path not in ("/api/v1/auth/login", "/api/v1/auth/logout", "/api/v1/auth/media-cookie")
        ):
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                try:
                    from datetime import timedelta

                    from invokeai.app.api.routers.auth import (
                        TOKEN_EXPIRATION_NORMAL,
                        TOKEN_EXPIRATION_REMEMBER_ME,
                    )
                    from invokeai.app.services.auth.token_service import TokenData, create_access_token, verify_token

                    token_data = verify_token(token)
                    if token_data is not None:
                        # Decide through `resolve_authorized_user` rather than re-deriving
                        # "is this token still honored" here. This middleware used to carry
                        # its own copy of the exists/active/epoch checks, and a copy is how
                        # a rule added later — the refusal of the internal `system` id —
                        # reaches every other entry point but not this one, leaving a token
                        # nothing will accept being renewed indefinitely anyway.
                        #
                        # Never refresh a token that is no longer honored. This runs after
                        # the route, so an authenticated route has already rejected it — but
                        # an unauthenticated route returning 2xx with a stale Bearer header
                        # still reaches here, and minting from the current record would
                        # launder a revoked token into a valid one.
                        #
                        # The lookup inside is a synchronous SQLite query behind a
                        # process-wide lock; run it off the event loop so a contended lock
                        # (e.g. generation-result writes) can't stall every concurrent
                        # request from inside this per-mutation middleware.
                        from invokeai.app.api.auth_dependencies import resolve_authorized_user

                        user = await run_in_threadpool(resolve_authorized_user, token_data)
                        if user is None:
                            return response
                        # Mint the replacement from the *database* record, not the old
                        # token's claims: otherwise a demoted administrator's stale is_admin
                        # claim (and the media cookie carrying it) would be renewed
                        # indefinitely by their own mutations.
                        # Use the remember_me claim from the token to determine the
                        # correct refresh duration. This avoids the bug where a 7-day
                        # token with <24h remaining would be silently downgraded to 1 day.
                        if token_data.remember_me:
                            expires_delta = timedelta(days=TOKEN_EXPIRATION_REMEMBER_ME)
                        else:
                            expires_delta = timedelta(days=TOKEN_EXPIRATION_NORMAL)

                        refreshed_data = TokenData(
                            user_id=user.user_id,
                            email=user.email,
                            is_admin=user.is_admin,
                            remember_me=token_data.remember_me,
                            token_epoch=user.token_epoch,
                        )
                        new_token = create_access_token(refreshed_data, expires_delta)
                        response.headers["X-Refreshed-Token"] = new_token
                except Exception:
                    pass  # Don't fail the request if token refresh fails

        return response


# Response types worth compressing. Everything else is passed through untouched.
#
# This is an allowlist rather than a blocklist of media types on purpose: a type missing from
# this list only loses compression it would barely have benefited from, whereas a binary type
# missing from a blocklist costs real CPU on the event loop. The app serves a small, known set
# of compressible things — the UI bundle, the API's JSON, SVG icons.
COMPRESSIBLE_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/javascript",
    "application/xml",
    "application/xhtml+xml",
    "application/manifest+json",
    "image/svg+xml",
)

# `text/` would otherwise match this, and compressing an event stream defeats its purpose by
# withholding events until the compressor flushes. Starlette excludes it by default too.
UNCOMPRESSIBLE_CONTENT_TYPES = ("text/event-stream",)


def _is_compressible(content_type: str) -> bool:
    if content_type.startswith(UNCOMPRESSIBLE_CONTENT_TYPES):
        return False
    return content_type.startswith(COMPRESSIBLE_CONTENT_TYPES)


class _ContentTypeAwareGZipResponder(GZipResponder):
    """Skips compression for response types that are already compressed.

    `content_type_is_excluded` is computed when the response starts and only read once the
    body arrives, so widening it right after the base class has set it is enough — no need to
    reimplement Starlette's streaming/pathsend handling.
    """

    async def send_with_compression(self, message: Message) -> None:
        await super().send_with_compression(message)
        if message["type"] == "http.response.start" and not self.content_type_is_excluded:
            self.content_type_is_excluded = not _is_compressible(
                Headers(raw=message["headers"]).get("content-type", "")
            )


class ContentTypeAwareGZipMiddleware(GZipMiddleware):
    """GZip, but only for content types that actually compress.

    Starlette's GZipMiddleware compresses every response type except `text/event-stream`. The
    gallery serves PNG, WebP and MP4 bytes, which are already compressed: a 3 MB PNG costs
    ~52ms of event-loop time to gzip and comes back *larger* than it went in. With auto-switch
    enabled the UI fetches the full image after every generated image, so that cost lands
    repeatedly during a batch — exactly when the server can least afford to stall.

    Lowering `compresslevel` does not help here: on incompressible input, level 1 costs
    essentially the same as level 9 because deflate still has to scan the data. It does help a
    great deal on the compressible path — see `configure_gzip`.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if "gzip" in Headers(scope=scope).get("Accept-Encoding", ""):
            responder: ASGIApp = _ContentTypeAwareGZipResponder(
                self.app, self.minimum_size, compresslevel=self.compresslevel
            )
        else:
            responder = IdentityResponder(self.app, self.minimum_size)

        await responder(scope, receive, send)


# Responses below this are not worth a compression pass; the gzip framing alone is ~20 bytes.
GZIP_MINIMUM_SIZE = 1000


def configure_gzip(app: FastAPI, compresslevel: int) -> None:
    """Install response compression, unless it is turned off.

    Compression runs on the event loop, so its cost is not paid by the requesting client alone —
    it stalls every other request and every socket.io event for its duration. That makes the
    level a real trade-off rather than a free win.

    Measured on the flat name list of a 200k-image library (8.48 MB of JSON): level 1 takes
    16.4ms and returns 6.1% of the input, level 9 takes 90.2ms and returns 5.7%. Level 9 costs
    5.5x the event-loop time for 0.4 percentage points of bandwidth, which is a poor deal for a
    locally-served app. The default stays at 9 so behavior is unchanged for existing installs;
    users who feel the stall on a large library can lower it.

    A `compresslevel` of 0 means "no compression". The middleware is then left out entirely
    rather than installed at level 0, so responses skip the responder altogether instead of
    being buffered and re-emitted as a stored-only gzip stream. Deployments behind a proxy that
    already compresses (nginx, Caddy) want this, both to avoid the duplicated work and because
    the proxy can compress off the event loop.
    """
    if compresslevel <= 0:
        return
    app.add_middleware(ContentTypeAwareGZipMiddleware, minimum_size=GZIP_MINIMUM_SIZE, compresslevel=compresslevel)


class RedirectRootWithQueryStringMiddleware(BaseHTTPMiddleware):
    """When a request is made to the root path with a query string, redirect to the root path without the query string.

    For example, to force a Gradio app to use dark mode, users may append `?__theme=dark` to the URL. Their browser may
    have this query string saved in history or a bookmark, so when the user navigates to `http://127.0.0.1:9090/`, the
    browser takes them to `http://127.0.0.1:9090/?__theme=dark`.

    This breaks the static file serving in the UI, so we redirect the user to the root path without the query string.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        # Be root_path-aware: behind a sub-path proxy, `scope["path"]` retains the public prefix
        # (see SubPathASGIMiddleware), so the app root is `<root_path>/`, not just `/`.
        root_path = request.scope.get("root_path", "")
        if request.url.path in ("/", root_path + "/") and request.url.query:
            return RedirectResponse(url=root_path + "/")

        response = await call_next(request)
        return response


def _identify_video_upload_user(scope: Scope) -> tuple[bool, str | None]:
    """Identify the uploader for auth and per-user slot accounting.

    Returns (authenticated, per_user_key). In single-user mode there is only one
    tenant, so per_user_key is None — the whole global capacity is theirs and no
    per-user quota applies.
    """
    if not ApiDependencies.invoker.services.configuration.multiuser:
        return True, None

    scheme, token = get_authorization_scheme_param(Headers(scope=scope).get("authorization"))
    if scheme.lower() != "bearer":
        return False, None
    from invokeai.app.services.auth.token_service import verify_token

    token_data = verify_token(token)
    if token_data is None:
        return False, None
    from invokeai.app.api.auth_dependencies import resolve_authorized_user

    if resolve_authorized_user(token_data) is None:
        return False, None
    return True, token_data.user_id


async def _identify_video_upload_user_async(scope: Scope) -> tuple[bool, str | None]:
    return await run_in_threadpool(_identify_video_upload_user, scope)


class VideoUploadLimitASGIMiddleware:
    """Bound video-upload ingress *before* FastAPI's multipart parser runs.

    The upload route's own MAX_UPLOAD_SIZE check only fires after the multipart body has
    been fully parsed (and spooled to temp storage), so oversized, chunked, or many
    concurrent uploads could exhaust temp space before ever being rejected. This
    middleware rejects oversized requests from the Content-Length header, aborts
    chunked bodies that exceed the cap mid-stream, and bounds concurrent uploads
    both globally and per user (so one tenant's slow uploads cannot starve the
    others into 429s).
    """

    def __init__(
        self,
        app: ASGIApp,
        max_body_bytes: int,
        max_concurrent: int,
        max_concurrent_per_user: int | None = None,
        identify_user: Callable[[Scope], tuple[bool, str | None] | Awaitable[tuple[bool, str | None]]] | None = None,
        idle_timeout_seconds: float = 120.0,
        max_upload_duration_seconds: float = 30 * 60,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_concurrent = max_concurrent
        self.max_concurrent_per_user = max_concurrent_per_user
        self.identify_user = identify_user
        self.idle_timeout_seconds = idle_timeout_seconds
        self.max_upload_duration_seconds = max_upload_duration_seconds
        self._active = 0
        self._active_by_user: dict[str, int] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # Behind a sub-path proxy the public path carries the prefix (see SubPathASGIMiddleware).
        route_path: str = scope.get("path", "").removeprefix(scope.get("root_path", ""))
        if not (scope.get("method") == "POST" and route_path == "/api/v1/videos/upload"):
            return await self.app(scope, receive, send)

        per_user_key: str | None = None
        if self.identify_user is not None:
            identity = self.identify_user(scope)
            if inspect.isawaitable(identity):
                identity = await identity
            authenticated, per_user_key = identity
            if not authenticated:
                response = JSONResponse(
                    {"detail": "Authentication required"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                return await response(scope, receive, send)

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None and content_length.isdigit() and int(content_length) > self.max_body_bytes:
            response = JSONResponse(
                {"detail": f"Video upload exceeds maximum request size ({self.max_body_bytes} bytes)"},
                status_code=413,
            )
            return await response(scope, receive, send)

        if self._active >= self.max_concurrent:
            response = JSONResponse(
                {"detail": "Too many concurrent video uploads; try again shortly"},
                status_code=429,
                headers={"Retry-After": "5"},
            )
            return await response(scope, receive, send)

        if (
            per_user_key is not None
            and self.max_concurrent_per_user is not None
            and self._active_by_user.get(per_user_key, 0) >= self.max_concurrent_per_user
        ):
            response = JSONResponse(
                {"detail": "Too many concurrent video uploads for this user; try again shortly"},
                status_code=429,
                headers={"Retry-After": "5"},
            )
            return await response(scope, receive, send)

        self._active += 1
        if per_user_key is not None:
            self._active_by_user[per_user_key] = self._active_by_user.get(per_user_key, 0) + 1
        received = 0
        upload_started_at = asyncio.get_running_loop().time()

        async def limited_receive() -> Message:
            # Backstop for clients that omit Content-Length (chunked transfer): count the
            # streamed body and abort the request once it exceeds the cap, so the multipart
            # parser stops spooling. A clean 413 isn't possible mid-parse; the aborted
            # request surfaces to the client as a dropped connection.
            nonlocal received
            remaining_duration = self.max_upload_duration_seconds - (
                asyncio.get_running_loop().time() - upload_started_at
            )
            if remaining_duration <= 0:
                return {"type": "http.disconnect"}
            try:
                message = await asyncio.wait_for(receive(), timeout=min(self.idle_timeout_seconds, remaining_duration))
            except TimeoutError:
                return {"type": "http.disconnect"}
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    return {"type": "http.disconnect"}
            return message

        try:
            await self.app(scope, limited_receive, send)
        finally:
            self._active -= 1
            if per_user_key is not None:
                remaining = self._active_by_user.get(per_user_key, 0) - 1
                if remaining > 0:
                    self._active_by_user[per_user_key] = remaining
                else:
                    self._active_by_user.pop(per_user_key, None)


class SubPathASGIMiddleware:
    """Make the app work behind a reverse proxy that serves it under a sub-path (e.g. `/invoke`).

    Handles both common proxy styles:
      - The proxy STRIPS the sub-path: the incoming path is already `/api/...`. We restore the prefix
        so `scope["path"]` is the public path, and advertise it via `root_path`.
      - The proxy PRESERVES the sub-path: the incoming path is already `/invoke/api/...`. We leave the
        path as-is and advertise the prefix via `root_path`.

    Either way `scope["path"]` ends up holding the full public path (prefix included), as the ASGI
    spec requires. Starlette's router strips `root_path` for route matching (via `get_route_path`)
    and builds redirect/`url_for` URLs from the full path, so server-generated redirects keep the
    prefix.

    Note: we deliberately do NOT use uvicorn's `root_path`, because uvicorn also PREPENDS it to the
    request path, which would double the prefix for the preserve-style proxy. Doing the rewrite here
    covers both styles.
    """

    def __init__(self, app: ASGIApp, base_path: str) -> None:
        self.app = app
        self.base_path = base_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            scope = dict(scope)
            path: str = scope.get("path", "")
            # Strip-style proxy: the prefix is missing, so restore it. Per the ASGI spec `scope["path"]`
            # must include `root_path`; Starlette strips it again for matching and uses the full path
            # for redirects/url_for, keeping the prefix on server-generated redirects.
            if not (path == self.base_path or path.startswith(self.base_path + "/")):
                scope["path"] = self.base_path + path
                raw_path = scope.get("raw_path")
                if raw_path is not None:
                    scope["raw_path"] = self.base_path.encode("latin-1") + raw_path
            # Advertise the public prefix for both styles (openapi servers, /docs, redirects).
            scope["root_path"] = self.base_path
        await self.app(scope, receive, send)


# Add the middleware
app.add_middleware(RedirectRootWithQueryStringMiddleware)
app.add_middleware(SlidingWindowTokenMiddleware)
app.add_middleware(
    VideoUploadLimitASGIMiddleware,
    max_body_bytes=videos.MAX_UPLOAD_REQUEST_SIZE,
    max_concurrent=videos.MAX_CONCURRENT_VIDEO_UPLOADS,
    max_concurrent_per_user=videos.MAX_CONCURRENT_VIDEO_UPLOADS_PER_USER,
    identify_user=_identify_video_upload_user_async,
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
    expose_headers=["X-Refreshed-Token"],
)

configure_gzip(app, app_config.http_compression_level)


# Include all routers
# Authentication router should be first so it's registered before protected routes
app.include_router(auth.auth_router, prefix="/api")
app.include_router(utilities.utilities_router, prefix="/api")
app.include_router(model_manager.model_manager_router, prefix="/api")
app.include_router(image_moves.image_moves_router, prefix="/api")
app.include_router(images.images_router, prefix="/api")
app.include_router(videos.videos_router, prefix="/api")
app.include_router(gallery.gallery_router, prefix="/api")
app.include_router(boards.boards_router, prefix="/api")
app.include_router(board_images.board_images_router, prefix="/api")
app.include_router(virtual_boards.virtual_boards_router, prefix="/api")
app.include_router(model_relationships.model_relationships_router, prefix="/api")
app.include_router(app_info.app_router, prefix="/api")
app.include_router(session_queue.session_queue_router, prefix="/api")
app.include_router(workflows.workflows_router, prefix="/api")
app.include_router(style_presets.style_presets_router, prefix="/api")
app.include_router(system_prompts.system_prompts_router, prefix="/api")
app.include_router(client_state.client_state_router, prefix="/api")
app.include_router(recall_parameters.recall_parameters_router, prefix="/api")
app.include_router(custom_nodes.custom_nodes_router, prefix="/api")

app.openapi = get_openapi_func(app)


@app.get("/docs", include_in_schema=False)
def overridden_swagger(request: Request) -> HTMLResponse:
    # Prefix with the proxy root_path (if any) so the schema resolves under a sub-path deployment.
    root_path = request.scope.get("root_path", "")
    return get_swagger_ui_html(
        openapi_url=f"{root_path}{app.openapi_url}",  # type: ignore [arg-type] # this is always a string
        title=f"{app.title} - Swagger UI",
        swagger_favicon_url="static/docs/invoke-favicon-docs.svg",
    )


@app.get("/redoc", include_in_schema=False)
def overridden_redoc(request: Request) -> HTMLResponse:
    # Prefix with the proxy root_path (if any) so the schema resolves under a sub-path deployment.
    root_path = request.scope.get("root_path", "")
    return get_redoc_html(
        openapi_url=f"{root_path}{app.openapi_url}",  # type: ignore [arg-type] # this is always a string
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
