import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route


def _frontend(body: str):
    async def app(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/html")]})
        await send({"type": "http.response.body", "body": body.encode()})

    return app


def _make_app(frontend_root: Path) -> tuple[FastAPI, Mount, object]:
    app = FastAPI()
    core_router = APIRouter(prefix="/api")

    async def core_route() -> PlainTextResponse:
        return PlainTextResponse("core route")

    async def core_dependency() -> str:
        return "core dependency"

    async def protected_route(value: str = Depends(core_dependency)) -> PlainTextResponse:
        return PlainTextResponse(value)

    core_router.add_api_route("/core", core_route, methods=["GET"])
    core_router.add_api_route("/protected", protected_route, methods=["GET"])
    app.include_router(core_router)
    ui_route = Mount("/", app=_frontend(frontend_root.read_text()), name="ui")
    app.router.routes.append(ui_route)
    return app, ui_route, core_dependency


def test_custom_node_routes_cannot_hijack_core_or_frontend(tmp_path: Path) -> None:
    core_root = tmp_path / "core.html"
    core_root.write_text("core frontend")
    evil_root = tmp_path / "evil.html"
    evil_root.write_text("evil frontend")

    async def exercise() -> None:
        app, ui_route, _ = _make_app(core_root)

        async def evil_route(request) -> PlainTextResponse:
            return PlainTextResponse("evil route")

        async def evil_endpoint(request) -> PlainTextResponse:
            return PlainTextResponse("evil endpoint")

        # A node pack can mutate the live router and replace the mounted UI after startup.
        app.router.routes.insert(0, Route("/api/core", evil_route))
        app.router.routes.insert(1, Route("/api/pwn", evil_endpoint))
        ui_route.app = _frontend(evil_root.read_text())

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            assert (await client.get("/api/core")).text == "evil route"
            assert (await client.get("/api/pwn")).text == "evil endpoint"
            assert (await client.get("/")).text == "evil frontend"

        from invokeai.app.api.trusted_routes import (
            TrustedRouteGuard,
            get_original_frontend,
            snapshot_core_routes,
        )

        guarded_app, guarded_ui_route, core_dependency = _make_app(core_root)
        trusted_routes = snapshot_core_routes(guarded_app)
        trusted_frontend = get_original_frontend(guarded_app)
        trusted_schema = guarded_app.openapi()
        guarded_app.openapi = lambda: {"evil": True}
        guarded_app.openapi_url = None

        class EvilMiddleware:
            def __init__(self, app) -> None:
                self.app = app

            async def __call__(self, scope, receive, send) -> None:
                if scope.get("path") == "/api/core":
                    await send({"type": "http.response.start", "status": 200, "headers": []})
                    await send({"type": "http.response.body", "body": b"evil middleware"})
                    return
                await self.app(scope, receive, send)

        guarded_app.add_middleware(EvilMiddleware)
        guarded_app.router.routes.insert(0, Route("/api/core", evil_route))
        guarded_app.router.routes.insert(1, Route("/api/pwn", evil_endpoint))
        guarded_ui_route.app = _frontend(evil_root.read_text())

        async def evil_dependency() -> str:
            return "evil dependency"

        guarded_app.dependency_overrides[core_dependency] = evil_dependency
        guarded_app.openapi_schema = None

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded_app), base_url="http://testserver"
        ) as client:
            assert (await client.get("/api/protected")).text == "evil dependency"
            assert (await client.get("/openapi.json")).json() == {"evil": True}

        guarded = TrustedRouteGuard(app=guarded_app, routes=trusted_routes, frontend=trusted_frontend)
        guarded_app.router.routes.insert(0, Route("/api/runtime", evil_endpoint))
        guarded_ui_route.app = _frontend("runtime evil frontend")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            assert (await client.get("/api/core")).text == "core route"
            assert (await client.get("/api/protected")).text == "core dependency"
            assert (await client.get("/api/pwn")).status_code == 404
            assert (await client.post("/api/pwn")).status_code == 404
            assert (await client.options("/api/pwn")).status_code == 404
            assert (await client.get("/api/runtime")).status_code == 404
            assert (await client.get("/")).text == "core frontend"
            assert (await client.get("/projects/new")).text == "core frontend"
            schema = (await client.get("/openapi.json")).json()
            assert schema["paths"] == trusted_schema["paths"]
            assert "evil" not in schema

    asyncio.run(exercise())


def test_trusted_static_files_snapshot_serves_original_directories(tmp_path: Path) -> None:
    from invokeai.app.api.no_cache_staticfiles import NoCacheStaticFiles
    from invokeai.app.api.trusted_routes import TrustedRouteGuard, get_original_frontend, snapshot_core_routes

    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    (trusted_root / "index.html").write_text("trusted index")
    (trusted_root / "asset.js").write_text("trusted asset")
    docs_root = tmp_path / "trusted-docs"
    docs_root.mkdir()
    (docs_root / "guide.css").write_text("trusted docs")
    evil_root = tmp_path / "evil"
    evil_root.mkdir()
    (evil_root / "index.html").write_text("evil index")
    (evil_root / "asset.js").write_text("evil asset")
    (evil_root / "guide.css").write_text("evil docs")
    late_root = tmp_path / "late"
    late_root.mkdir()
    (late_root / "index.html").write_text("late index")
    (late_root / "asset.js").write_text("late asset")
    (late_root / "guide.css").write_text("late docs")

    app = FastAPI()
    ui = NoCacheStaticFiles(directory=trusted_root, html=True)
    static = NoCacheStaticFiles(directory=docs_root)
    ui_mount = Mount("/", app=ui, name="ui")
    static_mount = Mount("/static", app=static, name="static")
    app.router.routes.extend([static_mount, ui_mount])
    routes = snapshot_core_routes(app)
    trusted_frontend = get_original_frontend(app)

    # Mutation after capture but before guard construction must not affect either mount.
    ui.all_directories[:] = [evil_root]
    ui.html = False
    static.all_directories[:] = [evil_root]
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=trusted_frontend)

    # Nor may later edits to the original handlers redirect captured serving.
    ui.all_directories[:] = [late_root]
    ui.html = False
    static.all_directories[:] = [late_root]

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:

            async def request(path: str) -> httpx.Response:
                # Keep the loop polling while AnyIO's worker thread runs; this runner can miss its wakeup.
                task = asyncio.create_task(client.get(path))
                while not task.done():
                    await asyncio.sleep(0)
                return await task

            for path, expected in [
                ("/", "trusted index"),
                ("/asset.js", "trusted asset"),
                ("/projects/new", "trusted index"),
                ("/static/guide.css", "trusted docs"),
            ]:
                response = await request(path)
                assert response.status_code == 200
                assert response.text == expected
                assert "no-store" in response.headers["cache-control"]
                assert response.headers["pragma"] == "no-cache"
                assert response.headers["expires"] == "0"

    asyncio.run(exercise())


def test_trusted_middleware_snapshot_detaches_descriptors_and_options() -> None:
    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()

    class OptionsMiddleware:
        def __init__(self, app, *, options: dict[str, list[str]]) -> None:
            self.app = app
            self.options = options

        async def __call__(self, scope, receive, send) -> None:
            scope.setdefault("state", {}).setdefault("order", []).append(self.options["order"][0])
            await self.app(scope, receive, send)

    class OuterMiddleware:
        def __init__(self, app) -> None:
            self.app = app

        async def __call__(self, scope, receive, send) -> None:
            scope.setdefault("state", {}).setdefault("order", []).append("outer")
            await self.app(scope, receive, send)

    class EvilMiddleware:
        def __init__(self, app, *, options) -> None:
            self.app = app
            self.options = options

        async def __call__(self, scope, receive, send) -> None:
            scope.setdefault("state", {}).setdefault("order", []).append(f"evil:{self.options['order'][0]}")
            await self.app(scope, receive, send)

    @app.get("/api/middleware")
    async def middleware_route(request: Request) -> PlainTextResponse:
        return PlainTextResponse(",".join(request.state.order))

    @app.get("/api/error")
    async def error_route() -> None:
        raise HTTPException(status_code=418)

    @app.exception_handler(HTTPException)
    async def handle_http_exception(_request, exc: HTTPException) -> PlainTextResponse:
        return PlainTextResponse("handled", status_code=exc.status_code)

    options = {"order": ["captured"]}
    app.add_middleware(OptionsMiddleware, options=options)
    app.add_middleware(OuterMiddleware)
    routes = snapshot_core_routes(app)

    descriptor = next(middleware for middleware in app.user_middleware if middleware.cls is OptionsMiddleware)
    descriptor.cls = EvilMiddleware
    descriptor.kwargs["options"]["order"][0] = "changed"
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=None)

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/middleware")
            assert response.status_code == 200
            assert response.text == "outer,captured"

            response = await client.get("/api/error")
            assert response.status_code == 418
            assert response.text == "handled"

    asyncio.run(exercise())


def test_trusted_route_snapshot_detaches_method_sets() -> None:
    from fastapi.routing import APIRoute

    from invokeai.app.api.trusted_routes import snapshot_core_routes

    app = FastAPI()

    @app.get("/api/methods")
    async def methods_route() -> PlainTextResponse:
        return PlainTextResponse("core")

    route = next(route for route in app.router.routes if isinstance(route, APIRoute) and route.path == "/api/methods")
    routes = snapshot_core_routes(app)
    route.methods.add("DELETE")

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=routes.app), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/methods")
            assert response.status_code == 200
            assert response.text == "core"

            response = await client.delete("/api/methods")
            assert response.status_code == 405
            assert response.headers["allow"] == "GET"

    asyncio.run(exercise())


def test_trusted_route_snapshot_detaches_path_convertors() -> None:
    from fastapi.routing import APIRoute

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()

    @app.get("/api/convert/{value}")
    async def convert_route(value: str) -> PlainTextResponse:
        return PlainTextResponse(value)

    route = next(
        route for route in app.router.routes if isinstance(route, APIRoute) and route.path == "/api/convert/{value}"
    )
    routes = snapshot_core_routes(app)
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=None)

    class ChangedConvertor:
        def convert(self, _value: str) -> str:
            return "changed"

    route.param_convertors["value"] = ChangedConvertor()

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/convert/123")
            assert response.status_code == 200
            assert response.text == "123"

    asyncio.run(exercise())


def test_trusted_direct_route_rebinds_dependency_handler() -> None:
    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()

    async def dependency() -> str:
        return "captured"

    @app.get("/api/direct")
    async def direct_route(value: str = Depends(dependency)) -> PlainTextResponse:
        return PlainTextResponse(value)

    routes = snapshot_core_routes(app)
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=None)

    async def late_dependency() -> str:
        return "late override"

    app.dependency_overrides[dependency] = late_dependency

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/direct")
            assert response.status_code == 200
            assert response.text == "captured"

    asyncio.run(exercise())


def test_trusted_direct_websocket_route_rebinds_dependency_handler() -> None:
    from fastapi import WebSocket

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()

    async def dependency() -> str:
        return "captured"

    @app.websocket("/api/socket")
    async def socket_route(websocket: WebSocket, value: str = Depends(dependency)) -> None:
        await websocket.accept()
        await websocket.send_text(value)
        await websocket.close()

    routes = snapshot_core_routes(app)
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=_frontend("trusted UI"))

    async def late_dependency() -> str:
        return "late override"

    app.dependency_overrides[dependency] = late_dependency

    async def exercise() -> None:
        messages: list[dict] = []

        async def receive() -> dict[str, str]:
            return {"type": "websocket.connect"}

        async def send(message: dict) -> None:
            messages.append(message)

        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
            "path": "/api/socket",
            "raw_path": b"/api/socket",
            "query_string": b"",
            "headers": [],
            "subprotocols": [],
            "state": {},
        }
        await guarded(scope, receive, send)
        assert [message["text"] for message in messages if message["type"] == "websocket.send"] == ["captured"]

    asyncio.run(exercise())


def test_trusted_included_websocket_route_keeps_captured_dependency_override() -> None:
    from fastapi import WebSocket

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()
    router = APIRouter()

    async def dependency() -> str:
        return "default"

    async def override() -> str:
        return "captured override"

    @router.websocket("/socket")
    async def socket_route(websocket: WebSocket, value: str = Depends(dependency)) -> None:
        await websocket.accept()
        await websocket.send_text(value)
        await websocket.close()

    app.dependency_overrides[dependency] = override
    app.include_router(router, prefix="/api")
    guarded = TrustedRouteGuard(app=app, routes=snapshot_core_routes(app), frontend=None)

    async def late_override() -> str:
        return "late override"

    app.dependency_overrides[dependency] = late_override

    async def exercise(target) -> list[dict]:
        messages: list[dict] = []

        async def receive() -> dict[str, str]:
            return {"type": "websocket.connect"}

        async def send(message: dict) -> None:
            messages.append(message)

        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
            "path": "/api/socket",
            "raw_path": b"/api/socket",
            "query_string": b"",
            "headers": [],
            "subprotocols": [],
            "state": {},
        }
        await target(scope, receive, send)
        return messages

    async def exercise_both() -> None:
        original_messages = await exercise(app)
        guarded_messages = await exercise(guarded)
        original_values = [message["text"] for message in original_messages if message["type"] == "websocket.send"]
        guarded_values = [message["text"] for message in guarded_messages if message["type"] == "websocket.send"]
        assert original_values == ["late override"]
        assert guarded_values == ["captured override"]

    asyncio.run(exercise_both())


def test_trusted_guard_preserves_api_resolution_and_ui_fallback() -> None:
    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()
    router = APIRouter(prefix="/api/v1")

    @router.get("/boards/")
    async def boards() -> PlainTextResponse:
        return PlainTextResponse("boards")

    app.include_router(router)

    async def exercise() -> None:
        for frontend in (None, _frontend("trusted SPA")):
            routes = snapshot_core_routes(app)
            guarded = TrustedRouteGuard(app=app, routes=routes, frontend=frontend)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=guarded), base_url="http://testserver", follow_redirects=False
            ) as client:
                redirect = await client.get("/api/v1/boards")
                assert redirect.status_code == 307
                assert redirect.headers["location"].endswith("/api/v1/boards/")

                wrong_method = await client.delete("/api/v1/boards/")
                assert wrong_method.status_code == 405
                assert wrong_method.headers["allow"] == "GET"

                for path in ("/api", "/api/missing"):
                    for method in ("GET", "POST", "OPTIONS"):
                        response = await client.request(method, path)
                        assert response.status_code == 404
                        assert "text/html" not in response.headers.get("content-type", "")

                if frontend is not None:
                    for path in ("/projects/new", "/apix"):
                        response = await client.get(path)
                        assert response.status_code == 200
                        assert response.text == "trusted SPA"

                    # root_path must be stripped before deciding whether the UI mount is an API fallback.
                    rooted = httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=guarded, root_path="/invoke"),
                        base_url="http://testserver",
                        follow_redirects=False,
                    )
                    async with rooted:
                        response = await rooted.get("/invoke/api/missing")
                        assert response.status_code == 404
                        response = await rooted.get("/invoke/apix")
                        assert response.status_code == 200
                        assert response.text == "trusted SPA"

    asyncio.run(exercise())


def test_trusted_nested_websocket_routes_keep_their_prefix_clone(monkeypatch) -> None:
    import weakref

    from fastapi import WebSocket
    from fastapi.routing import APIWebSocketRoute

    import invokeai.app.api.trusted_routes as trusted_routes

    app = FastAPI()
    for prefix, value in (("/route-a/nested", "socket a"), ("/route-b/nested", "socket b")):
        router = APIRouter()

        @router.websocket("/socket")
        async def socket_route(websocket: WebSocket, value: str = value) -> None:
            await websocket.accept()
            await websocket.send_text(value)
            await websocket.close()

        app.include_router(router, prefix=prefix)

    # Reuse synthetic ids only after weakly referenced source routes are collected.
    original_id = id

    slots: list[tuple[int, weakref.ReferenceType[APIWebSocketRoute]]] = []
    next_id = 1 << 100

    def route_id(value: object) -> int:
        nonlocal next_id
        if isinstance(value, APIWebSocketRoute) and value.path != "/socket":
            for index, (slot_id, reference) in enumerate(slots):
                if reference() is value:
                    return slot_id
                if reference() is None:
                    slots[index] = (slot_id, weakref.ref(value))
                    return slot_id
            slots.append((next_id, weakref.ref(value)))
            next_id += 1
            return slots[-1][0]
        return original_id(value)

    monkeypatch.setattr(
        trusted_routes,
        "id",
        route_id,
        raising=False,
    )
    guarded = trusted_routes.TrustedRouteGuard(app=app, routes=trusted_routes.snapshot_core_routes(app), frontend=None)

    async def exercise(path: str) -> list[dict]:
        messages: list[dict] = []

        async def receive() -> dict[str, str]:
            return {"type": "websocket.connect"}

        async def send(message: dict) -> None:
            messages.append(message)

        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": "",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "subprotocols": [],
            "state": {},
        }
        await guarded(scope, receive, send)
        return messages

    async def exercise_both() -> None:
        for path, expected in (
            ("/route-a/nested/socket", "socket a"),
            ("/route-b/nested/socket", "socket b"),
        ):
            messages = await exercise(path)
            assert any(message["type"] == "websocket.accept" for message in messages)
            assert [message["text"] for message in messages if message["type"] == "websocket.send"] == [expected]

    asyncio.run(exercise_both())


def test_trusted_direct_route_detaches_dependant_graph() -> None:
    from fastapi.routing import APIRoute

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()

    async def dependency() -> str:
        return "captured"

    @app.get("/api/dependant")
    async def dependant_route(value: str = Depends(dependency)) -> PlainTextResponse:
        return PlainTextResponse(value)

    route = next(route for route in app.router.routes if isinstance(route, APIRoute) and route.path == "/api/dependant")
    routes = snapshot_core_routes(app)
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=None)

    async def late_dependency() -> str:
        return "late dependant"

    route.dependant.dependencies[0].call = late_dependency

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/dependant")
            assert response.status_code == 200
            assert response.text == "captured"

    asyncio.run(exercise())


def test_trusted_direct_route_detaches_model_field_configuration() -> None:
    from fastapi.routing import APIRoute

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()

    @app.get("/api/query")
    async def query_route(value: str = Query()) -> PlainTextResponse:
        return PlainTextResponse(value)

    route = next(route for route in app.router.routes if isinstance(route, APIRoute) and route.path == "/api/query")
    routes = snapshot_core_routes(app)
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=None)
    route.dependant.query_params[0].field_info.default = "late"

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/query")
            assert response.status_code == 422

    asyncio.run(exercise())


def test_trusted_nested_include_context_detaches_dependencies() -> None:
    from fastapi.routing import _IncludedRouter

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()
    nested_router = APIRouter()

    @nested_router.get("/context")
    async def context_route() -> PlainTextResponse:
        return PlainTextResponse("captured")

    root_router = APIRouter()
    root_router.include_router(nested_router, prefix="/nested")
    app.include_router(root_router, prefix="/api")
    root_include = next(route for route in app.router.routes if isinstance(route, _IncludedRouter))
    nested_include = next(route for route in root_include.original_router.routes if isinstance(route, _IncludedRouter))
    routes = snapshot_core_routes(app)
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=None)

    async def late_dependency(required: str = Header()) -> None:
        return None

    nested_include.include_context.dependencies.append(Depends(late_dependency))

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/nested/context")
            assert response.status_code == 200
            assert response.text == "captured"

    asyncio.run(exercise())


def test_trusted_nested_include_context_detaches_dependency_descriptors() -> None:
    from fastapi import Security
    from fastapi.routing import _IncludedRouter
    from fastapi.security import SecurityScopes

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()
    nested_router = APIRouter()

    async def captured_dependency(request: Request, scopes: SecurityScopes) -> None:
        request.state.value = ",".join(scopes.scopes)

    @nested_router.get("/context")
    async def context_route(request: Request) -> PlainTextResponse:
        return PlainTextResponse(request.state.value)

    root_router = APIRouter()
    root_router.include_router(
        nested_router,
        prefix="/nested",
        dependencies=[Security(captured_dependency, scopes=["captured"])],
    )
    app.include_router(root_router, prefix="/api")
    root_include = next(route for route in app.router.routes if isinstance(route, _IncludedRouter))
    nested_include = next(route for route in root_include.original_router.routes if isinstance(route, _IncludedRouter))
    routes = snapshot_core_routes(app)
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=None)

    nested_include.include_context.dependencies[0].scopes.append("late")

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/nested/context")
            assert response.status_code == 200
            assert response.text == "captured"

    asyncio.run(exercise())


def test_trusted_default_response_class_placeholder_is_captured() -> None:
    from fastapi.routing import APIRoute

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI(servers=[{"url": "/captured"}])

    @app.get("/api/default-response")
    async def default_response() -> dict[str, str]:
        return {"value": "captured"}

    route = next(
        route for route in app.router.routes if isinstance(route, APIRoute) and route.path == "/api/default-response"
    )
    routes = snapshot_core_routes(app)
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=None)
    route.response_class.value = PlainTextResponse
    app.servers[0]["url"] = "/late"

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/default-response")
            assert response.status_code == 200
            assert response.json() == {"value": "captured"}

            schema = (await client.get("/openapi.json")).json()
            content = schema["paths"]["/api/default-response"]["get"]["responses"]["200"]["content"]
            assert "application/json" in content
            assert schema["servers"] == [{"url": "/captured"}]

    asyncio.run(exercise())


def test_trusted_openapi_includes_request_root_path_server() -> None:
    app = FastAPI(servers=[{"url": "/configured"}])

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    guarded = TrustedRouteGuard(app=app, routes=snapshot_core_routes(app), frontend=None)

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded, root_path="/invoke"),
            base_url="http://testserver",
        ) as client:
            schema = (await client.get("/invoke/openapi.json")).json()
            assert schema["servers"] == [{"url": "/invoke"}, {"url": "/configured"}]

    asyncio.run(exercise())


def test_trusted_guard_runs_router_lifespan_once() -> None:
    from fastapi.testclient import TestClient

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    events: list[str] = []

    @asynccontextmanager
    async def lifespan(_app):
        events.append("start")
        yield
        events.append("stop")

    app = FastAPI(lifespan=lifespan)
    guarded = TrustedRouteGuard(app=app, routes=snapshot_core_routes(app), frontend=None)

    with TestClient(guarded):
        assert events == ["start"]

    assert events == ["start", "stop"]


def test_trusted_included_route_detaches_model_field_configuration() -> None:
    from fastapi.routing import APIRoute

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()
    included = APIRouter()

    @included.get("/query")
    async def query_route(value: str = Query()) -> PlainTextResponse:
        return PlainTextResponse(value)

    app.include_router(included, prefix="/api")
    route = next(route for route in included.routes if isinstance(route, APIRoute) and route.path == "/query")
    guarded = TrustedRouteGuard(app=app, routes=snapshot_core_routes(app), frontend=None)
    route.dependant.query_params[0].field_info.default = "late"

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/query")
            assert response.status_code == 422

    asyncio.run(exercise())


def test_trusted_openapi_cache_is_not_shared_with_mutable_app() -> None:
    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    guarded = TrustedRouteGuard(app=app, routes=snapshot_core_routes(app), frontend=None)

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=guarded), base_url="http://testserver"
        ) as client:
            first_schema = (await client.get("/openapi.json")).json()
            assert "/api/health" in first_schema["paths"]
            app.openapi_schema["paths"]["/api/evil"] = {}

            second_schema = (await client.get("/openapi.json")).json()
            assert "/api/evil" not in second_schema["paths"]

    asyncio.run(exercise())


def test_trusted_guard_detaches_included_router_lifespan_handlers() -> None:
    from fastapi.testclient import TestClient

    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    events: list[str] = []

    @asynccontextmanager
    async def root_lifespan(_app):
        events.append("root start")
        yield
        events.append("root stop")

    included = APIRouter()

    async def captured_startup() -> None:
        events.append("included start")

    async def captured_shutdown() -> None:
        events.append("included stop")

    included.on_startup.append(captured_startup)
    included.on_shutdown.append(captured_shutdown)
    app = FastAPI(lifespan=root_lifespan)
    app.include_router(included, prefix="/api")
    guarded = TrustedRouteGuard(app=app, routes=snapshot_core_routes(app), frontend=None)

    async def late_startup() -> None:
        events.append("late start")

    async def late_shutdown() -> None:
        events.append("late stop")

    included.on_startup.append(late_startup)
    included.on_shutdown.append(late_shutdown)

    with TestClient(guarded):
        assert events == ["root start", "included start"]

    assert events == ["root start", "included start", "included stop", "root stop"]


def test_trusted_socketio_mount_detaches_dispatcher_configuration(tmp_path: Path) -> None:
    from invokeai.app.api.sockets import SocketIO
    from invokeai.app.api.trusted_routes import TrustedRouteGuard, snapshot_core_routes

    app = FastAPI()
    socket_io = SocketIO(app)
    routes = snapshot_core_routes(app)
    exposed_file = tmp_path / "exposed.txt"
    exposed_file.write_text("late static route")
    socket_io._app.static_files["/ws/exposed.txt"] = {
        "filename": str(exposed_file),
        "content_type": "text/plain",
    }
    custom_app = FastAPI()

    @custom_app.get("/custom")
    async def custom_route() -> PlainTextResponse:
        return PlainTextResponse("exposed")

    late_app = FastAPI()

    @late_app.get("/custom")
    async def late_custom_route() -> PlainTextResponse:
        return PlainTextResponse("late exposure")

    # Simulate a node changing Engine.IO's fallback dispatcher after core routes were captured.
    socket_io._app.other_asgi_app = custom_app
    guarded = TrustedRouteGuard(app=app, routes=routes, frontend=None)

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=guarded)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            sid: str | None = None
            try:
                handshake = await client.get("/ws/socket.io/", params={"EIO": "4", "transport": "polling"})
                assert handshake.status_code == 200
                open_packet = json.loads(handshake.text[1:])
                sid = open_packet["sid"]
                assert isinstance(sid, str) and sid
                assert sid in socket_io._sio.eio.sockets

                original_transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=original_transport, base_url="http://testserver") as original:
                    exposed = await original.get("/ws/exposed.txt")
                    assert exposed.status_code == 200
                    assert exposed.text == "late static route"

                response = await client.get("/ws/custom")
                assert response.status_code == 404
                assert (await client.get("/ws/exposed.txt")).status_code == 404

                # Reassigning the original dispatcher after construction must not change the guard.
                socket_io._app.other_asgi_app = late_app
                assert (await client.get("/ws/custom")).status_code == 404
            finally:
                if sid is not None:
                    closed = await client.post(
                        "/ws/socket.io/", params={"EIO": "4", "transport": "polling", "sid": sid}, content="1"
                    )
                    assert closed.status_code == 200
                await socket_io._sio.shutdown()

    asyncio.run(exercise())
