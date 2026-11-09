import asyncio
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, FastAPI
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
