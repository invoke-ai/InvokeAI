"""Serve the core ASGI routes captured before custom nodes are loaded."""

from __future__ import annotations

import copy
from types import MethodType
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter, _FrontendRouteGroup, _IncludedRouter
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Mount, Route


class _RouteSnapshot(list[BaseRoute]):
    def __init__(self, router: APIRouter, app: FastAPI) -> None:
        super().__init__(router.routes)
        self.router = router
        self.app = app
        self.openapi_url = app.openapi_url


def _clone_router(router: APIRouter, routers: dict[int, APIRouter], routes: dict[int, BaseRoute]) -> APIRouter:
    existing = routers.get(id(router))
    if existing is not None:
        return existing

    clone = copy.copy(router)
    routers[id(router)] = clone
    # Router.app is a bound method; copy.copy keeps it bound to the live router.
    clone.app = MethodType(router.app.__func__, clone)
    clone.middleware_stack = clone.app
    clone.routes = [_clone_route(route, routers, routes) for route in router.routes]
    clone._low_priority_routes = [_clone_route(route, routers, routes) for route in router._low_priority_routes]
    clone._frontend_routes = (
        _clone_route(router._frontend_routes, routers, routes) if router._frontend_routes is not None else None
    )
    return clone


def _clone_route(route: BaseRoute, routers: dict[int, APIRouter], routes: dict[int, BaseRoute]) -> BaseRoute:
    existing = routes.get(id(route))
    if existing is not None:
        return existing

    if isinstance(route, _IncludedRouter):
        router = _clone_router(route.original_router, routers, routes)
        context = copy.copy(route.include_context)
        context.included_router = router
        clone = _IncludedRouter(original_router=router, include_context=context)
    elif isinstance(route, _FrontendRouteGroup):
        clone = copy.copy(route)
        clone.routes = [copy.copy(child) for child in route.routes]
    else:
        clone = copy.copy(route)
    routes[id(route)] = clone
    return clone


def _rebind_dependency_overrides(router: APIRouter, original_app: FastAPI, captured_app: FastAPI) -> None:
    seen: set[int] = set()

    def visit(current: APIRouter) -> None:
        if id(current) in seen:
            return
        seen.add(id(current))
        if current.dependency_overrides_provider is original_app:
            current.dependency_overrides_provider = captured_app
        for route in current.routes:
            if isinstance(route, _IncludedRouter):
                context = route.include_context
                if context.dependency_overrides_provider is original_app:
                    context.dependency_overrides_provider = captured_app
                visit(route.original_router)
            elif getattr(route, "dependency_overrides_provider", None) is original_app:
                route.dependency_overrides_provider = captured_app

    visit(router)


def snapshot_core_routes(app: FastAPI) -> list[BaseRoute]:
    """Copy route nodes and included-router links without following app or router back-references."""
    router = _clone_router(app.router, {}, {})
    core_app = copy.copy(app)
    core_app.router = router
    core_app.dependency_overrides = dict(app.dependency_overrides)
    core_app.user_middleware = list(app.user_middleware)
    core_app.exception_handlers = dict(app.exception_handlers)
    core_app.middleware_stack = None
    _rebind_dependency_overrides(router, app, core_app)
    if getattr(app.openapi, "__self__", None) is app:
        core_app.openapi = MethodType(app.openapi.__func__, core_app)
    return _RouteSnapshot(router, core_app)


def get_original_frontend(app: FastAPI) -> Any | None:
    """Return the root mount's original ASGI application."""
    return next(
        (route.app for route in app.router.routes if isinstance(route, Mount) and route.path in ("", "/")),
        None,
    )


async def _not_found(_request: Request) -> Response:
    return Response(status_code=404)


class TrustedRouteGuard:
    """Serve an immutable core route graph while retaining the app's middleware and lifespan."""

    def __init__(self, *, app: FastAPI, routes: list[BaseRoute], frontend: Any) -> None:
        if isinstance(routes, _RouteSnapshot):
            self.app = routes.app
            openapi_url = routes.openapi_url
        else:
            self.app = copy.copy(app)
            self.app.router = copy.copy(app.router)
            self.app.router.routes = routes
            self.app.user_middleware = list(app.user_middleware)
            self.app.exception_handlers = dict(app.exception_handlers)
            openapi_url = app.openapi_url
        # Root UI mounts otherwise turn unknown API requests into successful SPA responses.
        captured_routes: list[BaseRoute] = []
        for route in self.app.router.routes:
            if isinstance(route, Mount) and route.path in ("", "/"):
                continue
            if openapi_url is not None and getattr(route, "path", None) == openapi_url:
                captured_routes.append(Route(openapi_url, self._openapi, name="openapi"))
            else:
                captured_routes.append(route)
        captured_routes.extend(
            [Route("/api", _not_found, methods=[]), Route("/api/{path:path}", _not_found, methods=[])]
        )
        if frontend is not None:
            captured_routes.append(Mount("/", app=frontend, name="trusted-ui"))
        self.app.router.routes = captured_routes
        self.app.router._mark_routes_changed()
        self.app.state = app.state
        self.app.middleware_stack = None
        if getattr(self.app.openapi, "__self__", None) is self.app:
            self._generate_openapi = self.app.openapi
        else:
            from invokeai.app.util.custom_openapi import get_openapi_func

            self._generate_openapi = get_openapi_func(self.app)
        self.original_app = app

    async def _openapi(self, _request: Request) -> JSONResponse:
        # Custom node reloads invalidate the original app's cache; regenerate from this trusted route graph.
        if self.original_app.openapi_schema is None:
            self.app.openapi_schema = None
        schema = self._generate_openapi()
        self.original_app.openapi_schema = schema
        return JSONResponse(schema)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)
