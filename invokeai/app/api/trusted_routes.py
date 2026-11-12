"""Serve the core ASGI routes captured before custom nodes are loaded."""

from __future__ import annotations

import copy
from dataclasses import fields
from types import MethodType
from typing import Any

from fastapi import FastAPI
from fastapi._compat import ModelField
from fastapi.datastructures import DefaultPlaceholder
from fastapi.dependencies.models import Dependant
from fastapi.params import Depends as DependsConfig
from fastapi.responses import JSONResponse
from fastapi.routing import (
    APIRoute,
    APIRouter,
    APIWebSocketRoute,
    _DefaultLifespan,
    _EffectiveRouteContext,
    _FrontendRouteGroup,
    _IncludedRouter,
    _merge_lifespan_context,
    get_websocket_app,
    request_response,
    websocket_session,
)
from pydantic.fields import FieldInfo
from socketio import ASGIApp as SocketIOASGIApp
from starlette.requests import Request
from starlette.routing import BaseRoute, Match, Mount, Route, get_route_path
from starlette.staticfiles import StaticFiles


# FastAPI 0.141.1 stores included routers and frontend routes as private lazy route nodes.
# Snapshot those nodes and rebuild handlers with FastAPI's routing factories to retain their dispatch behavior.
class _RouteSnapshot(list[BaseRoute]):
    def __init__(self, router: APIRouter, app: FastAPI) -> None:
        super().__init__(router.routes)
        self.router = router
        self.app = app
        self.openapi_url = app.openapi_url


def _copy_config(value: Any, copied: dict[int, Any] | None = None) -> Any:
    """Copy nested built-in containers while keeping services and callables shared."""
    if copied is None:
        copied = {}
    if id(value) in copied:
        return copied[id(value)]
    if isinstance(value, ModelField):
        clone = ModelField(
            field_info=_copy_config(value.field_info, copied),
            name=value.name,
            mode=value.mode,
            config=_copy_config(value.config, copied),
        )
        copied[id(value)] = clone
        return clone
    if isinstance(value, FieldInfo):
        clone = copy.copy(value)
        copied[id(value)] = clone
        for cls in reversed(type(value).__mro__):
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for attr in slots:
                if attr not in ("__dict__", "__weakref__") and hasattr(value, attr):
                    setattr(clone, attr, _copy_config(getattr(value, attr), copied))
        for attr, item in getattr(value, "__dict__", {}).items():
            setattr(clone, attr, _copy_config(item, copied))
        return clone
    if isinstance(value, (Dependant, DependsConfig)):
        clone = copy.copy(value)
        copied[id(value)] = clone
        for field in fields(value):
            attr = field.name
            object.__setattr__(clone, attr, _copy_config(getattr(value, attr), copied))
        return clone
    if isinstance(value, DefaultPlaceholder):
        clone = copy.copy(value)
        copied[id(value)] = clone
        clone.value = _copy_config(value.value, copied)
        return clone
    if type(value) is dict:
        clone = {}
        copied[id(value)] = clone
        clone.update((key, _copy_config(item, copied)) for key, item in value.items())
        return clone
    if type(value) is list:
        clone = []
        copied[id(value)] = clone
        clone.extend(_copy_config(item, copied) for item in value)
        return clone
    if type(value) is tuple:
        clone = tuple(_copy_config(item, copied) for item in value)
        copied[id(value)] = clone
        return clone
    if type(value) is set:
        clone = set()
        copied[id(value)] = clone
        clone.update(_copy_config(item, copied) for item in value)
        return clone
    if type(value) is frozenset:
        clone = frozenset(_copy_config(item, copied) for item in value)
        copied[id(value)] = clone
        return clone
    return value


def _copy_middleware(middleware: Any) -> Any:
    clone = copy.copy(middleware)
    copied: dict[int, Any] = {}
    clone.args = _copy_config(middleware.args, copied)
    clone.kwargs = _copy_config(middleware.kwargs, copied)
    return clone


def _clone_lifespan_context(
    context: Any,
    routers: dict[int, tuple[APIRouter, APIRouter]],
    routes: dict[int, tuple[BaseRoute, BaseRoute]],
) -> Any:
    if isinstance(context, _DefaultLifespan):
        return _DefaultLifespan(_clone_router(context._router, routers, routes))

    merged_context = getattr(context, "__wrapped__", None)
    code = getattr(merged_context, "__code__", None)
    if (
        code is not None
        and getattr(merged_context, "__module__", None) == "fastapi.routing"
        and code.co_name == "merged_lifespan"
    ):
        closure = getattr(merged_context, "__closure__", None) or ()
        cells = dict(zip(code.co_freevars, (cell.cell_contents for cell in closure), strict=True))
        if {"original_context", "nested_context"} <= cells.keys():
            return _merge_lifespan_context(
                _clone_lifespan_context(cells["original_context"], routers, routes),
                _clone_lifespan_context(cells["nested_context"], routers, routes),
            )
    return context


def _clone_callbacks(
    callbacks: list[BaseRoute] | None,
    routers: dict[int, tuple[APIRouter, APIRouter]],
    routes: dict[int, tuple[BaseRoute, BaseRoute]],
) -> list[BaseRoute] | None:
    return [_clone_route(callback, routers, routes) for callback in callbacks] if callbacks is not None else None


def _clone_router(
    router: APIRouter,
    routers: dict[int, tuple[APIRouter, APIRouter]],
    routes: dict[int, tuple[BaseRoute, BaseRoute]],
) -> APIRouter:
    existing = routers.get(id(router))
    if existing is not None:
        return existing[1]

    # Rebind FastAPI's default lifecycle handlers to captured routers while keeping custom lifespan callables shared.
    clone = copy.copy(router)
    routers[id(router)] = (router, clone)
    # A shallow copy retains the original bound method, so bind it to the captured router.
    clone.app = MethodType(router.app.__func__, clone)
    if getattr(router.default, "__self__", None) is router:
        clone.default = MethodType(router.default.__func__, clone)
    clone.middleware_stack = clone.app
    clone.on_startup = list(router.on_startup)
    clone.on_shutdown = list(router.on_shutdown)
    clone.lifespan_context = _clone_lifespan_context(router.lifespan_context, routers, routes)
    for attr in (
        "tags",
        "dependencies",
        "responses",
        "default_response_class",
        "generate_unique_id_function",
        "strict_content_type",
    ):
        setattr(clone, attr, _copy_config(getattr(router, attr)))
    clone.callbacks = _clone_callbacks(router.callbacks, routers, routes)
    clone.routes = [_clone_route(route, routers, routes) for route in router.routes]
    clone._low_priority_routes = [_clone_route(route, routers, routes) for route in router._low_priority_routes]
    clone._frontend_routes = (
        _clone_route(router._frontend_routes, routers, routes) if router._frontend_routes is not None else None
    )
    return clone


def _clone_route(
    route: BaseRoute,
    routers: dict[int, tuple[APIRouter, APIRouter]],
    routes: dict[int, tuple[BaseRoute, BaseRoute]],
) -> BaseRoute:
    existing = routes.get(id(route))
    if existing is not None:
        return existing[1]

    if isinstance(route, _IncludedRouter):
        router = _clone_router(route.original_router, routers, routes)
        context = copy.copy(route.include_context)
        context.included_router = router
        clone = _IncludedRouter(original_router=router, include_context=context)
        routes[id(route)] = (route, clone)
        for attr in (
            "tags",
            "dependencies",
            "responses",
            "default_response_class",
            "generate_unique_id_function",
            "strict_content_type",
        ):
            setattr(context, attr, _copy_config(getattr(context, attr)))
        context.callbacks = _clone_callbacks(route.include_context.callbacks, routers, routes) or []
    elif isinstance(route, _FrontendRouteGroup):
        clone = copy.copy(route)
        routes[id(route)] = (route, clone)
        clone.routes = []
        for child in route.routes:
            child_clone = copy.copy(child)
            child_clone.methods = set(child.methods)
            child_clone.app = _snapshot_asgi_app(child.app)
            clone.routes.append(child_clone)
        clone.dependencies = _copy_config(route.dependencies)
        clone.dependant = _copy_config(route.dependant)
    else:
        clone = copy.copy(route)
        routes[id(route)] = (route, clone)
        if hasattr(route, "param_convertors"):
            clone.param_convertors = dict(route.param_convertors)
        if isinstance(route, APIRoute):
            for attr in (
                "methods",
                "tags",
                "dependencies",
                "responses",
                "openapi_extra",
                "response_model_include",
                "response_model_exclude",
                "response_class",
                "generate_unique_id_function",
                "strict_content_type",
            ):
                setattr(clone, attr, _copy_config(getattr(route, attr)))
            clone.callbacks = _clone_callbacks(route.callbacks, routers, routes)
            clone.dependant = _copy_config(route.dependant)
            clone.body_field = _copy_config(route.body_field)
            clone.response_field = _copy_config(route.response_field)
            clone.stream_item_field = _copy_config(route.stream_item_field)
            clone.response_fields = _copy_config(route.response_fields)
        elif isinstance(route, APIWebSocketRoute):
            clone.dependencies = _copy_config(route.dependencies)
            clone.dependant = _copy_config(route.dependant)
        elif isinstance(route, Route):
            if route.methods is not None:
                clone.methods = set(route.methods)
        if isinstance(route, Mount):
            clone.app = _snapshot_asgi_app(route.app)
    routes[id(route)] = (route, clone)
    return clone


def _snapshot_asgi_app(app: Any) -> Any:
    if isinstance(app, SocketIOASGIApp):
        clone = copy.copy(app)
        clone.static_files = _copy_config(app.static_files)
        return clone
    if isinstance(getattr(app, "_app", None), SocketIOASGIApp):
        clone = copy.copy(app)
        clone._app = _snapshot_asgi_app(app._app)
        return clone
    if not isinstance(app, StaticFiles):
        return app
    clone = copy.copy(app)
    clone.all_directories = list(app.all_directories)
    clone.packages = list(app.packages) if app.packages is not None else None
    return clone


def _rebind_dependency_overrides(router: APIRouter, original_app: FastAPI, captured_app: FastAPI) -> None:
    seen: set[int] = set()

    def visit(current: APIRouter) -> None:
        if id(current) in seen:
            return
        seen.add(id(current))
        if current.dependency_overrides_provider is original_app:
            current.dependency_overrides_provider = captured_app
        candidates = [*current.routes, *current._low_priority_routes]
        if current._frontend_routes is not None:
            candidates.append(current._frontend_routes)
        for route in candidates:
            if isinstance(route, _IncludedRouter):
                context = route.include_context
                if context.dependency_overrides_provider is original_app:
                    context.dependency_overrides_provider = captured_app
                visit(route.original_router)
            elif getattr(route, "dependency_overrides_provider", None) is original_app:
                route.dependency_overrides_provider = captured_app

    visit(router)


def _rebuild_route_handlers(router: APIRouter, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    if id(router) in seen:
        return
    seen.add(id(router))

    for route in (*router.routes, *router._low_priority_routes):
        if isinstance(route, _IncludedRouter):
            _rebuild_route_handlers(route.original_router, seen)
        elif isinstance(route, APIRoute):
            route.app = request_response(route.get_route_handler())
        elif isinstance(route, APIWebSocketRoute):
            _rebuild_websocket_handler(route, router.dependency_overrides_provider)


def _rebuild_websocket_handler(route: APIWebSocketRoute, dependency_overrides_provider: Any) -> None:
    route.app = websocket_session(
        get_websocket_app(
            dependant=route.dependant,
            dependency_overrides_provider=dependency_overrides_provider,
            embed_body_fields=route._embed_body_fields,
        )
    )


def _snapshot_effective_context(
    context: _EffectiveRouteContext,
    routers: dict[int, tuple[APIRouter, APIRouter]],
    routes: dict[int, tuple[BaseRoute, BaseRoute]],
    dependency_overrides_provider: Any,
) -> _EffectiveRouteContext:
    clone = copy.copy(context)
    copied: dict[int, Any] = {}
    for field in fields(context):
        if field.name in ("original_route", "starlette_route"):
            continue
        value = getattr(context, field.name)
        if field.name == "callbacks":
            value = _clone_callbacks(value, routers, routes)
        else:
            value = _copy_config(value, copied)
        setattr(clone, field.name, value)
    if context.starlette_route is not None:
        clone.starlette_route = _clone_route(context.starlette_route, routers, routes)
        if isinstance(clone.starlette_route, APIWebSocketRoute):
            _rebuild_websocket_handler(
                clone.starlette_route,
                clone.dependency_overrides_provider
                if clone.dependency_overrides_provider is not None
                else dependency_overrides_provider,
            )
    return clone


def _snapshot_included_router_contexts(
    included: _IncludedRouter,
    routers: dict[int, tuple[APIRouter, APIRouter]],
    routes: dict[int, tuple[BaseRoute, BaseRoute]],
    seen: set[int] | None = None,
) -> None:
    if seen is None:
        seen = set()
    if id(included) in seen:
        return
    seen.add(id(included))
    candidates = included.effective_candidates()
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, _IncludedRouter):
            _snapshot_included_router_contexts(candidate, routers, routes, seen)
        else:
            candidates[index] = _snapshot_effective_context(
                candidate, routers, routes, included.include_context.dependency_overrides_provider
            )

    low_priority = included.effective_low_priority_routes()
    for index, context in enumerate(low_priority):
        low_priority[index] = _snapshot_effective_context(
            context, routers, routes, included.include_context.dependency_overrides_provider
        )


def _snapshot_included_routes(
    router: APIRouter,
    routers: dict[int, tuple[APIRouter, APIRouter]],
    routes: dict[int, tuple[BaseRoute, BaseRoute]],
) -> None:
    seen: set[int] = set()
    for route in (*router.routes, *router._low_priority_routes):
        if isinstance(route, _IncludedRouter):
            _snapshot_included_router_contexts(route, routers, routes, seen)


def snapshot_core_routes(app: FastAPI) -> list[BaseRoute]:
    """Copy route nodes and included-router links without following app or router back-references."""
    # Retain each source with its clone so Python cannot reuse its id during this snapshot.
    routers: dict[int, tuple[APIRouter, APIRouter]] = {}
    routes: dict[int, tuple[BaseRoute, BaseRoute]] = {}
    router = _clone_router(app.router, routers, routes)
    webhooks = _clone_router(app.webhooks, routers, routes)
    core_app = copy.copy(app)
    core_app.router = router
    core_app.webhooks = webhooks
    core_app.dependency_overrides = dict(app.dependency_overrides)
    core_app.user_middleware = [_copy_middleware(middleware) for middleware in app.user_middleware]
    core_app.exception_handlers = dict(app.exception_handlers)
    core_app.openapi_schema = _copy_config(app.openapi_schema)
    for attr in (
        "contact",
        "license_info",
        "openapi_tags",
        "servers",
        "openapi_external_docs",
        "swagger_ui_parameters",
    ):
        setattr(core_app, attr, _copy_config(getattr(app, attr)))
    core_app.middleware_stack = None
    _rebind_dependency_overrides(router, app, core_app)
    _rebind_dependency_overrides(webhooks, app, core_app)
    _snapshot_included_routes(router, routers, routes)
    _snapshot_included_routes(webhooks, routers, routes)
    _rebuild_route_handlers(router)
    _rebuild_route_handlers(webhooks)
    if getattr(app.openapi, "__self__", None) is app:
        core_app.openapi = MethodType(app.openapi.__func__, core_app)
    return _RouteSnapshot(router, core_app)


def get_original_frontend(app: FastAPI) -> Any | None:
    """Return the root mount's original ASGI application."""
    return next(
        (
            _snapshot_asgi_app(route.app)
            for route in app.router.routes
            if isinstance(route, Mount) and route.path in ("", "/")
        ),
        None,
    )


class _TrustedUIMount(Mount):
    def matches(self, scope: dict[str, Any]) -> tuple[Match, dict[str, Any]]:
        if scope["type"] in ("http", "websocket"):
            path = get_route_path(scope)
            if path == "/api" or path.startswith("/api/"):
                return Match.NONE, {}
        return super().matches(scope)


class TrustedRouteGuard:
    """Serve captured core routes and configuration while retaining live services and the app lifespan."""

    def __init__(self, *, app: FastAPI, routes: list[BaseRoute], frontend: Any) -> None:
        if isinstance(routes, _RouteSnapshot):
            self.app = routes.app
            openapi_url = routes.openapi_url
        else:
            self.app = copy.copy(app)
            self.app.router = copy.copy(app.router)
            self.app.router.routes = routes
            self.app.user_middleware = [_copy_middleware(middleware) for middleware in app.user_middleware]
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
        if frontend is not None:
            captured_routes.append(_TrustedUIMount("/", app=frontend, name="trusted-ui"))
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

    async def _openapi(self, request: Request) -> JSONResponse:
        # Custom node reloads invalidate the original app's cache; regenerate from this trusted route graph.
        invalidated = self.original_app.openapi_schema is None
        if invalidated:
            self.app.openapi_schema = None
        schema = self._generate_openapi()
        if invalidated:
            self.original_app.openapi_schema = _copy_config(schema)
        root_path = request.scope.get("root_path", "").rstrip("/")
        if root_path and self.app.root_path_in_servers:
            server_urls = {server.get("url") for server in schema.get("servers", [])}
            if root_path not in server_urls:
                schema = {**schema, "servers": [{"url": root_path}, *schema.get("servers", [])]}
        return JSONResponse(schema)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)
