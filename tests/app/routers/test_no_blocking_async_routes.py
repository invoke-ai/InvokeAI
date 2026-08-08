"""No route handler may be `async def` without awaiting something.

Nearly every service in the backend is synchronous. A route declared `async def` that only
calls those services runs their blocking work *on the event loop*, so for its duration the
process serves no other request and delivers no socket.io event. On a large library that is
seconds per request, which users experience as the application freezing mid-generation.

`tests/app/routers/test_event_loop_blocking.py` proves the effect for a handful of routes.
This test enforces the rule for all of them, because a per-route test cannot cover a route
that does not exist yet — and the failure mode is invisible until someone has a big enough
library to notice.

See docs/contributing/blocking-work-in-api-routes for the rule itself.
"""

import ast
import pathlib

ROUTERS_DIR = pathlib.Path(__file__).parents[3] / "invokeai" / "app" / "api" / "routers"

# Decorators that register a function as a route. `api_route` takes the method as a keyword,
# the rest name it directly.
ROUTE_DECORATORS = {"get", "post", "put", "patch", "delete", "head", "options", "api_route"}

# A floor, not an exact count: routes come and go, but a collapse means this test stopped
# finding the routers rather than that the app shrank.
MIN_EXPECTED_ROUTE_HANDLERS = 150


def _is_route_handler(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr in ROUTE_DECORATORS:
            return True
    return False


def _awaits_something(node: ast.AsyncFunctionDef) -> bool:
    return any(isinstance(child, (ast.Await, ast.AsyncWith, ast.AsyncFor)) for child in ast.walk(node))


def test_no_route_handler_is_async_without_awaiting() -> None:
    offenders: list[str] = []
    handlers = 0

    for path in sorted(ROUTERS_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) or not _is_route_handler(node):
                continue
            handlers += 1
            if isinstance(node, ast.AsyncFunctionDef) and not _awaits_something(node):
                offenders.append(f"{path.name}:{node.lineno} {node.name}")

    assert handlers >= MIN_EXPECTED_ROUTE_HANDLERS, (
        f"Only {handlers} route handlers found in {ROUTERS_DIR}, expected at least "
        f"{MIN_EXPECTED_ROUTE_HANDLERS}. This test has stopped seeing the routers - fix the discovery, "
        "do not lower this floor."
    )

    assert not offenders, (
        "These route handlers are declared `async def` but never await anything, so their "
        "synchronous work runs on the event loop and stalls the whole server while it runs:\n  "
        + "\n  ".join(offenders)
        + "\n\nDeclare them `def` - FastAPI will run them in a threadpool. If a handler genuinely "
        "needs to be async, it must await something; blocking calls inside it belong in "
        "`run_in_threadpool`. See docs/contributing/blocking-work-in-api-routes."
    )
