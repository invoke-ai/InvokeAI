import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from invokeai.app.api.routers import projects
from invokeai.app.api_app import ProjectWriteLimitASGIMiddleware, SubPathASGIMiddleware

MAX_BODY = 128
MAX_CONCURRENT = 2


def _build_app() -> tuple[FastAPI, list[str]]:
    app = FastAPI()
    calls: list[str] = []

    @app.post("/api/v1/projects/")
    async def create_project() -> dict[str, bool]:
        calls.append("create")
        return {"ok": True}

    @app.put("/api/v1/projects/{project_id}")
    async def update_project(project_id: str) -> dict[str, bool]:
        calls.append(f"update:{project_id}")
        return {"ok": True}

    @app.post("/api/v1/projects/import")
    async def unrelated_project_route() -> dict[str, bool]:
        calls.append("import")
        return {"ok": True}

    return app, calls


def _client(base_path: str | None = None) -> tuple[TestClient, list[str]]:
    app, calls = _build_app()
    wrapped = ProjectWriteLimitASGIMiddleware(app, max_body_bytes=MAX_BODY, max_concurrent=MAX_CONCURRENT)
    if base_path is not None:
        wrapped = SubPathASGIMiddleware(wrapped, base_path)
        return TestClient(wrapped, base_url=f"http://testserver{base_path}"), calls
    return TestClient(wrapped), calls


def test_create_is_rejected_before_json_parsing_when_request_is_oversized() -> None:
    client, calls = _client()

    response = client.post("/api/v1/projects/", content=b" " * (MAX_BODY + 1))

    assert response.status_code == 413
    assert response.json()["detail"] == {
        "actual_bytes": MAX_BODY + 1,
        "code": "project_request_too_large",
        "max_bytes": MAX_BODY,
    }
    assert calls == []


def test_update_is_rejected_before_json_parsing_when_request_is_oversized() -> None:
    client, calls = _client()

    response = client.put("/api/v1/projects/project-1", content=b" " * (MAX_BODY + 1))

    assert response.status_code == 413
    assert calls == []


def test_project_write_at_the_ingress_limit_passes_through() -> None:
    client, calls = _client()

    response = client.post(
        "/api/v1/projects/",
        content=b"{}" + b" " * (MAX_BODY - 2),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert calls == ["create"]


def test_non_document_project_routes_are_not_limited() -> None:
    client, calls = _client()

    response = client.post("/api/v1/projects/import", content=b"x" * (MAX_BODY + 1))

    assert response.status_code == 200
    assert calls == ["import"]


def test_project_write_limit_is_root_path_aware() -> None:
    client, calls = _client("/invoke")

    response = client.put("/api/v1/projects/project-1", content=b"x" * (MAX_BODY + 1))

    assert response.status_code == 413
    assert calls == []


def test_project_write_concurrency_refusal_is_retryable() -> None:
    app, calls = _build_app()
    middleware = ProjectWriteLimitASGIMiddleware(app, max_body_bytes=MAX_BODY, max_concurrent=1)
    middleware._active = 1
    client = TestClient(middleware)

    response = client.post("/api/v1/projects/", content=b"{}")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert response.json()["detail"]["code"] == "project_write_busy"
    assert calls == []


def test_project_write_limits_bound_peak_memory_and_slow_clients() -> None:
    assert projects.MAX_CONCURRENT_PROJECT_WRITES == 2
    assert projects.MAX_CONCURRENT_PROJECT_WRITES_PER_USER == 1
    assert projects.PROJECT_WRITE_IDLE_TIMEOUT_SECONDS <= 30
    assert projects.PROJECT_WRITE_MAX_DURATION_SECONDS <= 120


def test_chunked_project_write_is_cut_off_at_the_ingress_limit() -> None:
    seen: list[dict[str, Any]] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            seen.append(message)
            if message["type"] != "http.request" or not message.get("more_body"):
                break

    middleware = ProjectWriteLimitASGIMiddleware(app, max_body_bytes=MAX_BODY, max_concurrent=MAX_CONCURRENT)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/projects/",
        "root_path": "",
        "headers": [],
    }
    chunks = [b"x" * 80, b"x" * 80]

    async def receive() -> dict[str, Any]:
        if chunks:
            return {"type": "http.request", "body": chunks.pop(0), "more_body": bool(chunks)}
        return {"type": "http.disconnect"}

    async def send(_message: dict[str, Any]) -> None:
        pass

    asyncio.run(middleware(scope, receive, send))  # type: ignore[arg-type]

    assert seen[-1]["type"] == "http.disconnect"
    assert sum(len(message.get("body", b"")) for message in seen) == 80
