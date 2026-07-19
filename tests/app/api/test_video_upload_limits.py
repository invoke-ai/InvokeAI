"""Tests for VideoUploadLimitASGIMiddleware (PR #9163 review fix).

The upload route's MAX_UPLOAD_SIZE check runs only after FastAPI has parsed (and spooled)
the entire multipart body, so oversized/chunked/concurrent requests could exhaust temp
storage before rejection. The middleware bounds ingress before the parser runs.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from invokeai.app.api_app import SubPathASGIMiddleware, VideoUploadLimitASGIMiddleware

MAX_BODY = 1024  # tiny cap for tests
MAX_CONCURRENT = 2


def _build_app() -> tuple[FastAPI, list[str]]:
    app = FastAPI()
    calls: list[str] = []

    @app.post("/api/v1/videos/upload")
    async def upload() -> dict[str, bool]:
        calls.append("upload")
        return {"ok": True}

    @app.post("/api/v1/images/upload")
    async def other() -> dict[str, bool]:
        calls.append("other")
        return {"ok": True}

    return app, calls


def _client(base_path: str | None = None) -> tuple[TestClient, list[str]]:
    app, calls = _build_app()
    wrapped = VideoUploadLimitASGIMiddleware(app, max_body_bytes=MAX_BODY, max_concurrent=MAX_CONCURRENT)
    if base_path is not None:
        wrapped = SubPathASGIMiddleware(wrapped, base_path)
        return TestClient(wrapped, base_url=f"http://testserver{base_path}"), calls
    return TestClient(wrapped), calls


def test_oversized_content_length_rejected_before_route_runs():
    client, calls = _client()
    response = client.post("/api/v1/videos/upload", content=b"x" * (MAX_BODY + 1))
    assert response.status_code == 413
    assert calls == []


def test_within_limit_passes_through():
    client, calls = _client()
    response = client.post("/api/v1/videos/upload", content=b"x" * 10)
    assert response.status_code == 200
    assert calls == ["upload"]


def test_other_routes_unaffected():
    client, calls = _client()
    response = client.post("/api/v1/images/upload", content=b"x" * (MAX_BODY + 1))
    assert response.status_code == 200
    assert calls == ["other"]


def test_concurrency_bound_returns_429():
    app, calls = _build_app()
    middleware = VideoUploadLimitASGIMiddleware(app, max_body_bytes=MAX_BODY, max_concurrent=MAX_CONCURRENT)
    middleware._active = MAX_CONCURRENT  # simulate saturated uploads
    client = TestClient(middleware)

    response = client.post("/api/v1/videos/upload", content=b"x")

    assert response.status_code == 429
    assert response.headers.get("retry-after") is not None
    assert calls == []


def test_active_count_released_after_request():
    app, _calls = _build_app()
    middleware = VideoUploadLimitASGIMiddleware(app, max_body_bytes=MAX_BODY, max_concurrent=MAX_CONCURRENT)
    client = TestClient(middleware)

    client.post("/api/v1/videos/upload", content=b"x")

    assert middleware._active == 0


@pytest.mark.parametrize("preserve", [True, False], ids=["preserve", "strip"])
def test_route_matching_is_root_path_aware(preserve: bool):
    """Behind a sub-path proxy the public path carries the prefix; the size cap must still
    apply to the proxied upload URL. SubPathASGIMiddleware wraps this middleware in
    run_app (it restores the prefix and advertises root_path before this middleware
    runs), so path matching must strip root_path for both proxy styles."""
    app, calls = _build_app()
    limited = VideoUploadLimitASGIMiddleware(app, max_body_bytes=MAX_BODY, max_concurrent=MAX_CONCURRENT)
    wrapped = SubPathASGIMiddleware(limited, "/invoke")
    root = "http://testserver/invoke" if preserve else "http://testserver"
    client = TestClient(wrapped, base_url=root)

    response = client.post("/api/v1/videos/upload", content=b"x" * (MAX_BODY + 1))

    assert response.status_code == 413
    assert calls == []


def test_chunked_body_aborted_once_over_cap():
    """Without a Content-Length header the counting backstop must stop feeding the app
    once the cap is exceeded — the app sees a disconnect instead of an unbounded body."""
    import asyncio

    seen: list[dict] = []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            seen.append(message)
            if message["type"] != "http.request" or not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = VideoUploadLimitASGIMiddleware(app, max_body_bytes=MAX_BODY, max_concurrent=MAX_CONCURRENT)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/videos/upload",
        "root_path": "",
        "headers": [],  # no content-length -> chunked path
    }

    chunks = [b"x" * 600, b"x" * 600]  # 1200 > MAX_BODY

    async def receive():
        if chunks:
            return {"type": "http.request", "body": chunks.pop(0), "more_body": bool(chunks)}
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    asyncio.run(middleware(scope, receive, send))

    assert seen[-1]["type"] == "http.disconnect"
    # Only the first chunk (under the cap) reached the app as a body message.
    body_bytes = sum(len(m.get("body", b"")) for m in seen if m["type"] == "http.request")
    assert body_bytes == 600
