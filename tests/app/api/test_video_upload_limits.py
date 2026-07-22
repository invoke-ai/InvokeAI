"""Tests for VideoUploadLimitASGIMiddleware (PR #9163 review fix).

The upload route's MAX_UPLOAD_SIZE check runs only after FastAPI has parsed (and spooled)
the entire multipart body, so oversized/chunked/concurrent requests could exhaust temp
storage before rejection. The middleware bounds ingress before the parser runs.
"""

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Response, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from invokeai.app import api_app
from invokeai.app.api.routers.videos import upload_video
from invokeai.app.api_app import (
    SubPathASGIMiddleware,
    VideoUploadLimitASGIMiddleware,
    _identify_video_upload_user,
)
from invokeai.app.services.auth.token_service import TokenData, create_access_token, set_jwt_secret
from invokeai.app.services.image_records.image_records_common import ImageCategory

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


def _call_asgi(app: VideoUploadLimitASGIMiddleware, path: str, authorization: str | None = None) -> int:
    headers = [(b"content-length", b"1")]
    if authorization is not None:
        headers.append((b"authorization", authorization.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"x", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    asyncio.run(app(scope, receive, send))  # type: ignore[arg-type]
    return next(message["status"] for message in messages if message["type"] == "http.response.start")


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


def _identify_by_bearer(scope: Any) -> tuple[bool, str | None]:
    """Test identify_user: 'Bearer <name>' authenticates as user <name>."""
    authorization = Headers(scope=scope).get("authorization")
    if authorization is None or not authorization.startswith("Bearer "):
        return False, None
    return True, authorization.removeprefix("Bearer ")


def test_unauthenticated_upload_is_rejected_without_consuming_capacity():
    app, calls = _build_app()
    middleware = VideoUploadLimitASGIMiddleware(
        app,
        max_body_bytes=MAX_BODY,
        max_concurrent=MAX_CONCURRENT,
        identify_user=_identify_by_bearer,
    )
    middleware._active = MAX_CONCURRENT

    status_code = _call_asgi(middleware, "/api/v1/videos/upload")

    assert status_code == 401
    assert middleware._active == MAX_CONCURRENT
    assert middleware._active_by_user == {}
    assert calls == []


def test_authenticated_upload_still_obeys_concurrency_bound():
    app, calls = _build_app()
    middleware = VideoUploadLimitASGIMiddleware(
        app,
        max_body_bytes=MAX_BODY,
        max_concurrent=MAX_CONCURRENT,
        identify_user=_identify_by_bearer,
    )
    middleware._active = MAX_CONCURRENT

    status_code = _call_asgi(middleware, "/api/v1/videos/upload", authorization="Bearer valid")

    assert status_code == 429
    assert calls == []


def test_per_user_quota_blocks_only_the_saturating_user():
    """One tenant at their per-user cap gets 429 while another tenant still uploads,
    even though global capacity remains."""
    app, calls = _build_app()
    middleware = VideoUploadLimitASGIMiddleware(
        app,
        max_body_bytes=MAX_BODY,
        max_concurrent=4,
        max_concurrent_per_user=2,
        identify_user=_identify_by_bearer,
    )
    middleware._active = 2
    middleware._active_by_user = {"alice": 2}

    status_a = _call_asgi(middleware, "/api/v1/videos/upload", authorization="Bearer alice")
    status_b = _call_asgi(middleware, "/api/v1/videos/upload", authorization="Bearer bob")

    assert status_a == 429
    assert status_b == 200
    assert calls == ["upload"]


def test_per_user_count_released_after_request():
    app, _calls = _build_app()
    middleware = VideoUploadLimitASGIMiddleware(
        app,
        max_body_bytes=MAX_BODY,
        max_concurrent=4,
        max_concurrent_per_user=2,
        identify_user=_identify_by_bearer,
    )

    status = _call_asgi(middleware, "/api/v1/videos/upload", authorization="Bearer alice")

    assert status == 200
    assert middleware._active == 0
    assert middleware._active_by_user == {}


def test_single_user_mode_has_no_per_user_quota():
    """identify_user returning (True, None) — single-user mode — must get the whole
    global capacity, not be clipped by the per-user cap."""
    app, calls = _build_app()
    middleware = VideoUploadLimitASGIMiddleware(
        app,
        max_body_bytes=MAX_BODY,
        max_concurrent=4,
        max_concurrent_per_user=2,
        identify_user=lambda scope: (True, None),
    )
    middleware._active = 3  # above the per-user cap, below the global cap

    status = _call_asgi(middleware, "/api/v1/videos/upload")

    assert status == 200
    assert calls == ["upload"]


@pytest.mark.parametrize(
    "multiuser,send_token,user_active,expected",
    [
        (False, False, False, (True, None)),
        (True, False, True, (False, None)),
        (True, True, False, (False, None)),
        (True, True, True, (True, "user")),
    ],
)
def test_production_upload_authentication(
    monkeypatch: pytest.MonkeyPatch,
    multiuser: bool,
    send_token: bool,
    user_active: bool,
    expected: tuple[bool, str | None],
) -> None:
    set_jwt_secret("test-secret-key-for-unit-tests-only-do-not-use-in-production")
    token_data = TokenData(user_id="user", email="user@example.com", is_admin=False)
    user = SimpleNamespace(is_active=user_active)
    invoker = SimpleNamespace(
        services=SimpleNamespace(
            configuration=SimpleNamespace(multiuser=multiuser),
            users=SimpleNamespace(get=lambda _user_id: user),
        )
    )
    monkeypatch.setattr(api_app.ApiDependencies, "invoker", invoker, raising=False)
    headers = []
    if send_token:
        headers.append((b"authorization", f"Bearer {create_access_token(token_data)}".encode()))

    assert _identify_video_upload_user({"headers": headers}) == expected  # type: ignore[arg-type]


def test_upload_video_closes_tmp_handle_when_stream_copy_fails():
    captured_handles: list[Any] = []
    real_named_tmp = tempfile.NamedTemporaryFile

    async def run_immediately(func: Any, *args: Any):
        return func(*args)

    def failing_named_tmp(*args: Any, **kwargs: Any):
        handle = real_named_tmp(*args, **kwargs)
        handle.write = MagicMock(side_effect=OSError("disk full"))
        captured_handles.append(handle)
        return handle

    upload = MagicMock(spec=UploadFile)
    upload.filename = "video.mp4"
    upload.content_type = "video/mp4"
    upload.read = AsyncMock(side_effect=[b"not-empty", b""])

    try:
        with (
            patch("invokeai.app.api.routers.videos.tempfile.NamedTemporaryFile", side_effect=failing_named_tmp),
            patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
            pytest.raises(OSError, match="disk full"),
        ):
            asyncio.run(
                upload_video(
                    current_user=TokenData(user_id="user", email="user@example.com", is_admin=False),
                    file=upload,
                    request=MagicMock(),
                    response=Response(),
                    video_category=ImageCategory.GENERAL,
                    is_intermediate=False,
                    board_id=None,
                    session_id=None,
                    metadata=None,
                )
            )

        assert len(captured_handles) == 1
        assert captured_handles[0].closed
        assert not Path(captured_handles[0].name).exists()
    finally:
        for handle in captured_handles:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)


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
