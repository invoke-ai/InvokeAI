"""Tests for VideoUploadLimitASGIMiddleware and the upload route's body handling.

The middleware (PR #9163 review fix) bounds ingress before any parsing happens, so
oversized, chunked or too-many-concurrent requests are rejected without touching temp
storage at all. The route itself then parses the multipart body and streams the file part
straight into one temp file, enforcing MAX_UPLOAD_SIZE as the bytes arrive — declaring
`file: UploadFile` used to make Starlette spool a second full-size copy first.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Response
from fastapi.testclient import TestClient
from python_multipart.multipart import MultipartParser, MultipartState
from starlette.datastructures import Headers

from invokeai.app import api_app
from invokeai.app.api.routers import videos
from invokeai.app.api.routers.videos import upload_video
from invokeai.app.api_app import (
    SubPathASGIMiddleware,
    VideoUploadLimitASGIMiddleware,
    _identify_video_upload_user,
    _identify_video_upload_user_async,
)
from invokeai.app.services.auth.token_service import TokenData, create_access_token, set_jwt_secret
from invokeai.app.services.board_records.board_records_common import BoardVisibility
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.util.video_thumbnails import VideoDecodeTimeoutError

MAX_BODY = 1024  # tiny cap for tests
MAX_CONCURRENT = 2


def test_configured_upload_slots_bound_peak_temp_storage_usage() -> None:
    """Peak temp storage is now one copy per in-flight upload, not two.

    The route parses the multipart body itself and writes the file part straight into its
    own temp file, so Starlette's spool no longer holds a second full-size copy.
    """
    assert videos.MAX_CONCURRENT_VIDEO_UPLOADS <= 2
    assert videos.MAX_CONCURRENT_VIDEO_UPLOADS_PER_USER <= 1
    assert videos.MAX_UPLOAD_SIZE * videos.MAX_CONCURRENT_VIDEO_UPLOADS <= 2 * 1024 * 1024 * 1024


def test_upload_probe_requires_a_decodable_frame(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(videos, "probe_video_with_codec", lambda path: (48, 32, 1.0, 8.0, "h264"))
    monkeypatch.setattr(videos, "extract_video_frame", lambda path, frame_index=0, raise_on_timeout=False: None)

    with pytest.raises(ValueError, match="decodable frame"):
        videos._probe_decodable_video(Path("metadata-only.mp4"))


def test_upload_probe_accepts_valid_metadata_and_frame(monkeypatch: pytest.MonkeyPatch):
    frame = MagicMock()
    monkeypatch.setattr(videos, "probe_video_with_codec", lambda path: (48, 32, 1.0, 8.0, "h264"))
    monkeypatch.setattr(videos, "extract_video_frame", lambda path, frame_index=0, raise_on_timeout=False: frame)

    assert videos._probe_decodable_video(Path("valid.mp4")) == ((48, 32, 1.0, 8.0), frame)


def test_upload_probe_timeout_is_inconclusive_not_a_rejection(monkeypatch: pytest.MonkeyPatch):
    """A decode-worker timeout is server contention, not evidence the video is bad — the
    upload must proceed (without a pre-extracted frame) rather than 415."""

    def _timeout(path, frame_index=0, raise_on_timeout=False):
        raise VideoDecodeTimeoutError("decode worker timed out")

    monkeypatch.setattr(videos, "probe_video_with_codec", lambda path: (48, 32, 1.0, 8.0, "h264"))
    monkeypatch.setattr(videos, "extract_video_frame", _timeout)

    assert videos._probe_decodable_video(Path("busy-server.mp4")) == ((48, 32, 1.0, 8.0), None)


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


def test_idle_chunked_upload_releases_capacity():
    messages: list[dict[str, Any]] = []
    receive_calls = 0

    async def app(scope, receive, send):
        while (message := await receive())["type"] == "http.request" and message.get("more_body"):
            pass
        await Response(status_code=400)(scope, receive, send)

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        if receive_calls == 1:
            return {"type": "http.request", "body": b"x", "more_body": True}
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/videos/upload",
        "root_path": "",
        "headers": [],
    }
    middleware = VideoUploadLimitASGIMiddleware(
        app,
        max_body_bytes=MAX_BODY,
        max_concurrent=1,
        idle_timeout_seconds=0.01,
    )

    asyncio.run(asyncio.wait_for(middleware(scope, receive, send), timeout=1))  # type: ignore[arg-type]

    assert middleware._active == 0
    assert any(message.get("status") == 400 for message in messages)


def test_drip_feed_upload_cannot_hold_capacity_past_absolute_timeout():
    receive_calls = 0

    async def app(scope, receive, send):
        while (await receive())["type"] == "http.request":
            pass
        await Response(status_code=400)(scope, receive, send)

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        await asyncio.sleep(0.005)
        return {"type": "http.request", "body": b"x", "more_body": True}

    async def send(_message: dict[str, Any]) -> None:
        pass

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/videos/upload",
        "root_path": "",
        "headers": [],
    }
    # The absolute deadline must dwarf the per-chunk delay: Windows event-loop timers
    # have ~15.6 ms granularity, so a 20 ms deadline could expire before the first
    # "5 ms" chunk was ever delivered, ending the request at receive_calls == 1 and
    # proving nothing about the drip-feed case.
    middleware = VideoUploadLimitASGIMiddleware(
        app,
        max_body_bytes=MAX_BODY,
        max_concurrent=1,
        idle_timeout_seconds=1.0,
        max_upload_duration_seconds=0.25,
    )

    asyncio.run(asyncio.wait_for(middleware(scope, receive, send), timeout=5))  # type: ignore[arg-type]

    assert receive_calls > 1
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


@pytest.mark.anyio
async def test_upload_authentication_does_not_block_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    set_jwt_secret("test-secret-key-for-unit-tests-only-do-not-use-in-production")
    token = create_access_token(TokenData(user_id="user", email="user@example.com", is_admin=False))
    offloaded: list[Any] = []

    async def run_in_threadpool(func: Any, *args: Any) -> Any:
        offloaded.append(func)
        await asyncio.sleep(0.1)
        return func(*args)

    monkeypatch.setattr(api_app, "run_in_threadpool", run_in_threadpool)

    def slow_get(_user_id: str) -> SimpleNamespace:
        return SimpleNamespace(is_active=True)

    invoker = SimpleNamespace(
        services=SimpleNamespace(
            configuration=SimpleNamespace(multiuser=True),
            users=SimpleNamespace(get=slow_get),
        )
    )
    monkeypatch.setattr(api_app.ApiDependencies, "invoker", invoker, raising=False)

    async def inner_app(scope, receive, send):
        await Response(status_code=200)(scope, receive, send)

    middleware = VideoUploadLimitASGIMiddleware(
        inner_app,
        max_body_bytes=MAX_BODY,
        max_concurrent=MAX_CONCURRENT,
        identify_user=_identify_video_upload_user_async,
    )
    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.4"},
        "method": "POST",
        "path": "/api/v1/videos/upload",
        "root_path": "",
        "headers": [(b"authorization", f"Bearer {token}".encode()), (b"content-length", b"0")],
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message: dict[str, Any]) -> None:
        pass

    started = time.monotonic()
    upload = asyncio.create_task(middleware(scope, receive, send))  # type: ignore[arg-type]
    await asyncio.sleep(0.01)
    heartbeat_elapsed = time.monotonic() - started
    await upload

    assert heartbeat_elapsed < 0.05
    assert offloaded == [_identify_video_upload_user]


BOUNDARY = "testboundary"


def _multipart_body(
    file_bytes: bytes,
    filename: str = "video.mp4",
    metadata: str | None = None,
    content_type: str = "video/mp4",
) -> bytes:
    parts = []
    if metadata is not None:
        parts.append(f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n{metadata}\r\n'.encode())
    parts.append(
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n".encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(parts)


def test_python_multipart_contract_the_upload_route_depends_on():
    """Pins the `python_multipart` behaviours the hand-rolled parsing relies on.

    The route drives `MultipartParser` directly instead of going through Starlette, so a
    version bump that changes any of this silently changes upload behaviour. If one of
    these ever fails, revisit `_stream_video_upload` rather than just updating the numbers.
    """
    body = _multipart_body(b"abc")
    closing = f"--{BOUNDARY}--\r\n".encode()

    # 1. `on_end` fires exactly once, at the closing boundary. `saw_end` is built on it.
    seen: list[str] = []
    parser = MultipartParser(
        BOUNDARY.encode(),
        {"on_part_begin": lambda: seen.append("part_begin"), "on_end": lambda: seen.append("end")},
    )
    parser.write(body)
    parser.finalize()
    assert seen.count("end") == 1
    assert seen[-1] == "end"
    assert parser.state == MultipartState.END

    # 2. `finalize()` does NOT verify the parser reached its end state — it is documented as
    #    a no-op with a TODO. A body cut off before the closing boundary finalizes cleanly,
    #    which is exactly why the route needs its own end-of-body check.
    truncated_end: list[str] = []
    truncated = MultipartParser(BOUNDARY.encode(), {"on_end": lambda: truncated_end.append("end")})
    truncated.write(body[: -len(closing)])
    truncated.finalize()  # does not raise
    assert truncated_end == []
    assert truncated.state != MultipartState.END

    # 3. `max_size` silently *truncates* rather than raising, so an over-cap body arrives at
    #    the same end-of-body check instead of erroring out of `write()`.
    capped_end: list[str] = []
    capped = MultipartParser(
        BOUNDARY.encode(), {"on_end": lambda: capped_end.append("end")}, max_size=len(body) - len(closing)
    )
    capped.write(body)
    capped.finalize()  # does not raise
    assert capped_end == []


def _fake_upload_request(body: bytes, chunk_size: int = 8) -> MagicMock:
    """A Request stand-in that dribbles the body out in small chunks."""
    request = MagicMock()
    request.headers = {"content-type": f"multipart/form-data; boundary={BOUNDARY}"}

    async def stream():
        for start in range(0, len(body), chunk_size):
            yield body[start : start + chunk_size]

    request.stream = stream
    return request


def _run_upload(request: MagicMock) -> Any:
    return asyncio.run(
        upload_video(
            current_user=TokenData(user_id="user", email="user@example.com", is_admin=False),
            request=request,
            response=Response(),
            video_category=ImageCategory.GENERAL,
            is_intermediate=False,
            board_id=None,
            session_id=None,
        )
    )


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

    try:
        with (
            patch("invokeai.app.api.routers.videos.tempfile.NamedTemporaryFile", side_effect=failing_named_tmp),
            patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
            pytest.raises(OSError, match="disk full"),
        ):
            _run_upload(_fake_upload_request(_multipart_body(b"not-empty")))

        assert len(captured_handles) == 1
        assert captured_handles[0].closed
        assert not Path(captured_handles[0].name).exists()
    finally:
        for handle in captured_handles:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)


def test_upload_video_writes_exactly_one_copy_of_the_body():
    """The file part is written straight to the route's temp file, in order, once."""
    payload = bytes(range(256)) * 8
    written: list[bytes] = []
    captured_handles: list[Any] = []
    real_named_tmp = tempfile.NamedTemporaryFile

    async def run_immediately(func: Any, *args: Any):
        return func(*args)

    def recording_named_tmp(*args: Any, **kwargs: Any):
        handle = real_named_tmp(*args, **kwargs)
        real_write = handle.write
        handle.write = lambda chunk: (written.append(bytes(chunk)), real_write(chunk))[1]
        captured_handles.append(handle)
        return handle

    try:
        with (
            patch("invokeai.app.api.routers.videos.tempfile.NamedTemporaryFile", side_effect=recording_named_tmp),
            patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
            patch("invokeai.app.api.routers.videos._is_mp4_file", return_value=False),
            pytest.raises(HTTPException) as error,
        ):
            _run_upload(_fake_upload_request(_multipart_body(payload), chunk_size=13))

        # Stops at the container check — the point is what reached the disk before that.
        assert error.value.status_code == 415
        assert b"".join(written) == payload
    finally:
        for handle in captured_handles:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)


def test_upload_video_rejects_bad_file_part_before_any_of_it_reaches_disk():
    """A rejected upload must not put any of its payload on disk.

    The old route could only reject after Starlette had spooled the whole thing; the
    filename/MIME gate now fires from the part headers, before the payload streams in.
    """
    payload = b"x" * 4096
    written: list[bytes] = []
    captured_handles: list[Any] = []
    real_named_tmp = tempfile.NamedTemporaryFile
    body = _multipart_body(payload, filename="notes.txt", content_type="text/plain")

    async def run_immediately(func: Any, *args: Any):
        return func(*args)

    def recording_named_tmp(*args: Any, **kwargs: Any):
        handle = real_named_tmp(*args, **kwargs)
        real_write = handle.write
        handle.write = lambda chunk: (written.append(bytes(chunk)), real_write(chunk))[1]
        captured_handles.append(handle)
        return handle

    try:
        with (
            patch("invokeai.app.api.routers.videos.tempfile.NamedTemporaryFile", side_effect=recording_named_tmp),
            patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
            pytest.raises(HTTPException) as error,
        ):
            _run_upload(_fake_upload_request(body, chunk_size=64))

        assert error.value.status_code == 415
        assert written == []
    finally:
        for handle in captured_handles:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)


def test_upload_video_rejects_bad_file_part_without_finishing_the_body():
    """A rejected upload must not require reading the rest of the body first.

    The old route could only reject after Starlette had spooled the whole thing; the
    filename/MIME gate now fires from the part headers, before the payload streams in.
    """
    consumed: list[int] = []
    body = _multipart_body(b"x" * 4096, filename="notes.txt", content_type="text/plain")

    async def run_immediately(func: Any, *args: Any):
        return func(*args)

    request = MagicMock()
    request.headers = {"content-type": f"multipart/form-data; boundary={BOUNDARY}"}

    async def stream():
        for start in range(0, len(body), 64):
            consumed.append(start)
            yield body[start : start + 64]

    request.stream = stream

    with (
        patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
        pytest.raises(HTTPException) as error,
    ):
        _run_upload(request)

    assert error.value.status_code == 415
    # Rejected from the part headers: only the first chunks were ever read.
    assert len(consumed) * 64 < len(body)


@pytest.mark.parametrize("header", ["multipart/form-data", "MULTIPART/Form-Data"])
def test_upload_video_accepts_the_content_type_in_any_case(header: str):
    """Content-Type is case-insensitive (RFC 7231) and parse_options_header preserves case,
    so an upper-case media type used to be a 422 instead of a normal upload."""

    async def run_immediately(func: Any, *args: Any):
        return func(*args)

    request = MagicMock()
    request.headers = {"content-type": f"{header}; boundary={BOUNDARY}"}
    body = _multipart_body(b"abc")

    async def stream():
        yield body

    request.stream = stream

    with (
        patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
        patch("invokeai.app.api.routers.videos._is_mp4_file", return_value=False),
        pytest.raises(HTTPException) as error,
    ):
        _run_upload(request)

    # Got as far as the container check either way — i.e. the body was parsed, not refused.
    assert error.value.status_code == 415


def test_upload_video_rejects_a_body_with_no_closing_boundary():
    """`MultipartParser.finalize()` does not check the parser reached its end state.

    Without the explicit end-of-body check a body that stops right after the file bytes
    looks complete, and a truncated file gets probed and persisted whenever the partial
    bytes happen to survive the MP4 checks.
    """
    complete = _multipart_body(b"y" * 512)
    truncated = complete[: -len(f"--{BOUNDARY}--\r\n".encode())]
    is_mp4 = MagicMock(return_value=True)

    async def run_immediately(func: Any, *args: Any):
        return func(*args)

    with (
        patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
        patch("invokeai.app.api.routers.videos._is_mp4_file", is_mp4),
        pytest.raises(HTTPException) as error,
    ):
        _run_upload(_fake_upload_request(truncated, chunk_size=64))

    assert error.value.status_code == 422
    # Stopped before the container check, so nothing downstream (probe, create) ran either.
    is_mp4.assert_not_called()

    # Control: the same body with its closing boundary gets past the completeness check.
    with (
        patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
        patch("invokeai.app.api.routers.videos._is_mp4_file", return_value=False),
        pytest.raises(HTTPException) as error,
    ):
        _run_upload(_fake_upload_request(complete, chunk_size=64))

    assert error.value.status_code == 415


def _forbidden_board_deps() -> MagicMock:
    deps = MagicMock()
    deps.invoker.services.boards.get_dto.return_value = SimpleNamespace(
        user_id="someone-else", board_visibility=BoardVisibility.Private
    )
    return deps


def _upload_scope(query_string: bytes = b"video_category=general&is_intermediate=false") -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/videos/upload",
        "raw_path": b"/api/v1/videos/upload",
        "query_string": query_string,
        "headers": [(b"content-type", f"multipart/form-data; boundary={BOUNDARY}".encode())],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }


def _drive_upload(
    middleware: VideoUploadLimitASGIMiddleware,
    body: bytes,
    scope: dict[str, Any] | None = None,
    chunk_delay: float = 0.0,
) -> tuple[int, list[tuple[bytes, bytes]], int, int]:
    """POST `body` in small chunks; return (status, response headers, chunks sent, chunks total)."""
    chunks = [body[start : start + 64] for start in range(0, len(body), 64)]
    sent = 0
    result: dict[str, Any] = {}

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if chunk_delay:
            await asyncio.sleep(chunk_delay)
        if sent >= len(chunks):
            return {"type": "http.request", "body": b"", "more_body": False}
        chunk = chunks[sent]
        sent += 1
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
            result["headers"] = [(name.lower(), value.lower()) for name, value in message["headers"]]
            result["sent_at_response"] = sent

    asyncio.run(asyncio.wait_for(middleware(_upload_scope() if scope is None else scope, receive, send), timeout=5))  # type: ignore[arg-type]
    return result["status"], result["headers"], result["sent_at_response"], len(chunks)


def _real_upload_app() -> FastAPI:
    """A FastAPI app carrying the real videos router, with only authentication stubbed."""
    from invokeai.app.api.auth_dependencies import get_current_user_or_default

    app = FastAPI()
    app.include_router(videos.videos_router, prefix="/api")
    app.dependency_overrides[get_current_user_or_default] = lambda: TokenData(
        user_id="user", email="user@example.com", is_admin=False
    )
    return app


@pytest.mark.parametrize(
    "query_string,expected_status",
    [
        (b"video_category=general&is_intermediate=false&board_id=not-mine", 403),  # rejected inside the route
        (b"video_category=bogus&is_intermediate=false", 422),  # rejected by FastAPI, before it
        (b"", 422),  # missing required query params — also before the route body
    ],
    ids=["forbidden-board", "invalid-query", "missing-query"],
)
def test_upload_answered_before_the_body_ends_closes_the_connection(query_string: bytes, expected_status: int):
    """An upload answered while the client is still sending must not leave it streaming.

    VideoUploadLimitASGIMiddleware releases its global and per-user leases as soon as the app
    returns, so a client that keeps sending afterwards would hold ingress with no slot
    charged against it — outside the 429 bound, the idle timeout and the duration cap.
    `Connection: close` ends that upload with the response instead.

    Draining the body would also close the hole, but it cannot: FastAPI answers an invalid
    query string before the route body runs at all (the `invalid-query`/`missing-query`
    cases), so no amount of route-level draining reaches those paths. It would also pin one
    of only MAX_CONCURRENT_VIDEO_UPLOADS slots for the full duration cap per rejection,
    making each early rejection a cheaper denial of service than the hole it closed.
    """
    body = _multipart_body(b"z" * 2048)
    scope = _upload_scope(query_string)
    middleware = VideoUploadLimitASGIMiddleware(_real_upload_app(), max_body_bytes=10 * len(body), max_concurrent=1)

    with patch.object(videos, "ApiDependencies", _forbidden_board_deps()):
        status, headers, sent_at_response, total = _drive_upload(middleware, body, scope)

    assert status == expected_status
    assert (b"connection", b"close") in headers
    # The point of closing rather than draining: answered immediately, slot given straight back.
    assert sent_at_response < total
    assert middleware._active == 0


def test_close_does_not_trust_a_client_supplied_content_length():
    """`Content-Length: 0` must not be taken as proof the client has finished sending.

    h11 accepts `Content-Length: 0` alongside `Transfer-Encoding: chunked` — the chunked
    framing wins — so a client that sends both could otherwise suppress the close and then
    stream indefinitely with no lease held and none of the caps applying (they all live in
    `limited_receive`, which an app that answered early never calls again).
    """
    body = _multipart_body(b"z" * 2048)
    scope = _upload_scope(b"video_category=general&is_intermediate=false&board_id=not-mine")
    scope["headers"] = [*scope["headers"], (b"content-length", b"0"), (b"transfer-encoding", b"chunked")]
    middleware = VideoUploadLimitASGIMiddleware(_real_upload_app(), max_body_bytes=10 * len(body), max_concurrent=1)

    with patch.object(videos, "ApiDependencies", _forbidden_board_deps()):
        status, headers, _sent, _total = _drive_upload(middleware, body, scope)

    assert status == 403
    assert (b"connection", b"close") in headers


@pytest.mark.parametrize(
    "limits,chunk_delay",
    [
        ({"max_body_bytes": 128}, 0.0),
        ({"max_body_bytes": 10**6, "max_upload_duration_seconds": 0.0}, 0.0),
        ({"max_body_bytes": 10**6, "idle_timeout_seconds": 0.01}, 0.2),
    ],
    ids=["byte-cap", "duration-cap", "idle-timeout"],
)
def test_aborting_a_still_uploading_client_closes_the_connection(limits: dict[str, Any], chunk_delay: float):
    """The byte cap, duration cap and idle timeout all synthesize a disconnect *because* the
    client is still uploading, so the response they produce must close the connection.

    Marking the body finished on those paths would suppress exactly the close they need.
    """
    body = _multipart_body(b"z" * 2048)
    middleware = VideoUploadLimitASGIMiddleware(_real_upload_app(), max_concurrent=1, **limits)

    with patch.object(videos, "ApiDependencies", MagicMock()):
        status, headers, sent_at_response, total = _drive_upload(middleware, body, chunk_delay=chunk_delay)

    assert sent_at_response < total  # aborted mid-body
    # A client-side abort, reported as such rather than as a server fault.
    assert status == 400
    assert (b"connection", b"close") in headers
    assert middleware._active == 0


def test_close_is_not_sent_on_http2():
    """`connection` is hop-by-hop and illegal in HTTP/2+."""
    body = _multipart_body(b"z" * 2048)
    scope = _upload_scope(b"video_category=general&is_intermediate=false&board_id=not-mine")
    scope["http_version"] = "2"
    middleware = VideoUploadLimitASGIMiddleware(_real_upload_app(), max_body_bytes=10 * len(body), max_concurrent=1)

    with patch.object(videos, "ApiDependencies", _forbidden_board_deps()):
        status, headers, _sent, _total = _drive_upload(middleware, body, scope)
    assert status == 403
    assert not any(name == b"connection" for name, _ in headers)

    # The middleware's own rejections are gated the same way.
    saturated = VideoUploadLimitASGIMiddleware(_real_upload_app(), max_body_bytes=10 * len(body), max_concurrent=0)
    status, headers, _sent, _total = _drive_upload(saturated, body, scope)
    assert status == 429
    assert not any(name == b"connection" for name, _ in headers)


def test_close_replaces_rather_than_duplicates_an_existing_connection_header():
    """Two `connection` headers on one response is not a well-formed message."""
    body = _multipart_body(b"z" * 2048)

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-length", b"0"), (b"connection", b"keep-alive")],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    middleware = VideoUploadLimitASGIMiddleware(app, max_body_bytes=10 * len(body), max_concurrent=1)
    _status, headers, _sent, _total = _drive_upload(middleware, body)

    assert [value for name, value in headers if name == b"connection"] == [b"close"]


def test_unauthenticated_rejection_closes_the_connection():
    """The 401 answers without reading the body too."""
    body = _multipart_body(b"z" * 2048)
    middleware = VideoUploadLimitASGIMiddleware(
        _real_upload_app(),
        max_body_bytes=10 * len(body),
        max_concurrent=1,
        identify_user=lambda scope: (False, None),
    )

    status, headers, _sent, _total = _drive_upload(middleware, body)
    assert status == 401
    assert (b"connection", b"close") in headers


def test_completed_upload_does_not_close_the_connection():
    """Control: a client that finished sending gets a normal keep-alive response.

    Without this, the close would be unconditional and every upload would cost a new
    connection.
    """
    body = _multipart_body(b"z" * 2048)
    deps = MagicMock()
    deps.invoker.services.videos.create.return_value = SimpleNamespace(
        video_url="/api/v1/videos/i/v.mp4/full", video_name="v.mp4"
    )
    middleware = VideoUploadLimitASGIMiddleware(_real_upload_app(), max_body_bytes=10 * len(body), max_concurrent=1)

    with (
        patch.object(videos, "ApiDependencies", deps),
        patch.object(videos, "_is_mp4_file", return_value=False),
    ):
        status, headers, sent_at_response, total = _drive_upload(middleware, body)

    # 415 from the container check — reached only after the whole body was read.
    assert status == 415
    assert sent_at_response == total
    assert not any(name == b"connection" for name, _ in headers)
    assert middleware._active == 0


def test_rejections_the_middleware_makes_itself_also_close_the_connection():
    """The 429/413 paths answer without reading the body too."""
    body = _multipart_body(b"z" * 2048)
    middleware = VideoUploadLimitASGIMiddleware(_real_upload_app(), max_body_bytes=10 * len(body), max_concurrent=0)

    status, headers, _sent, _total = _drive_upload(middleware, body)
    assert status == 429
    assert (b"connection", b"close") in headers

    oversized = VideoUploadLimitASGIMiddleware(_real_upload_app(), max_body_bytes=8, max_concurrent=1)
    scope = _upload_scope()
    scope["headers"] = [*scope["headers"], (b"content-length", str(len(body)).encode())]
    status, headers, _sent, _total = _drive_upload(oversized, body, scope)
    assert status == 413
    assert (b"connection", b"close") in headers


def test_upload_video_requires_a_file_part():
    async def run_immediately(func: Any, *args: Any):
        return func(*args)

    body = f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n{{}}\r\n--{BOUNDARY}--\r\n'.encode()

    with (
        patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
        pytest.raises(HTTPException) as error,
    ):
        _run_upload(_fake_upload_request(body))

    assert error.value.status_code == 422


def test_upload_video_rejects_oversized_file_part_mid_stream(monkeypatch: pytest.MonkeyPatch):
    """The size cap fires while the body streams, not after it has all landed on disk."""
    monkeypatch.setattr(videos, "MAX_UPLOAD_SIZE", 512)
    written = 0

    async def run_immediately(func: Any, *args: Any):
        return func(*args)

    real_named_tmp = tempfile.NamedTemporaryFile
    captured_handles: list[Any] = []

    def recording_named_tmp(*args: Any, **kwargs: Any):
        handle = real_named_tmp(*args, **kwargs)
        real_write = handle.write

        def counting_write(chunk: bytes) -> int:
            nonlocal written
            written += len(chunk)
            return real_write(chunk)

        handle.write = counting_write
        captured_handles.append(handle)
        return handle

    try:
        with (
            patch("invokeai.app.api.routers.videos.tempfile.NamedTemporaryFile", side_effect=recording_named_tmp),
            patch("invokeai.app.api.routers.videos.run_in_threadpool", side_effect=run_immediately),
            pytest.raises(HTTPException) as error,
        ):
            _run_upload(_fake_upload_request(_multipart_body(b"y" * 4096), chunk_size=64))

        assert error.value.status_code == 413
        # Only the bytes up to the cap (plus at most one chunk) were ever written.
        assert written <= 512 + 64
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
