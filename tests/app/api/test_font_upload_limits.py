"""Ingress limits for custom-font multipart uploads."""

import asyncio
from typing import Any

from invokeai.app import api_app
from invokeai.app.api.routers import fonts
from invokeai.app.api_app import RequestBodyLimitASGIMiddleware


def _scope(path: str, headers: list[tuple[bytes, bytes]] | None = None) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers or [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }


def test_font_upload_limiter_is_installed_with_configured_request_cap() -> None:
    limiters = [
        middleware
        for middleware in api_app.app.user_middleware
        if middleware.cls is RequestBodyLimitASGIMiddleware
        and middleware.kwargs["matches_request"] is api_app._is_font_upload
    ]

    assert len(limiters) == 1
    kwargs = limiters[0].kwargs
    assert kwargs["max_body_bytes"] == api_app.app_config.max_font_upload_bytes + fonts.FONT_UPLOAD_MULTIPART_OVERHEAD
    assert kwargs["max_concurrent"] == fonts.MAX_CONCURRENT_FONT_UPLOADS
    assert kwargs["max_concurrent_per_user"] == fonts.MAX_CONCURRENT_FONT_UPLOADS_PER_USER
    assert api_app._is_font_upload("POST", "/api/v1/fonts")
    assert api_app._is_font_upload("POST", "/api/v1/fonts/validate")
    assert not api_app._is_font_upload("GET", "/api/v1/fonts")
    assert not api_app._is_font_upload("GET", "/api/v1/fonts/font-id/file")


def test_oversized_font_request_is_rejected_before_downstream_body_read() -> None:
    calls: list[str] = []

    async def downstream(_scope: Any, _receive: Any, _send: Any) -> None:
        calls.append("route")
        raise AssertionError("an oversized font request must not reach multipart parsing")

    middleware = RequestBodyLimitASGIMiddleware(
        downstream,
        matches_request=api_app._is_font_upload,
        too_large_detail=lambda _actual, limit: {"code": "font_request_too_large", "max_bytes": limit},
        capacity_refusal_detail=lambda _per_user: "busy",
        max_body_bytes=10,
        max_concurrent=1,
        identify_user=lambda _scope: (True, None),
    )
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        raise AssertionError("the middleware can reject from Content-Length without reading the body")

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    asyncio.run(
        middleware(
            _scope("/api/v1/fonts", [(b"content-length", b"11")]),
            receive,
            send,
        )
    )

    assert next(message["status"] for message in messages if message["type"] == "http.response.start") == 413
    assert calls == []


def test_chunked_font_request_is_cut_off_at_ingress_cap() -> None:
    seen: list[dict[str, Any]] = []

    async def downstream(_scope: Any, receive: Any, send: Any) -> None:
        while True:
            message = await receive()
            seen.append(message)
            if message["type"] != "http.request" or not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = RequestBodyLimitASGIMiddleware(
        downstream,
        matches_request=api_app._is_font_upload,
        too_large_detail=lambda _actual, limit: {"code": "font_request_too_large", "max_bytes": limit},
        capacity_refusal_detail=lambda _per_user: "busy",
        max_body_bytes=10,
        max_concurrent=1,
        identify_user=lambda _scope: (True, None),
    )
    chunks = [b"x" * 7, b"x" * 7]

    async def receive() -> dict[str, Any]:
        if chunks:
            return {"type": "http.request", "body": chunks.pop(0), "more_body": bool(chunks)}
        return {"type": "http.disconnect"}

    async def send(_message: dict[str, Any]) -> None:
        pass

    asyncio.run(middleware(_scope("/api/v1/fonts/validate"), receive, send))

    assert seen[-1]["type"] == "http.disconnect"
    assert sum(len(message.get("body", b"")) for message in seen if message["type"] == "http.request") == 7
