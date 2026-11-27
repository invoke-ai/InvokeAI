"""Tests for `DisconnectTolerantASGIApp`.

Regression guard for the `KeyError: 'REQUEST_METHOD'` traceback that python-engineio raises when
a client drops before its first ASGI event arrives: `translate_request()` returns an empty
environ, then `AsyncServer.handle_request()` indexes `environ["REQUEST_METHOD"]` anyway.

The load-bearing test here is `test_disconnect_through_the_real_middleware_stack_is_silent`:
socket.io is mounted *below* `BaseHTTPMiddleware` subclasses in this app, and that changes what
a correct fix looks like. Simply returning without a response makes `call_next()` raise
`RuntimeError("No response returned.")`, which trades one traceback for another - so a test that
exercises the wrapper in isolation passes while the bug is still live in production.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from socketio import ASGIApp, AsyncServer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message, Receive, Scope, Send

from invokeai.app.api.sockets import DisconnectTolerantASGIApp

SOCKETIO_PATH = "/ws/socket.io"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _socketio_app() -> ASGIApp:
    return ASGIApp(socketio_server=AsyncServer(async_mode="asgi"), socketio_path=SOCKETIO_PATH)


def _events(*messages: Message) -> Receive:
    """A `receive` that yields the given messages, then repeats `http.disconnect` forever.

    Real servers keep reporting the disconnect rather than raising, so an over-read shows up as
    the behaviour under test instead of an unrelated `IndexError`.
    """
    pending = list(messages)

    async def receive() -> Message:
        return pending.pop(0) if pending else {"type": "http.disconnect"}

    return receive


def _collector() -> tuple[Send, list[Message]]:
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    return send, sent


def _http_scope(path: str = f"{SOCKETIO_PATH}/", query: bytes = b"EIO=4&transport=polling") -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": query,
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8000),
    }


class _PassThroughMiddleware(BaseHTTPMiddleware):
    """Stands in for `SlidingWindowTokenMiddleware` / `RedirectRootWithQueryStringMiddleware`."""

    async def dispatch(self, request: Any, call_next: Any) -> Any:
        return await call_next(request)


def _app_with_middleware(*, wrapped: bool) -> FastAPI:
    """The production topology: socket.io mounted below a `BaseHTTPMiddleware`."""
    app = FastAPI()
    socketio = _socketio_app()
    app.mount("/ws", DisconnectTolerantASGIApp(socketio) if wrapped else socketio)
    app.add_middleware(_PassThroughMiddleware)
    return app


@pytest.mark.anyio
async def test_disconnect_through_the_real_middleware_stack_is_silent():
    """The whole point of the change: nothing escapes when a client drops mid-poll."""
    send, _sent = _collector()

    # The unwrapped mount is what `main` does today - it raises.
    with pytest.raises(KeyError, match="REQUEST_METHOD"):
        await _app_with_middleware(wrapped=False)(_http_scope(), _events({"type": "http.disconnect"}), send)

    send, sent = _collector()
    await _app_with_middleware(wrapped=True)(_http_scope(), _events({"type": "http.disconnect"}), send)

    # A response must actually be emitted. Returning nothing instead makes BaseHTTPMiddleware
    # raise RuntimeError("No response returned."), i.e. a differently-named traceback.
    assert [message["type"] for message in sent] == ["http.response.start", "http.response.body"]
    assert sent[0]["status"] == 499


@pytest.mark.anyio
async def test_disconnect_never_reaches_socketio():
    """Called directly, the wrapper absorbs what the bare socket.io app cannot."""
    send, _sent = _collector()
    with pytest.raises(KeyError, match="REQUEST_METHOD"):
        await _socketio_app()(_http_scope(), _events({"type": "http.disconnect"}), send)

    send, sent = _collector()
    await DisconnectTolerantASGIApp(_socketio_app())(_http_scope(), _events({"type": "http.disconnect"}), send)
    assert [message["type"] for message in sent] == ["http.response.start", "http.response.body"]


@pytest.mark.anyio
async def test_handshake_still_works_through_the_wrapper():
    """The replay path, proven against real engine.io rather than a stand-in app."""
    send, sent = _collector()
    receive = _events({"type": "http.request", "body": b"", "more_body": False})

    await DisconnectTolerantASGIApp(_socketio_app())(_http_scope(), receive, send)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 200
    # engine.io's open packet is a `0` followed by a JSON blob carrying the session id.
    body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    assert body.startswith(b"0{")
    assert b'"sid"' in body


@pytest.mark.anyio
async def test_streamed_body_survives_the_replay():
    """The peeked event is replayed, and later chunks still stream through untouched."""
    seen: list[Message] = []

    async def recording_app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(await receive())
        seen.append(await receive())

    receive = _events(
        {"type": "http.request", "body": b"first", "more_body": True},
        {"type": "http.request", "body": b"second", "more_body": False},
    )
    send, _sent = _collector()

    await DisconnectTolerantASGIApp(recording_app)(_http_scope(), receive, send)

    assert [event["body"] for event in seen] == [b"first", b"second"]


@pytest.mark.anyio
async def test_non_http_scopes_pass_through_untouched():
    """Lifespan traffic must keep its own first event; engine.io runs a receive loop on it."""
    forwarded: dict[str, Any] = {}

    async def recording_app(scope: Scope, receive: Receive, send: Send) -> None:
        forwarded["receive"] = receive

    receive = _events({"type": "lifespan.startup"})
    send, _sent = _collector()

    await DisconnectTolerantASGIApp(recording_app)({"type": "lifespan"}, receive, send)

    # The original callable is handed over as-is, with nothing already read from it.
    assert forwarded["receive"] is receive
