"""Tests for reverse-proxy sub-path support (`SubPathASGIMiddleware` + root_path-aware redirect).

Exercises both proxy styles against a minimal FastAPI app so the subtle routing/redirect matrix
(routing, trailing-slash 307s, and the `?__theme=dark` root redirect) is protected from regression.
"""

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from invokeai.app.api_app import RedirectRootWithQueryStringMiddleware, SubPathASGIMiddleware

BASE_PATH = "/invoke"


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RedirectRootWithQueryStringMiddleware)

    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/api/v1/images")
    async def images() -> JSONResponse:
        return JSONResponse({"images": []})

    # Trailing slash so FastAPI emits a 307 redirect for the slash-less form.
    @app.get("/api/v1/boards/")
    async def boards() -> JSONResponse:
        return JSONResponse({"boards": []})

    return app


def _client(preserve: bool) -> TestClient:
    """A TestClient whose base_url selects the proxy style.

    - preserve=True: proxy forwards the sub-path unchanged -> requests hit `/invoke/...`.
    - preserve=False: proxy strips the sub-path -> requests hit `/...` (middleware restores the prefix).
    """
    wrapped = SubPathASGIMiddleware(_build_app(), BASE_PATH)
    root = f"http://testserver{BASE_PATH}" if preserve else "http://testserver"
    # Don't auto-follow redirects; we assert on the Location header.
    return TestClient(wrapped, base_url=root, follow_redirects=False)


@pytest.mark.parametrize("preserve", [True, False], ids=["preserve", "strip"])
def test_routing_works_both_styles(preserve: bool):
    client = _client(preserve)
    resp = client.get("/api/v1/images")
    assert resp.status_code == 200
    assert resp.json() == {"images": []}


@pytest.mark.parametrize("preserve", [True, False], ids=["preserve", "strip"])
def test_trailing_slash_redirect_keeps_prefix(preserve: bool):
    client = _client(preserve)
    resp = client.get("/api/v1/boards")  # no trailing slash -> 307
    assert resp.status_code == 307
    # The server-generated redirect must retain the public sub-path prefix.
    assert resp.headers["location"].endswith(f"{BASE_PATH}/api/v1/boards/")


@pytest.mark.parametrize("preserve", [True, False], ids=["preserve", "strip"])
def test_theme_query_redirect_keeps_prefix(preserve: bool):
    client = _client(preserve)
    resp = client.get("/?__theme=dark")
    assert resp.status_code == 307
    assert resp.headers["location"] == f"{BASE_PATH}/"


@pytest.mark.parametrize("preserve", [True, False], ids=["preserve", "strip"])
def test_socketio_handshake_works_both_styles(preserve: bool, monkeypatch: pytest.MonkeyPatch):
    """socket.io must keep working under a sub-path deployment.

    Regression guard: engine.io is not root_path-aware and matches the raw `scope["path"]`
    (which Starlette keeps as the full public path once `root_path` is set) against its
    configured `socketio_path`. `SocketIO` must therefore fold `base_url` into the path, or
    every handshake 404s and the UI loses all real-time updates behind the proxy.
    """
    from invokeai.app.api import sockets as sockets_module
    from invokeai.app.services.config.config_default import InvokeAIAppConfig

    monkeypatch.setattr(sockets_module, "get_config", lambda: InvokeAIAppConfig(base_url=BASE_PATH))

    app = FastAPI()
    sockets_module.SocketIO(app)
    wrapped = SubPathASGIMiddleware(app, BASE_PATH)
    root = f"http://testserver{BASE_PATH}" if preserve else "http://testserver"
    client = TestClient(wrapped, base_url=root)

    resp = client.get("/ws/socket.io/?EIO=4&transport=polling&t=abc")
    assert resp.status_code == 200
    # engine.io's open packet is a `0` followed by a JSON blob carrying the session id.
    assert resp.text.startswith("0{")
    assert '"sid"' in resp.text
