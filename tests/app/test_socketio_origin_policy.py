"""The Socket.IO origin policy, exercised at the Engine.IO handshake.

The case table runs against a stack assembled the way `run_app` and `api_app` assemble it: uvicorn's
`ProxyHeadersMiddleware` outermost, then `SubPathASGIMiddleware` when `base_url` is set, then the CORS middleware over
the `SocketIO` mount. Forwarding headers therefore reach the policy only the way a real server delivers them. One test
also runs against the real `api_app.app` with its configured middleware.
"""

import json
import logging
from dataclasses import dataclass, field

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from invokeai.app.api import sockets as sockets_module
from invokeai.app.api_app import SubPathASGIMiddleware
from invokeai.app.services.config.config_default import InvokeAIAppConfig

PROXY_IP = "10.0.0.2"
OTHER_IP = "192.0.2.7"
SEPARATE_FRONTEND = "https://studio.example.com"
POLLING = "/ws/socket.io/?EIO=4&transport=polling"
WEBSOCKET = "/ws/socket.io/?EIO=4&transport=websocket"


@dataclass(frozen=True)
class Case:
    server_url: str
    origin: str | None
    accepted: bool
    allow_origins: list[str] = field(default_factory=list)
    base_url: str | None = None
    peer: str = OTHER_IP
    headers: dict[str, str] = field(default_factory=dict)


CASES = {
    "same origin": Case("http://localhost:9090", "http://localhost:9090", accepted=True),
    "same origin, host case and default port": Case(
        "https://invoke.example.com", "https://invoke.example.com", True, headers={"host": "Invoke.Example.com:443"}
    ),
    "same origin, IPv6 host": Case("http://[::1]:9090", "http://[::1]:9090", accepted=True),
    "no origin (non-browser client)": Case("http://localhost:9090", None, accepted=True),
    "unlisted origin": Case("http://localhost:9090", "https://evil.example", accepted=False),
    "same host, other port": Case("http://localhost:9090", "http://localhost:5173", accepted=False),
    "same host, other scheme": Case("http://localhost:9090", "https://localhost:9090", accepted=False),
    "null origin": Case("http://localhost:9090", "null", accepted=False),
    "unclosed IPv6 origin": Case("http://[::1]:9090", "http://[::1", accepted=False),
    "invalid IPv6 origin": Case("http://localhost:9090", "http://[zz]:1", accepted=False),
    "non-numeric origin port": Case("http://localhost:9090", "http://localhost:abc", accepted=False),
    "malformed host header": Case(
        "http://localhost:9090", "http://localhost:9090", False, headers={"host": "localhost:abc"}
    ),
    "listed origin": Case("http://localhost:9090", SEPARATE_FRONTEND, True, allow_origins=[SEPARATE_FRONTEND]),
    "same origin alongside a listed origin": Case(
        "http://localhost:9090", "http://localhost:9090", True, allow_origins=[SEPARATE_FRONTEND]
    ),
    "listed origin matches exactly": Case(
        "http://localhost:9090", SEPARATE_FRONTEND, False, allow_origins=[SEPARATE_FRONTEND + "/"]
    ),
    "untrusted client spoofs forwarded host": Case(
        "http://localhost:9090",
        "https://evil.example",
        accepted=False,
        headers={"x-forwarded-host": "evil.example", "x-forwarded-proto": "https"},
    ),
    "untrusted client spoofs forwarded scheme": Case(
        "http://invoke.example.com",
        "https://invoke.example.com",
        accepted=False,
        headers={"x-forwarded-proto": "https"},
    ),
    "trusted proxy forwards scheme": Case(
        "http://invoke.example.com",
        "https://invoke.example.com",
        accepted=True,
        peer=PROXY_IP,
        headers={"x-forwarded-proto": "https", "x-forwarded-for": OTHER_IP},
    ),
    "trusted proxy's forwarded host does not count": Case(
        "http://127.0.0.1:9090",
        "https://invoke.example.com",
        accepted=False,
        peer=PROXY_IP,
        headers={"x-forwarded-proto": "https", "x-forwarded-host": "invoke.example.com"},
    ),
    "wildcard allows any origin": Case("http://localhost:9090", "https://evil.example", True, allow_origins=["*"]),
    "base_url, proxy keeps the prefix": Case(
        "http://invoke.example.com/invoke",
        "https://invoke.example.com",
        accepted=True,
        base_url="/invoke",
        peer=PROXY_IP,
        headers={"x-forwarded-proto": "https"},
    ),
    "base_url, proxy strips the prefix": Case(
        "http://invoke.example.com",
        "https://invoke.example.com",
        accepted=True,
        base_url="/invoke",
        peer=PROXY_IP,
        headers={"x-forwarded-proto": "https"},
    ),
}


def _client(case: Case, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    config = InvokeAIAppConfig(allow_origins=case.allow_origins, base_url=case.base_url)
    monkeypatch.setattr(sockets_module, "get_config", lambda: config)
    app = FastAPI()
    sockets_module.SocketIO(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_credentials=config.allow_credentials,
        allow_methods=config.allow_methods,
        allow_headers=config.allow_headers,
    )
    inner = SubPathASGIMiddleware(app, case.base_url) if case.base_url else app
    server = ProxyHeadersMiddleware(inner, trusted_hosts=PROXY_IP)
    return TestClient(server, base_url=case.server_url, client=(case.peer, 50000))


def _headers(case: Case, origin: str | None = None) -> dict[str, str]:
    origin = origin or case.origin
    return case.headers | ({"origin": origin} if origin else {})


def _ws_url(case: Case, query: str = WEBSOCKET) -> str:
    # Unlike HTTP requests, the test client does not resolve WebSocket URLs against its base URL.
    return case.server_url.replace("http", "ws", 1) + query


def _assert_refused(response) -> None:
    assert response.status_code == 400
    assert "not an accepted origin" in response.text
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_polling_handshake(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    response = _client(case, monkeypatch).get(POLLING, headers=_headers(case))

    if case.accepted:
        assert response.status_code == 200
        assert response.text.startswith('0{"sid"')
        if case.origin in case.allow_origins or "*" in case.allow_origins:
            # A separate-origin frontend's polling reads need this header, and browsers refuse it with two values.
            assert response.headers.get_list("access-control-allow-origin") in ([case.origin], ["*"])
            assert response.headers.get_list("access-control-allow-credentials") == ["true"]
    else:
        _assert_refused(response)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_websocket_handshake(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(case, monkeypatch)
    # The test client does not send `Upgrade`, which engine.io requires.
    headers = _headers(case) | {"upgrade": "websocket"}

    if case.accepted:
        with client.websocket_connect(_ws_url(case), headers=headers) as websocket:
            assert websocket.receive_text().startswith('0{"sid"')
    else:
        with pytest.raises(WebSocketDisconnect) as refusal:
            with client.websocket_connect(_ws_url(case), headers=headers):
                pass
        assert "not an accepted origin" in refusal.value.reason


def test_established_polling_session_upgrades_only_from_an_accepted_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    case = CASES["same origin alongside a listed origin"]
    foreign = "https://evil.example"
    with _client(case, monkeypatch) as client:
        sid = json.loads(client.get(POLLING, headers=_headers(case)).text[1:])["sid"]

        _assert_refused(client.get(f"{POLLING}&sid={sid}", headers=_headers(case, foreign)))
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                _ws_url(case, f"{WEBSOCKET}&sid={sid}"), headers=_headers(case, foreign) | {"upgrade": "websocket"}
            ):
                pass

        with client.websocket_connect(
            _ws_url(case, f"{WEBSOCKET}&sid={sid}"), headers=_headers(case) | {"upgrade": "websocket"}
        ) as websocket:
            websocket.send_text("2probe")
            assert websocket.receive_text() == "3probe"
            websocket.send_text("5")


def test_preflight_is_answered_by_the_cors_middleware(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(CASES["listed origin"], monkeypatch)
    preflight = {"access-control-request-method": "POST"}

    listed = client.options(POLLING, headers=preflight | {"origin": SEPARATE_FRONTEND})
    assert listed.status_code == 200
    assert listed.headers.get_list("access-control-allow-origin") == [SEPARATE_FRONTEND]

    assert client.options(POLLING, headers=preflight | {"origin": "https://evil.example"}).status_code == 400


def test_refusal_warns_once_per_origin(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    client = _client(CASES["unlisted origin"], monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=sockets_module.logger.name):
        for _ in range(3):
            client.get(POLLING, headers={"origin": "https://evil.example"})

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "https://evil.example is not an accepted origin." in warnings[0]


def test_real_app_refuses_foreign_origins() -> None:
    from invokeai.app.api_app import app

    client = TestClient(app)
    assert client.get(POLLING, headers={"origin": "http://testserver"}).status_code == 200
    _assert_refused(client.get(POLLING, headers={"origin": "https://evil.example"}))
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(WEBSOCKET, headers={"origin": "https://evil.example", "upgrade": "websocket"}):
            pass
