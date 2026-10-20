import http.server
import io
import logging
import socket
import threading
from typing import Any, Generator, Iterator

import pytest
import requests
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.external_generation.errors import ExternalProviderRequestError
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalReferenceImage,
)
from invokeai.app.services.external_generation.image_utils import encode_image_base64
from invokeai.app.services.external_generation.providers import alibabacloud as alibabacloud_module
from invokeai.app.services.external_generation.providers.alibabacloud import AlibabaCloudProvider
from invokeai.app.util import ssrf
from invokeai.app.util.ssrf import UnsafeDownloadURLException
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities


class DummyResponse:
    def __init__(
        self,
        ok: bool,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.ok = ok
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.content = content
        self.headers: dict[str, str] = headers or {}

    def json(self) -> dict:
        return self._json_data

    def iter_content(self, chunk_size: int = 65536) -> Iterator[bytes]:
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i : i + chunk_size]

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _make_image(color: str = "blue") -> Image.Image:
    return Image.new("RGB", (16, 16), color=color)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _build_model(provider_model_id: str) -> ExternalApiModelConfig:
    return ExternalApiModelConfig(
        key=f"alibabacloud_{provider_model_id}",
        name=provider_model_id,
        provider_id="alibabacloud",
        provider_model_id=provider_model_id,
        capabilities=ExternalModelCapabilities(
            modes=["txt2img"],
            supports_reference_images=True,
            supports_seed=True,
        ),
    )


def _build_request(
    model: ExternalApiModelConfig,
    reference_images: list[ExternalReferenceImage] | None = None,
) -> ExternalGenerationRequest:
    return ExternalGenerationRequest(
        model=model,
        mode="txt2img",  # type: ignore[arg-type]
        prompt="a cat",
        seed=42,
        num_images=1,
        width=1024,
        height=1024,
        image_size=None,
        init_image=None,
        mask_image=None,
        reference_images=reference_images or [],
        metadata=None,
    )


def _provider() -> AlibabaCloudProvider:
    config = InvokeAIAppConfig(external_alibabacloud_api_key="test-key")
    return AlibabaCloudProvider(config, logging.getLogger("test"))


def test_unknown_model_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request = _build_request(_build_model("not-a-real-model"))

    def fail_post(*_args: Any, **_kwargs: Any) -> DummyResponse:  # pragma: no cover - should not be called
        raise AssertionError("network must not be touched for unknown model")

    monkeypatch.setattr("requests.post", fail_post)

    with pytest.raises(ExternalProviderRequestError, match="Unknown DashScope model_id"):
        provider.generate(request)


def test_sync_routes_qwen_edit_max_with_reference_images(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    ref = _make_image("red")
    request = _build_request(
        _build_model("qwen-image-edit-max"),
        reference_images=[ExternalReferenceImage(image=ref)],
    )
    captured: dict[str, Any] = {}

    image_url = "https://example.invalid/result.png"
    image_bytes = _png_bytes(_make_image("green"))

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        captured["url"] = url
        captured["json"] = json
        return DummyResponse(
            ok=True,
            json_data={
                "request_id": "req-1",
                "output": {
                    "choices": [
                        {"message": {"content": [{"image": image_url}]}},
                    ]
                },
            },
        )

    def fake_get(_session: Any, url: str, timeout: int, stream: bool = False) -> DummyResponse:
        assert url == image_url
        return DummyResponse(
            ok=True,
            content=image_bytes,
            headers={"Content-Length": str(len(image_bytes))},
        )

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.Session.get", fake_get)

    result = provider.generate(request)

    assert "multimodal-generation" in captured["url"]
    payload = captured["json"]
    messages = payload["input"]["messages"]
    content = messages[0]["content"]
    # Reference image first, then prompt text — and no init_image entry.
    assert content[0]["image"].startswith("data:image/png;base64,")
    assert content[0]["image"].endswith(encode_image_base64(ref))
    assert content[1] == {"text": request.prompt}
    assert len(content) == 2
    assert payload["model"] == "qwen-image-edit-max"
    assert payload["parameters"]["seed"] == request.seed
    assert result.provider_request_id == "req-1"
    assert len(result.images) == 1


def test_sync_error_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        return DummyResponse(ok=False, status_code=400, text="bad request")

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(ExternalProviderRequestError, match="DashScope request failed"):
        provider.generate(request)


def test_sync_retries_on_429_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))
    image_bytes = _png_bytes(_make_image("yellow"))
    image_url = "https://example.invalid/r.png"

    calls = {"n": 0}

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> DummyResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResponse(ok=False, status_code=429, text="rate limited", headers={"Retry-After": "0"})
        return DummyResponse(
            ok=True,
            json_data={
                "request_id": "req-2",
                "output": {"choices": [{"message": {"content": [{"image": image_url}]}}]},
            },
        )

    def fake_get(_session: Any, url: str, timeout: int, stream: bool = False) -> DummyResponse:
        return DummyResponse(ok=True, content=image_bytes, headers={"Content-Length": str(len(image_bytes))})

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.Session.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    result = provider.generate(request)
    assert calls["n"] == 2
    assert len(result.images) == 1


def test_async_parser_does_not_double_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """A result with both `url` and `b64_image` must yield one image, not two."""
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))
    image_bytes = _png_bytes(_make_image("magenta"))
    image_url = "https://example.invalid/x.png"

    def fake_get(_session: Any, url: str, timeout: int, stream: bool = False) -> DummyResponse:
        return DummyResponse(ok=True, content=image_bytes, headers={"Content-Length": str(len(image_bytes))})

    monkeypatch.setattr("requests.Session.get", fake_get)

    output: dict[str, Any] = {
        "results": [
            {
                "url": image_url,
                "b64_image": encode_image_base64(_make_image("cyan")),
            }
        ]
    }
    result = provider._parse_async_response(output, request, request_id="rid")
    assert len(result.images) == 1


def test_async_parser_accepts_b64_only(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))
    output: dict[str, Any] = {
        "results": [
            {"b64_image": encode_image_base64(_make_image("cyan"))},
        ]
    }
    result = provider._parse_async_response(output, request, request_id="rid")
    assert len(result.images) == 1


def test_download_image_size_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    too_big = alibabacloud_module._DOWNLOAD_MAX_BYTES + 1

    def fake_get(_session: Any, url: str, timeout: int, stream: bool = False) -> DummyResponse:
        return DummyResponse(
            ok=True,
            content=b"\x00" * 16,  # body itself is small; we trip the Content-Length check first
            headers={"Content-Length": str(too_big)},
        )

    monkeypatch.setattr("requests.Session.get", fake_get)

    with pytest.raises(ExternalProviderRequestError, match="exceeds"):
        provider._download_image("https://example.invalid/big.png")


def test_download_image_rejects_unsafe_provider_url(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()

    def fail_unguarded_get(*_args: Any, **_kwargs: Any) -> DummyResponse:
        pytest.fail("provider response URLs must not use the unguarded requests.get path")

    def reject_unsafe_url(*_args: Any, **_kwargs: Any) -> DummyResponse:
        raise UnsafeDownloadURLException("non-public address")

    monkeypatch.setattr("requests.get", fail_unguarded_get)
    monkeypatch.setattr("requests.Session.get", reject_unsafe_url)

    with pytest.raises(ExternalProviderRequestError, match="unsafe image URL"):
        provider._download_image("http://127.0.0.1/internal.png")


# ------------- The guard itself, exercised end to end against a real socket -------------
#
# The stubbed test above pins the error translation, but it cannot fail if the guard is
# removed: it patches `requests.Session.get` to raise, which a plain unguarded Session
# does just as happily. These tests connect to a real listener instead, so the only thing
# that can stop the download is the socket-level peer check.


class _LoopbackImageHandler(http.server.BaseHTTPRequestHandler):
    """Serves a valid PNG, so a successful fetch is indistinguishable from a real one."""

    def do_GET(self):  # noqa: N802
        body = _png_bytes(_make_image("red"))
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def loopback_image_server() -> Generator[tuple[str, int], None, None]:
    """A loopback HTTP server, bound to whichever family `localhost` resolves to here.

    Following `localhost` rather than hard-coding 127.0.0.1 keeps the hostname-based tests
    below working on images where `localhost` is IPv6-only; otherwise they would fail on a
    connection refused rather than on the guard. Yields the matching host literal and port.
    """
    family, _socktype, _proto, _canonname, sockaddr = socket.getaddrinfo(
        "localhost", 0, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )[0]
    server_cls = type("_LoopbackImageServer", (http.server.HTTPServer,), {"address_family": family})
    srv = server_cls(sockaddr[:2], _LoopbackImageHandler)
    host = f"[{sockaddr[0]}]" if family == socket.AF_INET6 else sockaddr[0]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield host, srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def loopback_tls_port() -> Generator[int, None, None]:
    """A bare TCP listener that accepts and immediately hangs up, speaking no TLS.

    The peer check runs in `_new_conn()`, before the handshake, so an `https://` URL is
    refused here without any certificate machinery. If the HTTPS guard were missing the
    handshake would be attempted and fail with an `SSLError` instead, which is what makes
    this a real test of the https path rather than of a class name.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    srv.settimeout(0.25)
    stop = threading.Event()

    def accept_and_close() -> None:
        while not stop.is_set():
            try:
                conn, _addr = srv.accept()
            except OSError:  # timeout, or the socket closed under us
                continue
            conn.close()

    thread = threading.Thread(target=accept_and_close, daemon=True)
    thread.start()
    try:
        yield srv.getsockname()[1]
    finally:
        stop.set()
        thread.join(timeout=2)
        srv.close()


def _unsafe_in_chain(exc: BaseException) -> bool:
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, UnsafeDownloadURLException):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def test_download_image_refuses_live_loopback_server(loopback_image_server: tuple[str, int]) -> None:
    """A provider-supplied URL pointing at a live internal service must not be fetched.

    The port is open and serving a real PNG, so the TCP connect succeeds and the peer
    check is the only thing that can refuse it. Reverting `_download_image` to any
    unguarded client fails here.
    """
    host, port = loopback_image_server
    provider = _provider()

    with pytest.raises(ExternalProviderRequestError, match="unsafe image URL") as excinfo:
        provider._download_image(f"http://{host}:{port}/internal.png")
    assert _unsafe_in_chain(excinfo.value)


def test_download_image_refuses_loopback_over_https(loopback_tls_port: int) -> None:
    """The https path must be guarded too — DashScope hands back https URLs in practice.

    Asserting that `_GuardedHTTPSConnectionPool` is installed only checks a class name;
    this actually drives a connection through it. Pointing the https guard back at
    urllib3's stock `HTTPSConnection` reopens the fetch and is caught here and nowhere else.
    """
    provider = _provider()

    with pytest.raises(ExternalProviderRequestError, match="unsafe image URL") as excinfo:
        provider._download_image(f"https://127.0.0.1:{loopback_tls_port}/internal.png")
    assert _unsafe_in_chain(excinfo.value)


def test_download_image_blocks_loopback_reached_via_hostname(loopback_image_server: tuple[str, int]) -> None:
    """A hostname, not just an IP literal, must be caught — and caught at the socket.

    `_download_image` does no up-front URL inspection, so nothing resolves the host before
    `requests` does. That is what makes this rebinding-resistant: the address the guard
    judges is the one the client actually connected to, not one from an earlier lookup
    that an attacker controlling the zone could answer differently.
    """
    _host, port = loopback_image_server
    provider = _provider()

    with pytest.raises(ExternalProviderRequestError, match="unsafe image URL") as excinfo:
        provider._download_image(f"http://localhost:{port}/internal.png")
    assert _unsafe_in_chain(excinfo.value)


def test_download_image_blocks_percent_encoded_loopback_host(loopback_image_server: tuple[str, int]) -> None:
    """`requests` decodes unreserved percent-escapes in the host before connecting.

    `%6cocalhost` is not `localhost` to any check that reads the URL string, but it is the
    host `requests` actually dials, so only the socket check catches it.
    """
    _host, port = loopback_image_server
    provider = _provider()

    with pytest.raises(ExternalProviderRequestError, match="unsafe image URL") as excinfo:
        provider._download_image(f"http://%6cocalhost:{port}/internal.png")
    assert _unsafe_in_chain(excinfo.value)


def test_download_image_session_is_guarded_and_ignores_ambient_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session `_download_image` builds must be guarded *and* keep resolution in-process.

    Asserted against the session actually used rather than against `build_guarded_session`
    being called by name, so hand-rolling an equivalent session fails here too. The proxy
    half matters because a plain `requests.Session` carrying the guarded adapter still
    honours ambient `*_PROXY` variables — which hands destination resolution to the proxy
    and leaves the peer check inspecting the proxy instead of the target.
    """
    provider = _provider()
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(var, "http://198.51.100.9:8080")

    used: list[requests.Session] = []

    def capture(session: requests.Session, url: str, *_args: Any, **_kwargs: Any) -> DummyResponse:
        used.append(session)
        image_bytes = _png_bytes(_make_image("green"))
        return DummyResponse(ok=True, content=image_bytes, headers={"Content-Length": str(len(image_bytes))})

    monkeypatch.setattr("requests.Session.get", capture)
    provider._download_image("https://example.invalid/ok.png")

    assert len(used) == 1
    session = used[0]
    for prefix in ("http://", "https://"):
        adapter = session.get_adapter(prefix + "example.invalid")
        assert isinstance(adapter, ssrf.SsrfGuardedAdapter)
        assert adapter.poolmanager.pool_classes_by_scheme["http"] is ssrf._GuardedHTTPConnectionPool
        assert adapter.poolmanager.pool_classes_by_scheme["https"] is ssrf._GuardedHTTPSConnectionPool
    settings = session.merge_environment_settings("https://example.invalid/ok.png", {}, None, None, None)
    assert settings["proxies"] == {}


def test_poll_task_first_call_no_initial_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """First poll must not be preceded by a sleep — fast tasks should not pay the poll interval."""
    provider = _provider()
    request = _build_request(_build_model("qwen-image-2.0-pro"))
    image_bytes = _png_bytes(_make_image("teal"))
    image_url = "https://example.invalid/y.png"

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    def fake_get(url: str, headers: dict, timeout: int) -> DummyResponse:
        return DummyResponse(
            ok=True,
            json_data={
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [{"url": image_url}],
                }
            },
        )

    def fake_download_get(url: str, timeout: int, stream: bool = False) -> DummyResponse:
        return DummyResponse(ok=True, content=image_bytes, headers={"Content-Length": str(len(image_bytes))})

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.Session.get", lambda _session, *args, **kwargs: fake_download_get(*args, **kwargs))
    monkeypatch.setattr("time.sleep", fake_sleep)

    result = provider._poll_task(
        base_url="https://dashscope.invalid",
        headers={"Authorization": "Bearer test", "Content-Type": "application/json"},
        task_id="task-xyz",
        request=request,
        request_id="rid",
    )

    assert len(result.images) == 1
    # No sleep should have been recorded — task succeeded on the first poll.
    assert sleeps == []
