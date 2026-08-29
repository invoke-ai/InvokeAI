"""GZip must skip response types that are already compressed.

Starlette's GZipMiddleware compresses everything except `text/event-stream`. The gallery
serves PNG, WebP and MP4 bytes, which are already deflate-compressed: gzipping a 3 MB PNG
costs ~52ms of event-loop time and returns a *larger* body. With auto-switch on, the UI
fetches the full image after every generated image, so that cost lands repeatedly during a
batch.

These tests pin which content types are compressed, using a standalone app so they exercise
the middleware rather than the whole API surface.
"""

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from invokeai.app.api_app import ContentTypeAwareGZipMiddleware, configure_gzip

# Comfortably above the middleware's minimum_size, and large enough that a missing exclusion
# would be obvious rather than marginal.
BODY = b"x" * 50_000


def _build_app(compresslevel: int) -> TestClient:
    app = FastAPI()

    @app.get("/payload")
    def payload(content_type: str) -> Response:
        return Response(content=BODY, media_type=content_type)

    @app.get("/tiny")
    def tiny() -> Response:
        return Response(content=b"small", media_type="application/json")

    configure_gzip(app, compresslevel)
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return _build_app(compresslevel=1)


@pytest.mark.parametrize(
    "content_type",
    [
        "application/json",
        "text/html; charset=utf-8",
        "text/css",
        "application/javascript",
        "image/svg+xml",
    ],
)
def test_compressible_types_are_compressed(client: TestClient, content_type: str):
    r = client.get("/payload", params={"content_type": content_type}, headers={"Accept-Encoding": "gzip"})

    assert r.status_code == 200
    assert r.headers["content-encoding"] == "gzip"
    # httpx decodes transparently, so this also proves the compressed body round-trips.
    assert r.content == BODY
    assert int(r.headers["content-length"]) < len(BODY)


@pytest.mark.parametrize(
    "content_type",
    [
        "image/png",
        "image/webp",
        "image/jpeg",
        "video/mp4",
        "application/zip",
        "application/octet-stream",
        # `text/` prefix matches, but compressing an event stream withholds events until the
        # compressor flushes.
        "text/event-stream",
    ],
)
def test_already_compressed_types_are_passed_through(client: TestClient, content_type: str):
    r = client.get("/payload", params={"content_type": content_type}, headers={"Accept-Encoding": "gzip"})

    assert r.status_code == 200
    assert "content-encoding" not in r.headers, f"{content_type} should not be gzipped"
    assert r.content == BODY


def test_small_responses_are_left_alone(client: TestClient):
    r = client.get("/tiny", headers={"Accept-Encoding": "gzip"})

    assert r.status_code == 200
    assert "content-encoding" not in r.headers


def test_the_real_app_uses_the_content_type_aware_middleware():
    """Without this, the tests above would keep passing while the app served gzipped PNGs."""
    from starlette.middleware.gzip import GZipMiddleware

    from invokeai.app.api_app import app

    installed = [m.cls for m in app.user_middleware]
    assert ContentTypeAwareGZipMiddleware in installed
    assert GZipMiddleware not in installed


@pytest.mark.parametrize("content_type", ["application/json", "text/html; charset=utf-8", "image/svg+xml"])
def test_level_zero_disables_compression_entirely(content_type: str):
    """`http_compression_level: 0` is how a deployment behind a compressing proxy opts out."""
    disabled = _build_app(compresslevel=0)

    r = disabled.get("/payload", params={"content_type": content_type}, headers={"Accept-Encoding": "gzip"})

    assert r.status_code == 200
    assert "content-encoding" not in r.headers
    assert r.content == BODY


def test_level_zero_leaves_the_middleware_out():
    """Installing it at level 0 would still buffer every response through the responder."""
    app = FastAPI()
    configure_gzip(app, 0)

    assert [m.cls for m in app.user_middleware] == []


def test_the_configured_level_reaches_the_compressor():
    """A level that is accepted but ignored would silently keep the old 90ms-per-response cost."""
    # Repetitive but varied, so the higher level's larger window actually finds more matches —
    # `b"x" * n` would compress identically at every level and prove nothing.
    body = "".join(f'"{i:08x}-image-{i % 7}.png",' for i in range(20_000)).encode()

    sizes: dict[int, int] = {}
    for level in (1, 9):
        app = FastAPI()

        @app.get("/names")
        def names() -> Response:
            return Response(content=body, media_type="application/json")

        configure_gzip(app, level)
        r = TestClient(app).get("/names", headers={"Accept-Encoding": "gzip"})

        assert r.headers["content-encoding"] == "gzip"
        assert r.content == body
        sizes[level] = int(r.headers["content-length"])

    assert sizes[9] < sizes[1], "compresslevel is not being passed through to the compressor"


def test_the_real_app_uses_the_configured_level():
    from invokeai.app.api_app import app, app_config

    installed = [m for m in app.user_middleware if m.cls is ContentTypeAwareGZipMiddleware]
    assert len(installed) == 1
    assert installed[0].kwargs["compresslevel"] == app_config.http_compression_level


def test_the_default_level_is_unchanged():
    """Adding the setting must not change what existing installs do — the default is still 9."""
    from invokeai.app.services.config.config_default import InvokeAIAppConfig

    assert InvokeAIAppConfig().http_compression_level == 9


@pytest.mark.parametrize("level", [-1, 10])
def test_out_of_range_levels_are_rejected(level: int):
    """zlib would raise deep inside the responder, mid-response, rather than at startup."""
    from pydantic import ValidationError

    from invokeai.app.services.config.config_default import InvokeAIAppConfig

    with pytest.raises(ValidationError):
        InvokeAIAppConfig(http_compression_level=level)


def test_the_starlette_contract_the_responder_relies_on_still_holds():
    """`_ContentTypeAwareGZipResponder` widens Starlette's own `content_type_is_excluded` right
    after the base class has computed it. That is only safe while Starlette keeps exposing the
    flag and keeps *buffering* `http.response.start` rather than forwarding it immediately. If an
    upgrade changes either, the widening is silently lost and the app compresses PNGs again —
    this test names the cause instead of leaving a puzzling content-type failure.
    """
    import asyncio

    from starlette.middleware.gzip import IdentityResponder
    from starlette.types import Message

    responder = IdentityResponder(app=None, minimum_size=1)  # type: ignore[arg-type]
    assert hasattr(responder, "content_type_is_excluded"), (
        "Starlette no longer exposes `content_type_is_excluded`; the exclusion must be reimplemented"
    )

    forwarded: list[Message] = []

    async def capture(message: Message) -> None:
        forwarded.append(message)

    responder.send = capture
    start: Message = {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"image/png")]}
    asyncio.run(responder.send_with_compression(start))

    assert forwarded == [], (
        "Starlette now forwards http.response.start immediately; the content type must be inspected "
        "before delegating to the base responder"
    )


def test_clients_without_gzip_support_get_plain_bodies(client: TestClient):
    r = client.get("/payload", params={"content_type": "application/json"}, headers={"Accept-Encoding": "identity"})

    assert r.status_code == 200
    assert "content-encoding" not in r.headers
    assert r.content == BODY
