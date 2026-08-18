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

from invokeai.app.api_app import ContentTypeAwareGZipMiddleware

# Comfortably above the middleware's minimum_size, and large enough that a missing exclusion
# would be obvious rather than marginal.
BODY = b"x" * 50_000


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.get("/payload")
    def payload(content_type: str) -> Response:
        return Response(content=BODY, media_type=content_type)

    @app.get("/tiny")
    def tiny() -> Response:
        return Response(content=b"small", media_type="application/json")

    app.add_middleware(ContentTypeAwareGZipMiddleware, minimum_size=1000)
    return TestClient(app)


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


def test_clients_without_gzip_support_get_plain_bodies(client: TestClient):
    r = client.get("/payload", params={"content_type": "application/json"}, headers={"Accept-Encoding": "identity"})

    assert r.status_code == 200
    assert "content-encoding" not in r.headers
    assert r.content == BODY
