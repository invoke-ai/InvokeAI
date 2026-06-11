"""Body-delivery tests for the image-serving routes (get_image_full / get_image_thumbnail).

These routes stream instead of buffering: FileResponse when the backend exposes a
local path, otherwise StreamingResponse over open_stream(). Mount only the images
router so the unauthenticated content routes can be exercised in isolation.
"""

import io
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers.images import images_router
from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException


def _client(monkeypatch, images: MagicMock, *, raise_server_exceptions: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(images_router)
    invoker = MagicMock()
    invoker.services.images = images
    monkeypatch.setattr(ApiDependencies, "invoker", invoker, raising=False)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def test_get_image_full_uses_fileresponse_for_local_path(tmp_path, monkeypatch):
    data = b"local-png-bytes"
    p = tmp_path / "img.png"
    p.write_bytes(data)
    images = MagicMock()
    images.get_local_path.return_value = p
    client = _client(monkeypatch, images)

    r = client.get("/v1/images/i/img.png/full")

    assert r.status_code == 200
    assert r.content == data
    assert r.headers["content-type"].startswith("image/png")
    # A real local path must be streamed via FileResponse, not buffered via open_stream.
    images.open_stream.assert_not_called()


def test_get_image_full_streams_when_no_local_path(monkeypatch):
    data = b"s3-png-bytes"
    images = MagicMock()
    images.get_local_path.return_value = None
    images.open_stream.return_value = io.BytesIO(data)
    client = _client(monkeypatch, images)

    r = client.get("/v1/images/i/img.png/full")

    assert r.status_code == 200
    assert r.content == data
    assert r.headers["content-type"].startswith("image/png")


def test_get_image_thumbnail_streams(monkeypatch):
    data = b"webp-bytes"
    images = MagicMock()
    images.get_local_path.return_value = None
    images.open_stream.return_value = io.BytesIO(data)
    client = _client(monkeypatch, images)

    r = client.get("/v1/images/i/img.png/thumbnail")

    assert r.status_code == 200
    assert r.content == data
    assert r.headers["content-type"].startswith("image/webp")


def test_get_image_full_returns_404_when_missing(monkeypatch):
    images = MagicMock()
    images.get_local_path.return_value = None
    images.open_stream.side_effect = ImageFileNotFoundException()
    client = _client(monkeypatch, images)

    r = client.get("/v1/images/i/missing.png/full")

    assert r.status_code == 404


def test_get_image_full_falls_back_to_stream_when_local_path_missing(tmp_path, monkeypatch):
    # get_local_path returns a path that does not exist -> must not FileResponse a
    # missing file; falls back to open_stream (existence check runs in threadpool).
    data = b"streamed"
    images = MagicMock()
    images.get_local_path.return_value = tmp_path / "does-not-exist.png"
    images.open_stream.return_value = io.BytesIO(data)
    client = _client(monkeypatch, images)

    r = client.get("/v1/images/i/x.png/full")

    assert r.status_code == 200
    assert r.content == data
    images.open_stream.assert_called_once()


def test_get_image_full_unexpected_error_is_500_not_404(monkeypatch):
    # A non-not-found error must surface as 500, not be masked as 404.
    images = MagicMock()
    images.get_local_path.return_value = None
    images.open_stream.side_effect = RuntimeError("backend exploded")
    client = _client(monkeypatch, images, raise_server_exceptions=False)

    r = client.get("/v1/images/i/x.png/full")

    assert r.status_code == 500


def test_get_image_full_ignores_directory_local_path(tmp_path, monkeypatch):
    # A directory path must never be served via FileResponse (is_file() guards
    # it); it falls back to streaming instead.
    data = b"streamed-not-a-dir"
    images = MagicMock()
    images.get_local_path.return_value = tmp_path  # a directory, not a file
    images.open_stream.return_value = io.BytesIO(data)
    client = _client(monkeypatch, images)

    r = client.get("/v1/images/i/x.png/full")

    assert r.status_code == 200
    assert r.content == data
    images.open_stream.assert_called_once()


def test_inline_content_disposition_is_injection_safe():
    from invokeai.app.api.routers.images import _inline_content_disposition

    header = _inline_content_disposition('evil".png\r\nSet-Cookie: x=y')
    # No raw quotes/CRLF can leak into the header (percent-encoded via filename*).
    assert "\r" not in header and "\n" not in header
    assert header.startswith("inline; filename*=utf-8''")


def test_inline_content_disposition_plain_name():
    from invokeai.app.api.routers.images import _inline_content_disposition

    assert _inline_content_disposition("abc-123.png") == 'inline; filename="abc-123.png"'


def test_inline_content_disposition_escapes_path_separators():
    from invokeai.app.api.routers.images import _inline_content_disposition

    # A name with "/" must not be emitted as a path-like quoted filename; it
    # falls back to percent-encoded filename* (RFC 5987).
    assert _inline_content_disposition("a/b.png") == "inline; filename*=utf-8''a%2Fb.png"
