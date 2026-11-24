import base64
import io
import json
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import requests
from PIL import Image
from urllib3.response import HTTPResponse

from invokeai.app.services.external_generation import minimax_video as minimax
from invokeai.app.services.external_generation.errors import (
    ExternalProviderNotConfiguredError,
    ExternalProviderRequestError,
)
from invokeai.app.services.session_processor.session_processor_common import CanceledException

_GLOBAL = "https://api.minimax.io"
_CHINA = "https://api.minimax.cn"
_CDN = "https://93.184.216.34/video.mp4?signature=private-test-signature"
_REQUEST = minimax.MiniMaxVideoRequest("MiniMax-H3", "Clouds drift.", "768P", 5, "16:9")
_CREATED = {"task_id": "123"}
_SUCCEEDED = {"task": {"id": "123", "status": "succeeded", "content": {"url": _CDN}}}


def _response(body: dict | bytes, status: int = 200, **headers: str) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers)
    response.raw = HTTPResponse(
        body=io.BytesIO(json.dumps(body).encode() if isinstance(body, dict) else body), preload_content=False
    )
    return response


class MiniMaxHTTP:
    def __init__(self) -> None:
        self.script: list[requests.Response | Exception] = []
        self.calls: list[requests.PreparedRequest] = []

    def send(self, request: requests.PreparedRequest) -> requests.Response:
        self.calls.append(request)
        assert self.script, f"Unexpected {request.method} request"
        result = self.script.pop(0)
        if isinstance(result, Exception):
            raise result
        result.request = request
        result.url = request.url
        return result


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MiniMaxHTTP:
    http = MiniMaxHTTP()
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", lambda _, request, **kwargs: http.send(request))
    monkeypatch.setattr(minimax, "_POLL_INTERVAL", 0)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return http


def _generate(
    request: minimax.MiniMaxVideoRequest = _REQUEST,
    origin: str | None = None,
    is_canceled: Callable[[], bool] = lambda: False,
    **kwargs: Any,
):
    return minimax.generate_minimax_video(
        api_key="test-token", base_url=origin, request=request, is_canceled=is_canceled, **kwargs
    )


@pytest.mark.parametrize(("origin", "api"), [(None, _GLOBAL), (_CHINA + "/", _CHINA)])
def test_downloads_from_both_regions_without_sending_credentials_to_cdn(
    http: MiniMaxHTTP, origin: str | None, api: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 93.184.216.34 login localuser password local-password")
    monkeypatch.setenv("NETRC", str(netrc))
    http.script = [
        _response(_CREATED),
        _response({"task": {"id": "123", "status": "queued"}}),
        _response({"task": {"id": "123", "status": "queued"}}),
        _response({}, status=503),
        requests.Timeout(),
        _response({"task": {"id": "123", "status": "running"}}),
        _response(_SUCCEEDED),
        _response(b"video content"),
    ]
    statuses = []
    with _generate(origin=origin, on_progress=statuses.append) as path:
        assert path.read_bytes() == b"video content"
    assert not path.exists()
    create, *queries, download = http.calls
    assert (create.method, create.url) == ("POST", api + "/v2/video_generation")
    assert json.loads(create.body) == {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "Clouds drift."}],
        "resolution": "768P",
        "duration": 5,
        "ratio": "16:9",
        "generate_audio": True,
    }
    assert all(call.method == "GET" and call.url == api + "/v2/query/video_generation/123" for call in queries)
    assert all(call.headers.get("Authorization") == "Bearer test-token" for call in [create, *queries])
    assert (download.method, download.url) == ("GET", _CDN)
    assert "Authorization" not in download.headers
    assert statuses == ["queued", "running", "succeeded"]


@pytest.mark.parametrize(
    ("fields", "expected_roles", "ratio"),
    [
        (("first_frame",), ["first_frame"], "adaptive"),
        (("last_frame",), ["last_frame"], "adaptive"),
        (("first_frame", "last_frame"), ["first_frame", "last_frame"], "adaptive"),
        (("reference_images",), ["reference_image", "reference_image"], "16:9"),
    ],
)
def test_image_roles_keep_pixels_and_exclude_gallery_metadata(fields, expected_roles, ratio) -> None:
    image = Image.new("RGBA", (256, 320), (127, 32, 0, 111))
    exif = Image.Exif()
    exif[270] = "private gallery metadata"
    image.info["exif"] = exif.tobytes()
    inputs = {field: (image.copy, image.copy) if field == "reference_images" else image.copy for field in fields}
    payload = json.loads(replace(_REQUEST, **inputs, generate_audio=False).encode("global", lambda: False))
    assert payload["ratio"] == ratio
    assert payload["generate_audio"] is False
    items = payload["content"][1:]
    assert [item["role"] for item in items] == expected_roles
    for item in items:
        assert item["type"] == "image_url"
        prefix, encoded = item["image_url"]["url"].split(",")
        assert prefix == "data:image/png;base64"
        raw = base64.b64decode(encoded)
        with Image.open(io.BytesIO(raw)) as result:
            assert result.getpixel((0, 0)) == (127, 32, 0, 111)
            assert not result.getexif()


@pytest.mark.parametrize(
    ("region", "model", "resolution", "duration"),
    [
        ("global", "MiniMax-H3", "480P", 4),
        ("cn", "MiniMax-H3", "2K", 15),
        ("cn", "MiniMax-H3-Lite", "480P", 4),
        ("cn", "MiniMax-H3-Highspeed", "768P", 4),
        ("cn", "MiniMax-H3-Max", "480P", 5),
        ("global", "MiniMax-H3-Max-Turbo", "768P", 15),
        ("global", "MiniMax-H3-MG", "2K", 4),
        ("global", "MiniMax-H3-NC", "768P", 4),
    ],
)
def test_supported_regional_combinations_are_submitted_unchanged(region, model, resolution, duration) -> None:
    request = replace(_REQUEST, model=model, resolution=resolution, duration=duration)
    payload = json.loads(request.encode(region, lambda: False))
    assert (payload["model"], payload["resolution"], payload["duration"]) == (model, resolution, duration)


@pytest.mark.parametrize(
    ("origin", "changes"),
    [
        (_CHINA, {"resolution": "480P"}),
        (_CHINA, {"model": "MiniMax-H3-Max-Turbo"}),
        (_CHINA, {"model": "MiniMax-H3-NC"}),
        (_CHINA, {"model": "MiniMax-H3-MG"}),
        (_GLOBAL, {"model": "MiniMax-H3-MG", "resolution": "480P"}),
        (_GLOBAL, {"model": "MiniMax-H3-Highspeed", "resolution": "2K"}),
        (_GLOBAL, {"model": "MiniMax-H3-Max", "duration": 4}),
        (_GLOBAL, {"model": "MiniMax-H3-Max-Turbo", "duration": 4}),
        (_GLOBAL, {"duration": 16}),
        (_GLOBAL, {"duration": 4.5}),
        (_GLOBAL, {"ratio": "adaptive"}),
        (_GLOBAL, {"prompt": "   "}),
        (_GLOBAL, {"prompt": "a" * 7001}),
    ],
)
def test_invalid_model_or_input_is_rejected_before_submission(http, origin, changes) -> None:
    with pytest.raises(ValueError), _generate(replace(_REQUEST, **changes), origin):
        pytest.fail("Unexpected output")
    assert not http.calls


def test_incompatible_reference_roles_dimensions_and_size_are_rejected_before_submission(http, monkeypatch) -> None:
    image = Image.new("RGB", (256, 256))
    for changes in (
        {"first_frame": image.copy, "reference_images": [image.copy]},
        {"reference_images": [image.copy] * 10},
        {"reference_images": [image.copy], "model": "MiniMax-H3-Lite", "ratio": "adaptive"},
        {"reference_images": [image.copy], "model": "MiniMax-H3-NC", "ratio": "adaptive"},
        {"last_frame": lambda: Image.new("RGB", (256, 650))},
        {"first_frame": lambda: Image.new("RGB", (255, 256))},
    ):
        with pytest.raises(ValueError), _generate(replace(_REQUEST, **changes)):
            pytest.fail("Unexpected output")
    request = replace(_REQUEST, reference_images=[image.copy])
    monkeypatch.setattr(minimax, "_MAX_REQUEST_BYTES", 100)
    with pytest.raises(ValueError, match="64 MB"), _generate(request):
        pytest.fail("Unexpected output")
    monkeypatch.setattr(minimax, "_MAX_IMAGE_BYTES", 100)
    with pytest.raises(ValueError, match="30 MB"), _generate(request):
        pytest.fail("Unexpected output")
    assert not http.calls


@pytest.mark.parametrize(
    "origin", ["http://api.minimax.cn", "https://api.minimax.io.example", "https://api.minimax.cn/v1", ""]
)
def test_api_key_only_targets_official_https_origins(http, origin) -> None:
    with pytest.raises(ExternalProviderNotConfiguredError), _generate(origin=origin):
        pytest.fail("Unexpected output")
    assert not http.calls


@pytest.mark.parametrize(
    "first_response",
    [
        requests.Timeout("private-test-signature"),
        _response({}, 307, Location=_CDN),
        _response({"task_id": "../changed"}),
        _response({}, 500),
    ],
)
def test_never_follows_redirect_or_resubmits_an_ambiguous_create(http, first_response) -> None:
    http.script = [first_response]
    with pytest.raises(ExternalProviderRequestError) as error, _generate():
        pytest.fail("Unexpected output")
    assert "private-test-signature" not in str(error.value)
    assert [call.method for call in http.calls] == ["POST"]


@pytest.mark.parametrize(
    "task",
    [
        {"id": "123", "status": "failed", "error": {"message": _CDN}},
        {"id": "123", "status": "cancelled"},
        {"id": "123", "status": "unknown"},
        {"id": "123", "status": "succeeded", "content": {}},
        {"id": "wrong-task", "status": "succeeded", "content": {"url": _CDN}},
    ],
)
def test_invalid_or_failed_status_stops_before_download_without_echoing_sensitive_data(http, task) -> None:
    http.script = [_response(_CREATED), _response({"task": task})]
    with pytest.raises(ExternalProviderRequestError) as error, _generate():
        pytest.fail("Unexpected output")
    assert "123" in str(error.value)
    assert "signature" not in str(error.value)
    assert [call.method for call in http.calls] == ["POST", "GET"]


@pytest.mark.parametrize(
    "url", ["http://93.184.216.34/result", "https://127.0.0.1/private", "https://user:password@93.184.216.34/result"]
)
def test_unsafe_result_urls_are_rejected_before_download(http, url) -> None:
    http.script = [
        _response(_CREATED),
        _response({"task": {"id": "123", "status": "succeeded", "content": {"url": url}}}),
    ]
    with pytest.raises(ExternalProviderRequestError, match="unsafe URL"), _generate():
        pytest.fail("Unexpected output")
    assert len(http.calls) == 2


@pytest.mark.parametrize(
    "download",
    [
        _response(b""),
        _response(b"tiny", **{"Content-Length": "12345"}),
        _response(b"0123456789"),
        _response(b"", 302, Location="https://127.0.0.1/"),
    ],
)
def test_rejected_or_partial_download_is_cleaned_up(http, tmp_path, monkeypatch, download) -> None:
    monkeypatch.setattr(minimax, "MAX_VIDEO_BYTES", 9)
    http.script = [_response(_CREATED), _response(_SUCCEEDED), download]
    with pytest.raises(ExternalProviderRequestError), _generate():
        pytest.fail("Unexpected output")
    assert not list(tmp_path.iterdir())
    assert len(http.calls) == 3


def test_cancellation_during_poll_or_download_stops_locally_and_removes_partial_file(http, tmp_path) -> None:
    http.script = [_response(_CREATED), _response({"task": {"id": "123", "status": "queued"}})]
    with pytest.raises(CanceledException), _generate(is_canceled=lambda: len(http.calls) == 2):
        pytest.fail("Unexpected output")
    assert [call.method for call in http.calls] == ["POST", "GET"]

    http.script = [_response(_CREATED), _response(_SUCCEEDED), _response(b"x" * 70000)]
    with (
        pytest.raises(CanceledException),
        _generate(is_canceled=lambda: any(path.stat().st_size for path in tmp_path.rglob("video.mp4"))),
    ):
        pytest.fail("Unexpected output")
    assert not list(tmp_path.iterdir())


def test_timed_out_task_is_recoverable_by_id_and_not_resubmitted(http, monkeypatch) -> None:
    http.script = [_response(_CREATED)]
    monkeypatch.setattr(minimax, "_POLL_TIMEOUT", 0)
    with pytest.raises(ExternalProviderRequestError, match="Timed out.*123"), _generate():
        pytest.fail("Unexpected output")
    assert [call.method for call in http.calls] == ["POST"]


def test_stops_after_repeated_query_failures_and_preserves_the_paid_task_id(http) -> None:
    http.script = [_response(_CREATED), _response({}, 429), requests.Timeout(), _response({}, 503)]
    with pytest.raises(ExternalProviderRequestError, match="Unable to query MiniMax task 123"), _generate():
        pytest.fail("Unexpected output")
    assert [call.method for call in http.calls] == ["POST", "GET", "GET", "GET"]


def test_redirect_body_is_never_buffered_before_applying_the_download_cap(http, monkeypatch) -> None:
    redirect = _response(b"unbounded body", 302, Location="https://127.0.0.1/")
    monkeypatch.setattr(redirect, "iter_content", lambda *a, **kw: pytest.fail("Redirect body was read"))
    http.script = [_response(_CREATED), _response(_SUCCEEDED), redirect]
    with pytest.raises(ExternalProviderRequestError, match="redirect"), _generate():
        pytest.fail("Unexpected output")


def test_cancelled_request_does_not_load_inputs_or_submit_a_paid_task(http) -> None:
    request = replace(_REQUEST, first_frame=lambda: pytest.fail("Loaded a cancelled input"))
    with pytest.raises(CanceledException), _generate(request, is_canceled=lambda: True):
        pytest.fail("Unexpected output")
    assert not http.calls


def test_caller_ingest_failure_cleans_up_the_download(http, tmp_path) -> None:
    http.script = [_response(_CREATED), _response(_SUCCEEDED), _response(b"not a playable video")]
    with pytest.raises(ValueError, match="invalid media"), _generate() as path:
        assert path.exists()
        raise ValueError("invalid media")
    assert not list(tmp_path.iterdir())
