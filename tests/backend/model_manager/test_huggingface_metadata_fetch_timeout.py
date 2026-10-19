from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError
from requests import HTTPError, Request, Response, Timeout

from invokeai.backend.model_manager.metadata.fetch.huggingface import HuggingFaceMetadataFetch
from invokeai.backend.model_manager.metadata.metadata_base import (
    HF_METADATA_REQUEST_TIMEOUT_SECONDS,
    HuggingFaceMetadata,
    ModelMetadataUnavailableError,
    RemoteModelFile,
    UnknownMetadataException,
)
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant


def _requests_response(status_code: int, content: bytes = b"", headers: dict[str, str] | None = None) -> Response:
    response = Response()
    response.status_code = status_code
    response.url = "https://huggingface.co/api/models/org/repo"
    response._content = content
    response.headers.update(headers or {})
    response.request = Request("GET", response.url).prepare()
    return response


@pytest.mark.parametrize("use_custom_session", [False, True])
def test_model_info_requests_have_a_finite_timeout(monkeypatch: pytest.MonkeyPatch, use_custom_session: bool) -> None:
    expected_model_info = SimpleNamespace(id="org/repo", siblings=[])

    if use_custom_session:
        session = Mock()
        response = Mock(status_code=200)
        response.json.return_value = {"id": "org/repo", "siblings": []}
        session.get.return_value = response
        fetcher = HuggingFaceMetadataFetch(session)
        get_model_info = session.get
    else:
        api = Mock()
        api.model_info.return_value = expected_model_info
        monkeypatch.setattr("invokeai.backend.model_manager.metadata.fetch.huggingface.HfApi", lambda: api)
        fetcher = HuggingFaceMetadataFetch()
        get_model_info = api.model_info

    fetcher.from_id("org/repo")

    timeout = get_model_info.call_args.kwargs["timeout"]
    assert isinstance(timeout, (int, float))
    assert timeout > 0


def test_model_index_request_has_a_finite_timeout() -> None:
    session = Mock()
    response = Mock(status_code=200)
    response.json.return_value = {"text_encoder": []}
    session.get.return_value = response
    metadata = HuggingFaceMetadata(
        id="org/repo",
        name="repo",
        files=[
            RemoteModelFile(
                url="https://huggingface.co/org/repo/resolve/main/model_index.json",
                path=Path("model_index.json"),
            ),
            RemoteModelFile(
                url="https://huggingface.co/org/repo/resolve/main/text_encoder/config.json",
                path=Path("text_encoder/config.json"),
            ),
        ],
    )

    metadata.download_urls(session=session)

    timeout = session.get.call_args.kwargs["timeout"]
    assert timeout == HF_METADATA_REQUEST_TIMEOUT_SECONDS


@pytest.mark.parametrize("use_custom_session", [False, True])
def test_model_info_transport_errors_are_retryable_metadata_failures(
    monkeypatch: pytest.MonkeyPatch, use_custom_session: bool
) -> None:
    if use_custom_session:
        session = Mock()
        session.get.side_effect = Timeout("network timeout")
        fetcher = HuggingFaceMetadataFetch(session)
    else:
        api = Mock()
        api.model_info.side_effect = httpx.TimeoutException("network timeout")
        monkeypatch.setattr("invokeai.backend.model_manager.metadata.fetch.huggingface.HfApi", lambda: api)
        fetcher = HuggingFaceMetadataFetch()

    with pytest.raises(ModelMetadataUnavailableError):
        fetcher.from_id("org/repo")


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_hf_hub_http_errors_are_retryable_metadata_failures(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    request = httpx.Request("GET", "https://huggingface.co/api/models/org/repo")
    response = httpx.Response(status_code, request=request)
    api = Mock()
    api.model_info.side_effect = HfHubHTTPError("Hugging Face request failed", response=response)
    monkeypatch.setattr("invokeai.backend.model_manager.metadata.fetch.huggingface.HfApi", lambda: api)
    fetcher = HuggingFaceMetadataFetch()

    with pytest.raises(ModelMetadataUnavailableError):
        fetcher.from_id("org/repo")


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_injected_session_model_info_http_errors_are_retryable_metadata_failures(status_code: int) -> None:
    session = Mock()
    session.get.return_value = _requests_response(status_code)
    fetcher = HuggingFaceMetadataFetch(session)

    with pytest.raises(ModelMetadataUnavailableError):
        fetcher.from_id("org/repo")


def test_model_index_transport_errors_are_retryable_metadata_failures() -> None:
    session = Mock()
    session.get.side_effect = Timeout("network timeout")
    metadata = HuggingFaceMetadata(
        id="org/repo",
        name="repo",
        files=[
            RemoteModelFile(
                url="https://huggingface.co/org/repo/resolve/main/model_index.json",
                path=Path("model_index.json"),
            ),
            RemoteModelFile(
                url="https://huggingface.co/org/repo/resolve/main/text_encoder/config.json",
                path=Path("text_encoder/config.json"),
            ),
        ],
    )

    with pytest.raises(ModelMetadataUnavailableError):
        metadata.download_urls(session=session)


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_model_index_http_errors_are_retryable_metadata_failures(status_code: int) -> None:
    session = Mock()
    session.get.return_value = _requests_response(status_code)
    metadata = HuggingFaceMetadata(
        id="org/repo",
        name="repo",
        files=[
            RemoteModelFile(
                url="https://huggingface.co/org/repo/resolve/main/model_index.json",
                path=Path("model_index.json"),
            ),
            RemoteModelFile(
                url="https://huggingface.co/org/repo/resolve/main/text_encoder/config.json",
                path=Path("text_encoder/config.json"),
            ),
        ],
    )

    with pytest.raises(ModelMetadataUnavailableError):
        metadata.download_urls(session=session)


def test_model_index_nontransient_http_error_is_not_reclassified() -> None:
    session = Mock()
    session.get.return_value = _requests_response(403)
    metadata = HuggingFaceMetadata(
        id="org/repo",
        name="repo",
        files=[
            RemoteModelFile(
                url="https://huggingface.co/org/repo/resolve/main/model_index.json",
                path=Path("model_index.json"),
            ),
            RemoteModelFile(
                url="https://huggingface.co/org/repo/resolve/main/text_encoder/config.json",
                path=Path("text_encoder/config.json"),
            ),
        ],
    )

    with pytest.raises(HTTPError):
        metadata.download_urls(session=session)


def test_repository_not_found_remains_a_permanent_metadata_error() -> None:
    session = Mock()
    session.get.return_value = _requests_response(404)

    with pytest.raises(UnknownMetadataException):
        HuggingFaceMetadataFetch(session).from_id("org/repo")


def test_missing_variant_revision_falls_back_to_default_revision() -> None:
    session = Mock()
    session.get.side_effect = [
        _requests_response(404, headers={"X-Error-Code": "RevisionNotFound"}),
        _requests_response(200, b'{"id":"org/repo","siblings":[]}'),
    ]

    metadata = HuggingFaceMetadataFetch(session).from_id("org/repo", variant=ModelRepoVariant.FP16)

    assert metadata.id == "org/repo"
    assert session.get.call_count == 2
    assert "/revision/" in session.get.call_args_list[0].args[0]
    assert "/revision/" not in session.get.call_args_list[1].args[0]


def test_variant_fallback_gives_each_model_info_request_its_timeout() -> None:
    session = Mock()
    variant_not_found = _requests_response(404, headers={"X-Error-Code": "RevisionNotFound"})
    model_info_response = _requests_response(
        200,
        content=b'{"id":"org/repo","siblings":[]}',
    )
    session.get.side_effect = [variant_not_found, model_info_response]

    HuggingFaceMetadataFetch(session).from_id("org/repo", variant=ModelRepoVariant.FP16)

    assert session.get.call_count == 2
    assert [call.kwargs["timeout"] for call in session.get.call_args_list] == [
        HF_METADATA_REQUEST_TIMEOUT_SECONDS,
        HF_METADATA_REQUEST_TIMEOUT_SECONDS,
    ]
