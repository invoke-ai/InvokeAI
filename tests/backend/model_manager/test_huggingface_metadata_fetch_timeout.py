from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from requests import Timeout

from invokeai.backend.model_manager.metadata.fetch.huggingface import HuggingFaceMetadataFetch
from invokeai.backend.model_manager.metadata.metadata_base import (
    HuggingFaceMetadata,
    ModelMetadataUnavailableError,
    RemoteModelFile,
)


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
    assert isinstance(timeout, (int, float))
    assert timeout > 0


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
