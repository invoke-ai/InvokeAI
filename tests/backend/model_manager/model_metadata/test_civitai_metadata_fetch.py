import json
from typing import Any

import pytest
from requests_testadapter import TestAdapter, TestSession

from invokeai.backend.model_manager.metadata import CivitaiMetadataFetch, UnknownMetadataException
from invokeai.backend.model_manager.metadata.fetch.civitai import is_civitai_model_version_url


def _version_response(
    *,
    trained_words: list[Any] | None = None,
    files: list[dict[str, Any]] | None = None,
    model_type: str = "LORA",
) -> bytes:
    return json.dumps(
        {
            "id": 222,
            "modelId": 111,
            "name": "v1",
            "trainedWords": trained_words if trained_words is not None else ["alpha", " beta ", "alpha", ""],
            "model": {"name": "Test LoRA", "type": model_type},
            "files": files
            if files is not None
            else [
                {
                    "name": "training.zip",
                    "type": "Training Data",
                    "downloadUrl": "https://civitai.com/api/download/models/222?type=Training%20Data",
                    "sizeKB": 1,
                },
                {
                    "name": "test-lora.safetensors",
                    "type": "Model",
                    "downloadUrl": "https://civitai.com/api/download/models/222",
                    "sizeKB": 2.5,
                    "primary": True,
                    "hashes": {"SHA256": "ABC123"},
                },
            ],
        }
    ).encode()


def _model_response() -> bytes:
    version = json.loads(_version_response())
    version.pop("model")
    return json.dumps(
        {
            "id": 111,
            "name": "Test LoRA",
            "type": "LORA",
            "modelVersions": [version],
        }
    ).encode()


def _session_for_civitai(version_response: bytes | None = None) -> TestSession:
    session = TestSession()
    version_response = version_response or _version_response()
    session.mount("https://civitai.com/api/v1/model-versions/222", TestAdapter(version_response))
    session.mount("https://civitai.com/api/v1/models/111", TestAdapter(_model_response()))
    session.mount("https://civitai.com/api/v1/model-versions/by-hash/ABC123", TestAdapter(version_response))
    return session


@pytest.mark.parametrize(
    "url",
    [
        "https://civitai.com/models/111/test-lora?modelVersionId=222",
        "https://civitai.com/api/v1/model-versions/222",
        "https://civitai.com/api/download/models/222",
        "https://civitai.com/model-versions/222",
    ],
)
def test_civitai_fetcher_parses_supported_url_shapes(url: str) -> None:
    fetcher = CivitaiMetadataFetch(_session_for_civitai())

    metadata = fetcher.from_url(url)  # type: ignore[arg-type]

    assert metadata.type == "civitai"
    assert metadata.model_id == 111
    assert metadata.model_version_id == 222
    assert metadata.trained_words == ["alpha", "beta"]


def test_civitai_fetcher_rejects_generic_model_page_without_version() -> None:
    fetcher = CivitaiMetadataFetch(_session_for_civitai())

    with pytest.raises(UnknownMetadataException):
        fetcher.from_url("https://civitai.com/models/111/test-lora")  # type: ignore[arg-type]


def test_civitai_fetcher_selects_primary_model_file_and_skips_training_data() -> None:
    fetcher = CivitaiMetadataFetch(_session_for_civitai())

    metadata = fetcher.from_url("https://civitai.com/api/v1/model-versions/222")  # type: ignore[arg-type]

    assert len(metadata.files) == 1
    assert metadata.files[0].path.as_posix() == "test-lora.safetensors"
    assert str(metadata.files[0].url) == "https://civitai.com/api/download/models/222"
    assert metadata.files[0].size == 2560
    assert metadata.files[0].sha256 == "ABC123"


def test_civitai_fetcher_supports_hash_lookup() -> None:
    fetcher = CivitaiMetadataFetch(_session_for_civitai())

    metadata = fetcher.from_hash("ABC123")

    assert metadata.model_version_id == 222
    assert metadata.files[0].path.as_posix() == "test-lora.safetensors"


def test_civitai_fetcher_handles_empty_trained_words() -> None:
    fetcher = CivitaiMetadataFetch(_session_for_civitai(_version_response(trained_words=[])))

    metadata = fetcher.from_url("https://civitai.com/api/v1/model-versions/222")  # type: ignore[arg-type]

    assert metadata.trained_words == []


def test_civitai_fetcher_splits_separator_delimited_trained_words_only() -> None:
    fetcher = CivitaiMetadataFetch(
        _session_for_civitai(_version_response(trained_words=["alpha, beta", "gamma; delta\nomega", "dark fantasy"]))
    )

    metadata = fetcher.from_url("https://civitai.com/api/v1/model-versions/222")  # type: ignore[arg-type]

    assert metadata.trained_words == ["alpha", "beta", "gamma", "delta", "omega", "dark fantasy"]


def test_civitai_fetcher_parses_stored_api_response() -> None:
    fetcher = CivitaiMetadataFetch(_session_for_civitai())

    metadata = fetcher.from_api_response(_version_response(trained_words=["alpha, beta"]).decode())

    assert metadata.model_id == 111
    assert metadata.model_version_id == 222
    assert metadata.trained_words == ["alpha", "beta"]


def test_civitai_fetcher_raises_for_404() -> None:
    session = TestSession()
    session.mount("https://civitai.com/api/v1/model-versions/222", TestAdapter(b"Not found", status=404))
    fetcher = CivitaiMetadataFetch(session)

    with pytest.raises(UnknownMetadataException):
        fetcher.from_url("https://civitai.com/api/v1/model-versions/222")  # type: ignore[arg-type]


def test_civitai_fetcher_raises_for_malformed_response() -> None:
    fetcher = CivitaiMetadataFetch(_session_for_civitai(json.dumps({"id": 222, "modelId": 111}).encode()))

    with pytest.raises(UnknownMetadataException):
        fetcher.from_url("https://civitai.com/api/v1/model-versions/222")  # type: ignore[arg-type]


def test_civitai_fetcher_raises_when_no_model_file_exists() -> None:
    fetcher = CivitaiMetadataFetch(
        _session_for_civitai(
            _version_response(
                files=[
                    {
                        "name": "training.zip",
                        "type": "Training Data",
                        "downloadUrl": "https://civitai.com/api/download/models/222?type=Training%20Data",
                        "sizeKB": 1,
                    }
                ]
            )
        )
    )

    with pytest.raises(UnknownMetadataException):
        fetcher.from_url("https://civitai.com/api/v1/model-versions/222")  # type: ignore[arg-type]


def test_civitai_model_version_url_detection_requires_concrete_version() -> None:
    assert is_civitai_model_version_url("https://civitai.com/models/111/test-lora?modelVersionId=222")
    assert is_civitai_model_version_url("https://civitai.com/api/download/models/222")
    assert not is_civitai_model_version_url("http://www.civitai.com/models/12345")
    assert not is_civitai_model_version_url("https://civitai.com/api/v1/model-versions/by-hash/ABC123")
