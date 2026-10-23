"""Installing a Modular Diffusers repo by id downloads its components and nothing else from the repo."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
import requests_testadapter

from invokeai.backend.model_manager.metadata.metadata_base import (
    HuggingFaceMetadata,
    ModelMetadataUnavailableError,
    RemoteModelFile,
)

# The component part of black-forest-labs/flux-3-image-klein's modular_model_index.json (2026-10-15): the tokenizer
# loads from `text_encoder/`, the folder that holds its files.
_FLUX3_MODULAR_INDEX = {
    "_blocks_class_name": "Flux3DistilledAutoBlocks",
    "_class_name": "Flux3DistilledModularPipeline",
    "_diffusers_version": "0.41.0.dev0",
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration", {"subfolder": "text_encoder"}],
    "tokenizer": ["transformers", "Qwen3VLProcessor", {"subfolder": "text_encoder"}],
    "transformer": ["diffusers", "Flux3Transformer2DModel", {"subfolder": "transformer"}],
    "vae": ["diffusers", "AutoencoderKLFlux2", {"subfolder": "vae"}],
}

_REPO_FILES = [
    "modular_model_index.json",
    # BFL's own single-file package and quantized builds sit beside the pipeline in the same repo.
    "config.json",
    "model.safetensors",
    "ae.safetensors",
    "variants/fp8r/model.safetensors",
    "variants/fp8r/config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/tokenizer.json",
    "text_encoder/merges.txt",
    "text_encoder/chat_template.json",
]


# The commit an install pinned; the modular index must come from it, as the files do.
_PINNED_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_INDEX_URL = (
    f"https://huggingface.co/black-forest-labs/flux-3-image-klein/resolve/{_PINNED_COMMIT}/modular_model_index.json"
)
_COMPONENT_FILES = {
    f"flux-3-image-klein/{rfilename}"
    for rfilename in _REPO_FILES
    if rfilename.split("/")[0] in {"transformer", "vae", "text_encoder"}
} | {"flux-3-image-klein/modular_model_index.json"}


def _flux3_metadata() -> HuggingFaceMetadata:
    # HuggingFaceMetadataFetch.from_id roots every file path at the repo name.
    return HuggingFaceMetadata(
        id="black-forest-labs/flux-3-image-klein",
        name="flux-3-image-klein",
        files=[
            RemoteModelFile(
                url=f"https://huggingface.co/black-forest-labs/flux-3-image-klein/resolve/{_PINNED_COMMIT}/{rfilename}",
                path=Path("flux-3-image-klein", rfilename),
            )
            for rfilename in _REPO_FILES
        ],
    )


def test_a_modular_repo_downloads_only_its_component_folders() -> None:
    session = Mock()
    response = Mock(status_code=200)
    response.json.return_value = _FLUX3_MODULAR_INDEX
    session.get.return_value = response

    selected = {file.path.as_posix() for file in _flux3_metadata().download_urls(session=session)}

    assert session.get.call_args.args[0] == _INDEX_URL
    assert selected == _COMPONENT_FILES


class _GatedIndex(requests_testadapter.TestAdapter):
    """Serves the index only to a request with the granted token, as a gated repo does; records each Authorization."""

    def __init__(self) -> None:
        super().__init__(json.dumps(_FLUX3_MODULAR_INDEX).encode())
        self.authorizations: list[str | None] = []

    def send(self, request, **kwargs):
        self.authorizations.append(request.headers.get("Authorization"))
        if request.headers.get("Authorization") != "Bearer hf_granted":
            return requests_testadapter.TestAdapter(b"", status=401).send(request, **kwargs)
        return super().send(request, **kwargs)


def _gated_index_session() -> tuple[requests_testadapter.TestSession, _GatedIndex]:
    session = requests_testadapter.TestSession()
    index = _GatedIndex()
    session.mount(_INDEX_URL, index)
    return session, index


def test_a_gated_modular_index_is_read_with_the_token_for_its_url() -> None:
    session, index = _gated_index_session()

    files = _flux3_metadata().download_urls(session=session, access_token_for={_INDEX_URL: "hf_granted"}.get)

    assert index.authorizations == ["Bearer hf_granted"]
    assert {file.path.as_posix() for file in files} == _COMPONENT_FILES


def test_without_a_token_the_modular_index_request_is_unauthenticated_and_its_refusal_is_final() -> None:
    session, index = _gated_index_session()

    with pytest.raises(requests.HTTPError) as refusal:
        _flux3_metadata().download_urls(session=session)

    assert index.authorizations == [None]
    assert refusal.value.response.status_code == 401


def test_a_transient_modular_index_failure_is_retryable() -> None:
    response = requests.Response()
    response.status_code = 503
    session = Mock()
    session.get.return_value = response

    with pytest.raises(ModelMetadataUnavailableError):
        _flux3_metadata().download_urls(session=session)
