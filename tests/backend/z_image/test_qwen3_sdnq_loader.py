"""The SDNQ Qwen3 encoder's tokenizer wiring.

SDNQ exports carry packed weights only. This loader used to fetch ``Qwen/Qwen3-4B`` at load time
while the two sibling Qwen3 encoder loaders in the same module already served the vendored copy —
the drift this test exists to stop recurring.
"""

from types import SimpleNamespace

import pytest

import invokeai.backend.model_manager.load.model_loaders.z_image as z_image_loaders
from invokeai.backend.model_manager.load.model_loaders.z_image import Qwen3EncoderSDNQLoader
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.qwen3.qwen3_tokenizer import load_bundled_qwen3_tokenizer


def _config():
    from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_SDNQ_Config

    return Qwen3Encoder_SDNQ_Config.model_construct(path="unused.safetensors", name="sdnq")


def test_the_tokenizer_comes_from_the_bundle_and_never_the_network(monkeypatch) -> None:
    loader = object.__new__(Qwen3EncoderSDNQLoader)
    monkeypatch.setattr(
        z_image_loaders,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda *a, **k: pytest.fail("the SDNQ loader reached HuggingFace for a bundled tokenizer")
        ),
        raising=False,
    )

    assert loader._load_model(_config(), SubModelType.Tokenizer) is load_bundled_qwen3_tokenizer()


def test_the_served_tokenizer_encodes_a_prompt() -> None:
    """Guards the wiring end to end: a tokenizer that returns an empty sequence would otherwise
    produce an image generated from no conditioning, with nothing in the log.
    """
    loader = object.__new__(Qwen3EncoderSDNQLoader)

    assert loader._load_model(_config(), SubModelType.Tokenizer)("A cinematic photo of a cat").input_ids


def test_an_unsupported_submodel_is_refused_by_name() -> None:
    loader = object.__new__(Qwen3EncoderSDNQLoader)

    with pytest.raises(ValueError, match="Only TextEncoder and Tokenizer"):
        loader._load_model(_config(), SubModelType.VAE)
