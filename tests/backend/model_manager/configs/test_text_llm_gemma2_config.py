"""Regression tests for TextLLM classification of Gemma 2 causal LMs.

PiD added a dedicated Gemma2 encoder config. To keep automatic classification from producing a
generic ``text_llm`` entry for a Gemma2 directory (which also matches the encoder config), TextLLM
defers on ``Gemma2ForCausalLM``. That deferral must only apply to *automatic* classification: when a
user explicitly asks for ``type=text_llm`` the generic causal-LM loader still supports the model, so
the explicit request must remain valid.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Diffusers_Config
from invokeai.backend.model_manager.taxonomy import ModelType

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test-model",
    "file_size": 1000,
    "name": "test-model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _make_causal_lm_dir(root: Path, *, architecture: str) -> MagicMock:
    root.joinpath("config.json").write_text(json.dumps({"architectures": [architecture]}))
    root.joinpath("tokenizer.json").write_text("{}")
    mod = MagicMock()
    mod.path = root
    return mod


def test_gemma2_defers_during_automatic_classification() -> None:
    """Without an explicit type, a Gemma2 directory defers to the dedicated encoder config."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_causal_lm_dir(Path(tmpdir), architecture="Gemma2ForCausalLM")
        with pytest.raises(NotAMatchError, match="dedicated encoder config"):
            TextLLM_Diffusers_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_gemma2_matches_when_explicitly_requested() -> None:
    """An explicit type=text_llm keeps Gemma2 causal LMs usable as generic TextLLM models."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_causal_lm_dir(Path(tmpdir), architecture="Gemma2ForCausalLM")
        fields = dict(_OVERRIDE_FIELDS, type=ModelType.TextLLM)
        config = TextLLM_Diffusers_Config.from_model_on_disk(mod, fields)
        assert config.type == ModelType.TextLLM


def test_non_specialised_causal_lm_matches_automatically() -> None:
    """A plain causal LM (e.g. Llama) is still classified as TextLLM automatically."""
    with TemporaryDirectory() as tmpdir:
        mod = _make_causal_lm_dir(Path(tmpdir), architecture="LlamaForCausalLM")
        config = TextLLM_Diffusers_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type == ModelType.TextLLM
