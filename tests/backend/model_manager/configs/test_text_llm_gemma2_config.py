"""Regression tests for TextLLM classification of Gemma 2 causal LMs.

PiD added a dedicated Gemma2 encoder config that only accepts Gemma-2-2b (2304-dim hidden state).
Automatic classification must:
  - defer a 2304-dim Gemma2 to the encoder config (so it becomes Gemma2Encoder, not a generic TextLLM),
  - but keep larger Gemma 2 variants (9B=3584, 27B=4608) — which the encoder config rejects — classifiable
    as a generic TextLLM instead of falling through to Unknown, and
  - always honor an explicit type=text_llm request (the generic AutoModelForCausalLM loader supports these).
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


def _write_causal_lm(root: Path, *, architecture: str, hidden_size: int | None = None) -> None:
    config: dict[str, object] = {"architectures": [architecture]}
    if hidden_size is not None:
        config["hidden_size"] = hidden_size
    root.joinpath("config.json").write_text(json.dumps(config))
    root.joinpath("tokenizer.json").write_text("{}")


def _mock_mod(root: Path) -> MagicMock:
    mod = MagicMock()
    mod.path = root
    return mod


def test_gemma2_2b_defers_during_automatic_classification() -> None:
    """Without an explicit type, a 2304-dim Gemma2 directory defers to the PiD encoder config."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_causal_lm(root, architecture="Gemma2ForCausalLM", hidden_size=2304)
        with pytest.raises(NotAMatchError, match="PiD encoder config"):
            TextLLM_Diffusers_Config.from_model_on_disk(_mock_mod(root), dict(_OVERRIDE_FIELDS))


@pytest.mark.parametrize("hidden_size", [3584, 4608])
def test_larger_gemma2_variants_remain_text_llm(hidden_size: int) -> None:
    """Gemma 2 9B/27B are rejected by the encoder config, so they must stay classifiable as TextLLM."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_causal_lm(root, architecture="Gemma2ForCausalLM", hidden_size=hidden_size)
        config = TextLLM_Diffusers_Config.from_model_on_disk(_mock_mod(root), dict(_OVERRIDE_FIELDS))
        assert config.type == ModelType.TextLLM


def test_gemma2_matches_when_explicitly_requested() -> None:
    """An explicit type=text_llm keeps even a 2304-dim Gemma2 usable as a generic TextLLM model."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_causal_lm(root, architecture="Gemma2ForCausalLM", hidden_size=2304)
        fields = dict(_OVERRIDE_FIELDS, type=ModelType.TextLLM)
        config = TextLLM_Diffusers_Config.from_model_on_disk(_mock_mod(root), fields)
        assert config.type == ModelType.TextLLM


def test_non_specialised_causal_lm_matches_automatically() -> None:
    """A plain causal LM (e.g. Llama) is still classified as TextLLM automatically."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_causal_lm(root, architecture="LlamaForCausalLM")
        config = TextLLM_Diffusers_Config.from_model_on_disk(_mock_mod(root), dict(_OVERRIDE_FIELDS))
        assert config.type == ModelType.TextLLM


@pytest.mark.parametrize(
    ("hidden_size", "expected_type"),
    [
        (2304, ModelType.Gemma2Encoder),
        (3584, ModelType.TextLLM),
        (4608, ModelType.TextLLM),
    ],
)
def test_factory_classifies_gemma2_by_size(hidden_size: int, expected_type: ModelType) -> None:
    """End-to-end: 2304 -> Gemma2Encoder; larger variants stay TextLLM (never Unknown)."""
    from invokeai.backend.model_manager.configs.factory import ModelConfigFactory

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_causal_lm(root, architecture="Gemma2ForCausalLM", hidden_size=hidden_size)
        result = ModelConfigFactory.from_model_on_disk(root, allow_unknown=True)
        assert result.config is not None
        assert result.config.type == expected_type
