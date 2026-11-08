"""TextLLM classification of Gemma-4 multimodal checkpoints (the LTX-2.5 prompt enhancer, Gemma-4-E2B).

Stock Gemma-4 folders name `Gemma4ForConditionalGeneration`, which `AutoModelForCausalLM` loads on the
pinned transformers, so they classify as TextLLM. Whether Expand Prompt may send them an image depends
on the folder carrying both the vision config and the processor config that `AutoProcessor` needs.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Diffusers_Config

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


def _write_model(root: Path, *, architecture: str, vision: bool, processor: bool) -> MagicMock:
    config: dict[str, object] = {"architectures": [architecture], "model_type": "gemma4", "text_config": {}}
    if vision:
        config["vision_config"] = {}
    root.joinpath("config.json").write_text(json.dumps(config))
    root.joinpath("tokenizer.json").write_text("{}")
    if processor:
        root.joinpath("processor_config.json").write_text("{}")
    mod = MagicMock()
    mod.path = root
    return mod


def test_gemma4_with_vision_and_processor_supports_images(tmp_path: Path) -> None:
    mod = _write_model(tmp_path, architecture="Gemma4ForConditionalGeneration", vision=True, processor=True)
    config = TextLLM_Diffusers_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
    assert config.supports_images is True


@pytest.mark.parametrize(("vision", "processor"), [(True, False), (False, True)])
def test_gemma4_without_both_image_parts_is_text_only(tmp_path: Path, vision: bool, processor: bool) -> None:
    mod = _write_model(tmp_path, architecture="Gemma4ForConditionalGeneration", vision=vision, processor=processor)
    config = TextLLM_Diffusers_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
    assert config.supports_images is False


def test_causal_lm_with_image_parts_is_still_text_only(tmp_path: Path) -> None:
    """Only the allow-listed class is known to load with its vision tower through the generic loader."""
    mod = _write_model(tmp_path, architecture="SomeVisionForCausalLM", vision=True, processor=True)
    config = TextLLM_Diffusers_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
    assert config.supports_images is False


def test_unlisted_conditional_generation_is_not_a_text_llm(tmp_path: Path) -> None:
    """The generic loader is not known to build other vision-language classes, so they are not claimed."""
    mod = _write_model(tmp_path, architecture="LlavaOnevisionForConditionalGeneration", vision=True, processor=True)
    with pytest.raises(NotAMatchError, match="not a causal language model"):
        TextLLM_Diffusers_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
