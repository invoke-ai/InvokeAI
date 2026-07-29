"""Regression tests for Qwen3 Encoder config probing.

See https://github.com/invoke-ai/InvokeAI/issues/9090

`Qwen2.5-1.5B-Instruct` (a standalone causal LM) was being misidentified as a
`Qwen3Encoder` because the diffusers-style config check matched any directory with
`config.json` at the root and a Qwen* class name. A complete causal LM also bundles
tokenizer files at the root, while standalone text_encoder downloads do not — we
use that to disambiguate.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Checkpoint_Config,
    Qwen3Encoder_Qwen3Encoder_Config,
    _has_qwen_vl_visual_tower,
)

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


def _write_config(path: Path, hidden_size: int = 2560, architecture: str = "Qwen2ForCausalLM") -> None:
    path.write_text(json.dumps({"architectures": [architecture], "hidden_size": hidden_size}))


@pytest.mark.parametrize("tokenizer_file", ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"])
def test_complete_causal_lm_is_rejected(tokenizer_file: str) -> None:
    """A directory with config.json + tokenizer files at root is a TextLLM, not a Qwen3 encoder."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_config(root / "config.json")
        (root / tokenizer_file).write_text("{}")

        mod = MagicMock()
        mod.path = root

        with pytest.raises(NotAMatchError, match="complete causal LM"):
            Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_standalone_text_encoder_subfolder_still_matches() -> None:
    """A standalone text_encoder download (config.json at root, no tokenizer files) should still match."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_config(root / "config.json")

        mod = MagicMock()
        mod.path = root

        config = Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "qwen3_encoder"


class TestQwen3EncoderRejectsVisualTower:
    """Regression for the single-file Qwen3-VL dual-match bug.

    A nested-layout single-file Qwen3-VL 4B encoder (``model.layers.*`` + ``model.visual.*``) previously
    matched BOTH the text-only Qwen3Encoder checkpoint config and the Qwen3VLEncoder config, because the
    reject helper only looked for bare ``visual.blocks.*``. The tie-break is nondeterministic, so the
    encoder could register as Qwen3Encoder and vanish from Krea-2's Qwen3-VL dropdown - hard-blocking the
    single-file/GGUF Krea-2 install path. The reject helper now matches the nested ``model.visual.*``
    layout too, keeping the two configs mutually exclusive.
    """

    @pytest.mark.parametrize(
        "visual_key",
        [
            "model.visual.blocks.0.attn.qkv.weight",  # ComfyUI single-file nested layout
            "visual.blocks.0.attn.qkv.weight",  # already-split (transformers) layout
            "model.visual.patch_embed.proj.weight",
        ],
    )
    def test_matches_nested_and_split_visual_layouts(self, visual_key: str) -> None:
        assert _has_qwen_vl_visual_tower({visual_key: object()}) is True

    def test_ignores_text_only_decoder(self) -> None:
        assert _has_qwen_vl_visual_tower({"model.layers.0.self_attn.q_proj.weight": object()}) is False

    @patch("invokeai.backend.model_manager.configs.qwen3_encoder.raise_if_not_file")
    @patch("invokeai.backend.model_manager.configs.qwen3_encoder.raise_for_override_fields")
    def test_checkpoint_config_rejects_nested_qwen3_vl(self, _rfo, _rif) -> None:
        mod = MagicMock()
        mod.path = Path("/fake/qwen3vl_4b.safetensors")
        mod.load_state_dict.return_value = {
            "model.layers.0.self_attn.q_proj.weight": object(),
            "model.visual.blocks.0.attn.qkv.weight": object(),
        }
        with pytest.raises(NotAMatchError, match="visual tower"):
            Qwen3Encoder_Checkpoint_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_nested_text_encoder_with_root_tokenizer_still_matches() -> None:
    """A model with text_encoder/config.json should match even if tokenizer files exist at root.

    The tokenizer-at-root heuristic only applies to the standalone (root-level config.json) case.
    """
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "text_encoder").mkdir()
        _write_config(root / "text_encoder" / "config.json")
        (root / "tokenizer.json").write_text("{}")

        mod = MagicMock()
        mod.path = root

        config = Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))
        assert config.type.value == "qwen3_encoder"
