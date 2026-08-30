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
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Checkpoint_Config,
    Qwen3Encoder_GGUF_Config,
    Qwen3Encoder_Qwen3Encoder_Config,
    _has_gemma2_keys,
    _has_qwen_vl_visual_tower,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk

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


def test_has_gemma2_keys_detects_post_norms() -> None:
    """Gemma-2/3 GGUFs are identified by their post-attention / post-feedforward norms."""
    assert _has_gemma2_keys({"token_embd.weight": 0, "blk.0.post_attention_norm.weight": 0}) is True
    assert _has_gemma2_keys({"blk.5.post_ffw_norm.weight": 0}) is True
    # A Qwen3-style state dict (attn q/k norms, no post-norms) is not mistaken for Gemma.
    assert _has_gemma2_keys({"token_embd.weight": 0, "blk.0.attn_q_norm.weight": 0}) is False


def test_gguf_config_rejects_gemma_state_dict() -> None:
    """A Gemma-2 GGUF satisfies the generic Qwen3 key heuristic but must be rejected by the Qwen3 GGUF
    config so it is not re-identified as a Qwen3 encoder (it should classify as Gemma2Encoder)."""
    mod = MagicMock()
    mod.load_state_dict.return_value = {
        "token_embd.weight": 0,
        "blk.0.attn_norm.weight": 0,
        "blk.0.post_attention_norm.weight": 0,
        "blk.0.post_ffw_norm.weight": 0,
    }
    with pytest.raises(NotAMatchError, match="Gemma-2"):
        Qwen3Encoder_GGUF_Config._validate_looks_like_qwen3_model(mod)


class TestShardedQwen3EncoderFolder:
    """A sharded text_encoder folder must still identify as a Qwen3 encoder.

    The starter-model download
    `black-forest-labs/FLUX.2-klein-{4B,9B}::text_encoder+tokenizer` lands the encoder as several
    `model-0000N-of-0000M.safetensors` shards. The SDNQ rejection guard used to call
    `mod.load_state_dict()`, which raises ValueError ("Multiple weight files found") rather than a
    NotAMatchError when a folder holds more than one weight file. That aborted this config's probe,
    so the model was stored as `unknown`.
    """

    @staticmethod
    def _make_sharded_encoder(root: Path, *, sdnq: bool = False) -> Path:
        text_encoder = root / "text_encoder"
        text_encoder.mkdir(parents=True)
        _write_config(text_encoder / "config.json", hidden_size=4096, architecture="Qwen3ForCausalLM")

        shards: list[dict[str, torch.Tensor]] = [
            {"model.embed_tokens.weight": torch.zeros(8, 4096, dtype=torch.uint8 if sdnq else torch.float32)},
            {"model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=torch.uint8 if sdnq else torch.float32)},
        ]
        if sdnq:
            shards[1]["model.layers.0.self_attn.q_proj.scale"] = torch.zeros(8, 1, dtype=torch.float32)

        for i, shard in enumerate(shards, start=1):
            save_file(shard, str(text_encoder / f"model-0000{i}-of-00002.safetensors"))

        # Tokenizer files live in their own subfolder for the `text_encoder+tokenizer` download layout.
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer_config.json").write_text("{}")
        return root

    def test_sharded_encoder_matches(self, tmp_path: Path) -> None:
        root = self._make_sharded_encoder(tmp_path / "klein-9b-encoder")
        config = Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))

        assert config.type.value == "qwen3_encoder"
        assert config.variant.value == "qwen3_8b"

    def test_sharded_sdnq_encoder_is_still_rejected(self, tmp_path: Path) -> None:
        """The shard-safe check must keep detecting SDNQ weight+scale pairs across shards."""
        root = self._make_sharded_encoder(tmp_path / "klein-9b-encoder-sdnq", sdnq=True)
        with pytest.raises(NotAMatchError, match="SDNQ"):
            Qwen3Encoder_Qwen3Encoder_Config.from_model_on_disk(ModelOnDisk(root), dict(_OVERRIDE_FIELDS))
