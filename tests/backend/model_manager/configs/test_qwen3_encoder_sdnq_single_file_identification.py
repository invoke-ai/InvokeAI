"""Tests that Qwen3Encoder_SDNQ_Config (single-file) only accepts a real Qwen3 encoder.

A single-file SDNQ Qwen checkpoint has no config.json, so identification must key on the state dict.
`_has_qwen3_keys` alone is generic across Qwen2 / Qwen3 / Qwen-VL (same `model.layers.*` /
`model.embed_tokens.weight` layout). Because the SDNQ Qwen3 encoder loader always builds a text-only
`Qwen3ForCausalLM`, identification must reject anything it cannot load:

- a Qwen2 causal LM (no Qwen3 QK-norm params -> missing weights), and
- a Qwen-VL model (bundles a visual tower -> unexpected weights),

while still accepting a genuine Qwen3 checkpoint carrying its q_norm/k_norm parameters.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_SDNQ_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "file_size": 1000,
    "name": "sdnq-qwen3-single",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _write_sdnq_single_file(path: Path, *, qk_norm: bool, visual_tower: bool = False) -> Path:
    """Write a single-file SDNQ Qwen checkpoint (uint8 weight + fp32 scale pairs).

    qk_norm=True adds the Qwen3-only q_norm/k_norm weights; visual_tower=True adds Qwen-VL visual.*
    keys. Both switches let a single builder produce the Qwen3 / Qwen2 / Qwen-VL cases.
    """
    sd: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.zeros(1000, 2560, dtype=torch.uint8),
        "model.embed_tokens.scale": torch.zeros(1000, 1, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(64, 32, dtype=torch.uint8),
        "model.layers.0.self_attn.q_proj.scale": torch.zeros(64, 1, dtype=torch.float32),
    }
    if qk_norm:
        # Qwen3 QK-normalization — the discriminator vs. Qwen2. Norms are not SDNQ-quantized.
        sd["model.layers.0.self_attn.q_norm.weight"] = torch.ones(128, dtype=torch.float32)
        sd["model.layers.0.self_attn.k_norm.weight"] = torch.ones(128, dtype=torch.float32)
    if visual_tower:
        sd["visual.blocks.0.attn.qkv.weight"] = torch.zeros(64, 32, dtype=torch.uint8)
        sd["visual.blocks.0.attn.qkv.scale"] = torch.zeros(64, 1, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))
    return path


def test_real_qwen3_sdnq_single_file_accepted(tmp_path: Path):
    path = _write_sdnq_single_file(tmp_path / "qwen3.safetensors", qk_norm=True)
    mod = ModelOnDisk(path)
    config = Qwen3Encoder_SDNQ_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": path.as_posix()})
    assert config.format.value == "sdnq_quantized"


def test_qwen2_sdnq_single_file_rejected(tmp_path: Path):
    """A Qwen2 checkpoint lacks q_norm/k_norm; the loader would fail on the missing Qwen3 params."""
    path = _write_sdnq_single_file(tmp_path / "qwen2.safetensors", qk_norm=False)
    mod = ModelOnDisk(path)
    with pytest.raises(NotAMatchError):
        Qwen3Encoder_SDNQ_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": path.as_posix()})

    result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)
    assert not isinstance(result.config, Qwen3Encoder_SDNQ_Config)


def test_qwen_vl_sdnq_single_file_rejected(tmp_path: Path):
    """A Qwen-VL checkpoint bundles a visual tower; the text-only loader cannot consume it."""
    path = _write_sdnq_single_file(tmp_path / "qwen_vl.safetensors", qk_norm=True, visual_tower=True)
    mod = ModelOnDisk(path)
    with pytest.raises(NotAMatchError):
        Qwen3Encoder_SDNQ_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS, "path": path.as_posix()})

    result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)
    assert not isinstance(result.config, Qwen3Encoder_SDNQ_Config)
