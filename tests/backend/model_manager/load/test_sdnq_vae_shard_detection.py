"""Tests that SDNQ VAE detection handles sharded / arbitrarily named safetensors folders.

The shared sdnq_sd_loader supports sharded directories, so a VAE whose SDNQ weight and its scale are
split across standard shard files must still be detected as SDNQ (and routed to _load_sdnq_vae),
not fall through to the generic diffusers loader which cannot reconstruct SDNQTensor values.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.load.model_loaders.vae import _is_sdnq_vae_folder


def test_detects_sdnq_across_two_shards_without_marker(tmp_path: Path):
    # weight in shard 1, its scale in shard 2 — a per-shard scan would miss the pair.
    save_file(
        {"decoder.conv_in.weight": torch.zeros(64, 32, dtype=torch.uint8)},
        str(tmp_path / "diffusion_pytorch_model-00001-of-00002.safetensors"),
    )
    save_file(
        {"decoder.conv_in.scale": torch.zeros(64, 1, dtype=torch.float32)},
        str(tmp_path / "diffusion_pytorch_model-00002-of-00002.safetensors"),
    )
    assert _is_sdnq_vae_folder(tmp_path)


def test_detects_sdnq_via_quantization_config_marker(tmp_path: Path):
    (tmp_path / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}), encoding="utf-8")
    save_file(
        {"decoder.conv_in.weight": torch.zeros(64, 32, dtype=torch.uint8)},
        str(tmp_path / "diffusion_pytorch_model-00001-of-00003.safetensors"),
    )
    assert _is_sdnq_vae_folder(tmp_path)


def test_plain_sharded_vae_is_not_detected_as_sdnq(tmp_path: Path):
    # No scale siblings and no marker -> not SDNQ.
    save_file(
        {"decoder.conv_in.weight": torch.zeros(64, 32, dtype=torch.float32)},
        str(tmp_path / "diffusion_pytorch_model-00001-of-00002.safetensors"),
    )
    save_file(
        {"decoder.conv_out.weight": torch.zeros(3, 32, dtype=torch.float32)},
        str(tmp_path / "diffusion_pytorch_model-00002-of-00002.safetensors"),
    )
    assert not _is_sdnq_vae_folder(tmp_path)
