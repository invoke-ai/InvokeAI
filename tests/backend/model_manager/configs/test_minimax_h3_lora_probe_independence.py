"""MiniMax H3 LoRA probe tests, including mutual exclusivity with Wan / Anima / Krea-2.

``Config_Base.CONFIG_CLASSES`` is a set, so probe order is non-deterministic across
process restarts — every architecture pair must be mutually exclusive at the
per-config level. H3's native key layout shares the ``blocks.N.`` shape with Wan and
Anima (and carries ``blocks.N.mlp.fc1``, which Wan's Anima anti-pattern matches), so
each test feeds one fixed state dict to every probe individually and asserts exactly
the right one accepts.
"""

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import torch

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import (
    LoRA_LyCORIS_Anima_Config,
    LoRA_LyCORIS_Krea2_Config,
    LoRA_LyCORIS_MiniMaxH3_Config,
    LoRA_LyCORIS_Wan_Config,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType


def _make_mod(path: Path, sd: dict) -> MagicMock:
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    return mod


def _overrides(p: Path, name: str) -> dict:
    return {
        "hash": "test-hash",
        "path": str(p),
        "file_size": 0,
        "name": name,
        "source": str(p),
        "source_type": "path",
    }


_TMP_DIR = TemporaryDirectory()
atexit.register(_TMP_DIR.cleanup)


def _probe(cls, sd: dict, name: str = "test-lora"):
    # The probes require a real file on disk (raise_if_not_file); contents are irrelevant
    # because the state dict is mocked.
    path = Path(_TMP_DIR.name) / f"{name}.safetensors"
    path.touch()
    try:
        return True, cls.from_model_on_disk(_make_mod(path, sd), _overrides(path, name))
    except NotAMatchError as e:
        return False, e


def _h3_turbo_keys(prefix: str = "") -> dict[str, torch.Tensor]:
    """Realistic key shape from larryvrh/MiniMax-H3-Turbo-Lora (rank 64 backbone,
    rank 16 AdaLN, PEFT lora_A/lora_B, no alpha tensors)."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        base = f"{prefix}blocks.{block}"
        sd[f"{base}.attn.qkv_proj.lora_A.weight"] = torch.zeros(64, 5376)
        sd[f"{base}.attn.qkv_proj.lora_B.weight"] = torch.zeros(3 * 5376, 64)
        sd[f"{base}.attn.out_proj.lora_A.weight"] = torch.zeros(64, 7168)
        sd[f"{base}.attn.out_proj.lora_B.weight"] = torch.zeros(5376, 64)
        sd[f"{base}.mlp.fc1.lora_A.weight"] = torch.zeros(64, 5376)
        sd[f"{base}.mlp.fc1.lora_B.weight"] = torch.zeros(2 * 14336, 64)
        sd[f"{base}.mlp.fc2.lora_A.weight"] = torch.zeros(64, 14336)
        sd[f"{base}.mlp.fc2.lora_B.weight"] = torch.zeros(5376, 64)
        sd[f"{base}.adaln_proj.linear.lora_A.weight"] = torch.zeros(16, 2688)
        sd[f"{base}.adaln_proj.linear.lora_B.weight"] = torch.zeros(96768, 16)
    sd[f"{prefix}token_refiner.blocks.0.attn.qkv_proj.lora_A.weight"] = torch.zeros(64, 5376)
    sd[f"{prefix}token_refiner.blocks.0.attn.qkv_proj.lora_B.weight"] = torch.zeros(3 * 5376, 64)
    sd[f"{prefix}final_layer.adaln_proj.linear.lora_A.weight"] = torch.zeros(16, 2688)
    sd[f"{prefix}final_layer.adaln_proj.linear.lora_B.weight"] = torch.zeros(10752, 16)
    return sd


def _wan_native_keys() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for sub in ("self_attn", "cross_attn"):
            for proj in ("q", "k", "v", "o"):
                base = f"diffusion_model.blocks.{block}.{sub}.{proj}"
                sd[f"{base}.lora_down.weight"] = torch.zeros(64, 5120)
                sd[f"{base}.lora_up.weight"] = torch.zeros(5120, 64)
    return sd


def _anima_kohya_keys() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        sd[f"lora_unet_blocks_{block}_cross_attn_q_proj.lora_down.weight"] = torch.zeros(16, 2048)
        sd[f"lora_unet_blocks_{block}_cross_attn_q_proj.lora_up.weight"] = torch.zeros(2048, 16)
        sd[f"lora_unet_blocks_{block}_mlp_layer1.lora_down.weight"] = torch.zeros(16, 2048)
        sd[f"lora_unet_blocks_{block}_mlp_layer1.lora_up.weight"] = torch.zeros(8192, 16)
    return sd


def test_h3_lora_identifies_as_h3():
    for prefix in ("", "diffusion_model."):
        accepted, result = _probe(LoRA_LyCORIS_MiniMaxH3_Config, _h3_turbo_keys(prefix))
        assert accepted, f"H3 probe rejected its own layout (prefix={prefix!r}): {result}"
        assert result.base is BaseModelType.MiniMaxH3


def test_h3_lora_rejected_by_other_probes():
    sd = _h3_turbo_keys()
    for cls in (LoRA_LyCORIS_Wan_Config, LoRA_LyCORIS_Anima_Config, LoRA_LyCORIS_Krea2_Config):
        accepted, result = _probe(cls, sd)
        assert not accepted, f"{cls.__name__} wrongly accepted an H3 LoRA: {result}"


def test_wan_lora_rejected_by_h3_probe():
    sd = _wan_native_keys()
    accepted, result = _probe(LoRA_LyCORIS_MiniMaxH3_Config, sd)
    assert not accepted, f"H3 probe wrongly accepted a Wan LoRA: {result}"
    accepted, result = _probe(LoRA_LyCORIS_Wan_Config, sd)
    assert accepted, f"Wan probe rejected its own layout: {result}"


def test_anima_lora_rejected_by_h3_probe():
    sd = _anima_kohya_keys()
    accepted, result = _probe(LoRA_LyCORIS_MiniMaxH3_Config, sd)
    assert not accepted, f"H3 probe wrongly accepted an Anima LoRA: {result}"
    accepted, result = _probe(LoRA_LyCORIS_Anima_Config, sd)
    assert accepted, f"Anima probe rejected its own layout: {result}"


def test_h3_probe_requires_lora_suffixes():
    # A base-model checkpoint fragment (plain weights, no LoRA suffixes) must not probe as a LoRA.
    sd = {
        "blocks.0.attn.qkv_proj.weight": torch.zeros(3 * 5376, 5376),
        "blocks.0.adaln_proj.linear.weight": torch.zeros(96768, 2688),
    }
    accepted, result = _probe(LoRA_LyCORIS_MiniMaxH3_Config, sd)
    assert not accepted, f"H3 probe wrongly accepted a non-LoRA state dict: {result}"


def test_h3_probe_rejects_already_diffusers_keys():
    # v1 supports the published native layout only; diffusers-style transformer_blocks keys
    # are deliberately rejected rather than half-converted.
    sd = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(64, 5376),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(5376, 64),
    }
    accepted, result = _probe(LoRA_LyCORIS_MiniMaxH3_Config, sd)
    assert not accepted, f"H3 probe wrongly accepted diffusers-naming keys: {result}"
