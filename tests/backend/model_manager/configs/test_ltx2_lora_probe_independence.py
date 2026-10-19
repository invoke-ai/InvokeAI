"""LTX-2 LoRA probe tests, including mutual exclusivity with Wan / MiniMax H3 / Anima / Krea-2.

``Config_Base.CONFIG_CLASSES`` is a set, so probe order is non-deterministic across process
restarts — every architecture pair must be mutually exclusive at the per-config level, or which
config claims a file depends on dictionary iteration order.

LTX-2 is the awkward one: its published layout uses ``transformer_blocks.N.attn1.to_q``, which it
shares with Wan's diffusers layout and with QwenImage. What makes it identifiable is the second,
audio-side tower and the cross-modal attentions that only a dual-stream video+audio transformer
has. These tests feed one fixed state dict to every probe individually and assert exactly one
accepts.
"""

import atexit
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.lora import (
    LoRA_LyCORIS_Anima_Config,
    LoRA_LyCORIS_Flux2_Config,
    LoRA_LyCORIS_FLUX_Config,
    LoRA_LyCORIS_Krea2_Config,
    LoRA_LyCORIS_LTX2_Config,
    LoRA_LyCORIS_MiniMaxH3_Config,
    LoRA_LyCORIS_QwenImage_Config,
    LoRA_LyCORIS_Wan_Config,
    LoRA_LyCORIS_ZImage_Config,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType


def _z(*shape: int) -> torch.Tensor:
    """A zero tensor with the real layout but a single element of storage.

    The probes read key names and shapes, never an element. Materializing LTX-2's rank-450 widths
    would cost GBs across the file, which parallel CI processes cannot hold at once.
    """
    return torch.zeros((), dtype=torch.float32).expand(shape)


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
    path = Path(_TMP_DIR.name) / f"{name}.safetensors"
    path.touch()
    try:
        return True, cls.from_model_on_disk(_make_mod(path, sd), _overrides(path, name))
    except NotAMatchError as e:
        return False, e


def _ltx2_distilled_keys(prefix: str = "diffusion_model.") -> dict[str, torch.Tensor]:
    """Key shape and widths read from ltx-2.5-22b-distilled-lora-450 (rank 450, PEFT
    ``lora_A``/``lora_B``, no ``.alpha`` tensors). Two blocks stand in for the released 48."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        base = f"{prefix}transformer_blocks.{block}"
        # Video tower — shared in name with Wan's diffusers layout and QwenImage.
        for proj in ("to_q", "to_k", "to_v"):
            sd[f"{base}.attn1.{proj}.lora_A.weight"] = _z(450, 4096)
            sd[f"{base}.attn1.{proj}.lora_B.weight"] = _z(4096, 450)
        # Audio tower and cross-modal attentions — the part that is LTX-2 alone.
        for proj in ("to_q", "to_k", "to_v"):
            sd[f"{base}.audio_attn1.{proj}.lora_A.weight"] = _z(450, 2048)
            sd[f"{base}.audio_attn1.{proj}.lora_B.weight"] = _z(2048, 450)
            sd[f"{base}.audio_to_video_attn.{proj}.lora_A.weight"] = _z(450, 4096)
            sd[f"{base}.audio_to_video_attn.{proj}.lora_B.weight"] = _z(4096, 450)
            sd[f"{base}.video_to_audio_attn.{proj}.lora_A.weight"] = _z(450, 2048)
            sd[f"{base}.video_to_audio_attn.{proj}.lora_B.weight"] = _z(2048, 450)
        sd[f"{base}.audio_ff.net.2.lora_A.weight"] = _z(450, 8192)
        sd[f"{base}.audio_ff.net.2.lora_B.weight"] = _z(2048, 450)
    # Top-level cross-modal modulation heads.
    sd[f"{prefix}av_ca_a2v_gate_adaln_single.linear.lora_A.weight"] = _z(450, 4096)
    sd[f"{prefix}av_ca_a2v_gate_adaln_single.linear.lora_B.weight"] = _z(4096, 450)
    sd[f"{prefix}audio_adaln_single.linear.lora_A.weight"] = _z(450, 2048)
    sd[f"{prefix}audio_adaln_single.linear.lora_B.weight"] = _z(18432, 450)
    sd[f"{prefix}patchify_proj.lora_A.weight"] = _z(128, 128)
    sd[f"{prefix}patchify_proj.lora_B.weight"] = _z(4096, 128)
    return sd


def _wan_native_keys() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for sub in ("self_attn", "cross_attn"):
            for proj in ("q", "k", "v", "o"):
                base = f"diffusion_model.blocks.{block}.{sub}.{proj}"
                sd[f"{base}.lora_down.weight"] = _z(64, 5120)
                sd[f"{base}.lora_up.weight"] = _z(5120, 64)
    return sd


def _wan_diffusers_keys() -> dict[str, torch.Tensor]:
    """The layout that actually collides with LTX-2 by name: ``transformer_blocks.N.attn1``."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for attn in ("attn1", "attn2"):
            for proj in ("to_q", "to_k", "to_v"):
                base = f"diffusion_model.transformer_blocks.{block}.{attn}.{proj}"
                sd[f"{base}.lora_A.weight"] = _z(64, 5120)
                sd[f"{base}.lora_B.weight"] = _z(5120, 64)
    return sd


def _h3_turbo_keys() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        base = f"blocks.{block}"
        sd[f"{base}.attn.qkv_proj.lora_A.weight"] = _z(64, 5376)
        sd[f"{base}.attn.qkv_proj.lora_B.weight"] = _z(3 * 5376, 64)
        sd[f"{base}.adaln_proj.linear.lora_A.weight"] = _z(16, 2688)
        sd[f"{base}.adaln_proj.linear.lora_B.weight"] = _z(96768, 16)
    sd["final_layer.adaln_proj.linear.lora_A.weight"] = _z(16, 2688)
    sd["final_layer.adaln_proj.linear.lora_B.weight"] = _z(10752, 16)
    return sd


def _anima_kohya_keys() -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        sd[f"lora_unet_blocks_{block}_cross_attn_q_proj.lora_down.weight"] = _z(16, 2048)
        sd[f"lora_unet_blocks_{block}_cross_attn_q_proj.lora_up.weight"] = _z(2048, 16)
        sd[f"lora_unet_blocks_{block}_mlp_layer1.lora_down.weight"] = _z(16, 2048)
        sd[f"lora_unet_blocks_{block}_mlp_layer1.lora_up.weight"] = _z(8192, 16)
    return sd


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "transformer.", "base_model.model.transformer."])
def test_an_ltx2_lora_identifies_as_ltx2(prefix: str) -> None:
    matched, result = _probe(LoRA_LyCORIS_LTX2_Config, _ltx2_distilled_keys(prefix))

    assert matched, result
    assert result.base is BaseModelType.LTX2


@pytest.mark.parametrize(
    "other",
    [
        LoRA_LyCORIS_Wan_Config,
        LoRA_LyCORIS_MiniMaxH3_Config,
        LoRA_LyCORIS_Anima_Config,
        LoRA_LyCORIS_Krea2_Config,
        # These four also precede the LTX-2 entry in the union, so they are the ones whose order
        # actually decides which config claims a file. Omitting them left the exclusivity this
        # file's docstring promises resting on the four that happen to come after.
        LoRA_LyCORIS_FLUX_Config,
        LoRA_LyCORIS_Flux2_Config,
        LoRA_LyCORIS_QwenImage_Config,
        LoRA_LyCORIS_ZImage_Config,
    ],
    ids=lambda cls: cls.__name__,
)
def test_an_ltx2_lora_is_rejected_by_every_other_probe(other) -> None:
    matched, _ = _probe(other, _ltx2_distilled_keys())

    assert not matched


@pytest.mark.parametrize(
    ("name", "keys"),
    [
        ("wan-native", _wan_native_keys),
        ("wan-diffusers", _wan_diffusers_keys),
        ("minimax-h3", _h3_turbo_keys),
        ("anima", _anima_kohya_keys),
    ],
)
def test_another_familys_lora_is_rejected_by_the_ltx2_probe(name: str, keys) -> None:
    """``wan-diffusers`` is the one that matters: it shares ``transformer_blocks.N.attn1`` with
    LTX-2, so only the absent audio tower separates them."""
    matched, _ = _probe(LoRA_LyCORIS_LTX2_Config, keys(), name=name)

    assert not matched


def test_the_ltx2_probe_requires_a_lora_suffix() -> None:
    """LTX-2 module names alone are the base checkpoint, not a LoRA."""
    sd = {
        k.replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".bias"): v
        for k, v in _ltx2_distilled_keys().items()
    }

    matched, _ = _probe(LoRA_LyCORIS_LTX2_Config, sd)

    assert not matched


@pytest.mark.parametrize(
    "magnitude_key",
    [
        "dora_scale",
        # The PEFT / ai-toolkit spelling. LTX-2 LoRAs ship in the PEFT layout, so this is the one
        # the family will actually meet -- and DoRA is the only variant that reaches this guard,
        # since LoKR and LoHA files carry no lora_A/lora_B and fail the generic suffix test first.
        "lora_magnitude_vector.weight",
        "magnitude",
    ],
)
def test_the_ltx2_probe_rejects_dora_in_either_spelling(magnitude_key: str) -> None:
    """A different patch shape than the conversion emits. Admitting one defers the failure to the
    denoise -- after the 22B transformer has loaded -- which is what this guard exists to prevent."""
    sd = _ltx2_distilled_keys()
    sd[f"diffusion_model.transformer_blocks.0.attn1.to_q.{magnitude_key}"] = _z(4096)

    matched, error = _probe(LoRA_LyCORIS_LTX2_Config, sd)

    assert not matched
    assert "LoKR/LoHA/DoRA" in str(error)


def test_the_ltx2_probe_takes_the_comfy_prefix_the_converter_handles() -> None:
    """The converter strips `model.diffusion_model.` (diffusers' own rename table does), so refusing
    it at install would reject a file that converts perfectly."""
    matched, result = _probe(LoRA_LyCORIS_LTX2_Config, _ltx2_distilled_keys("model.diffusion_model."))

    assert matched, result
    assert result.base is BaseModelType.LTX2


def test_the_ltx2_probe_rejects_nested_prefixes() -> None:
    """The converter strips exactly one prefix. Admitting a doubled one would map every layer onto
    a nonexistent module path — a LoRA that installs, loads, and applies nothing."""
    matched, _ = _probe(LoRA_LyCORIS_LTX2_Config, _ltx2_distilled_keys("diffusion_model.transformer."))

    assert not matched


def test_the_ltx2_probe_rejects_already_converted_keys() -> None:
    """A state dict already in diffusers naming has been through the conversion once; running it
    through again would rename nothing and resolve nothing."""
    sd = _ltx2_distilled_keys()
    sd["diffusion_model.time_embed.linear.lora_A.weight"] = _z(450, 4096)
    sd["diffusion_model.time_embed.linear.lora_B.weight"] = _z(36864, 450)

    matched, _ = _probe(LoRA_LyCORIS_LTX2_Config, sd)

    assert not matched
