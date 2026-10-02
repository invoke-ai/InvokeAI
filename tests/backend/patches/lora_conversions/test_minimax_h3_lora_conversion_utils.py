"""Tests for MiniMax H3 LoRA state-dict conversion.

The critical properties are the fused-tensor transforms: the fused-QKV fan-out and the
fused-SwiGLU half swap of ``lora_up``. Content-level checks with distinguishable segments
are mandatory here — a shape/key-only test cannot catch a wrong half order (the exact bug
class the base-model converter shipped with and had to fix).
"""

import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.minimax_h3_state_dict_utils import (
    convert_minimax_h3_checkpoint_to_diffusers,
)
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.minimax_h3_lora_constants import (
    MINIMAX_H3_LORA_TRANSFORMER_PREFIX,
)
from invokeai.backend.patches.lora_conversions.minimax_h3_lora_conversion_utils import (
    is_minimax_h3_adaln_layer_path,
    is_state_dict_likely_in_minimax_h3_format,
    lora_model_from_minimax_h3_state_dict,
)

# Tiny stand-in dims (real model: hidden 5376, ffn 14336, time_embed 2688, rank 64/16).
HIDDEN = 8
FFN = 12
TIME_EMBED = 6
RANK = 4


def _lora_pair(out_features: int, in_features: int, rank: int = RANK) -> tuple[torch.Tensor, torch.Tensor]:
    down = torch.randn(rank, in_features)
    up = torch.randn(out_features, rank)
    return down, up


def _make_turbo_style_state_dict(prefix: str = "") -> dict[str, torch.Tensor]:
    """One transformer block + one refiner block + final layer, in the published layout."""
    sd: dict[str, torch.Tensor] = {}
    for stem in ("blocks.0", "token_refiner.blocks.0"):
        down, up = _lora_pair(3 * HIDDEN, HIDDEN)
        sd[f"{prefix}{stem}.attn.qkv_proj.lora_A.weight"] = down
        sd[f"{prefix}{stem}.attn.qkv_proj.lora_B.weight"] = up
        down, up = _lora_pair(HIDDEN, HIDDEN)
        sd[f"{prefix}{stem}.attn.out_proj.lora_A.weight"] = down
        sd[f"{prefix}{stem}.attn.out_proj.lora_B.weight"] = up
        down, up = _lora_pair(2 * FFN, HIDDEN)
        sd[f"{prefix}{stem}.mlp.fc1.lora_A.weight"] = down
        sd[f"{prefix}{stem}.mlp.fc1.lora_B.weight"] = up
        down, up = _lora_pair(HIDDEN, FFN)
        sd[f"{prefix}{stem}.mlp.fc2.lora_A.weight"] = down
        sd[f"{prefix}{stem}.mlp.fc2.lora_B.weight"] = up
    down, up = _lora_pair(6 * HIDDEN * 3, TIME_EMBED, rank=2)
    sd[f"{prefix}blocks.0.adaln_proj.linear.lora_A.weight"] = down
    sd[f"{prefix}blocks.0.adaln_proj.linear.lora_B.weight"] = up
    down, up = _lora_pair(2 * HIDDEN, TIME_EMBED, rank=2)
    sd[f"{prefix}final_layer.adaln_proj.linear.lora_A.weight"] = down
    sd[f"{prefix}final_layer.adaln_proj.linear.lora_B.weight"] = up
    return sd


EXPECTED_LAYER_PATHS = {
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.attn.to_k",
    "transformer_blocks.0.attn.to_v",
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.0.ff.net.0.proj",
    "transformer_blocks.0.ff.net.2",
    "transformer_blocks.0.adaln_proj.linear",
    "token_refiner.refiner_blocks.0.attn.to_q",
    "token_refiner.refiner_blocks.0.attn.to_k",
    "token_refiner.refiner_blocks.0.attn.to_v",
    "token_refiner.refiner_blocks.0.attn.to_out.0",
    "token_refiner.refiner_blocks.0.ff.net.0.proj",
    "token_refiner.refiner_blocks.0.ff.net.2",
    "norm_out.linear",
}


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "transformer.", "base_model.model.transformer."])
def test_layer_path_mapping(prefix: str):
    sd = _make_turbo_style_state_dict(prefix)
    patch = lora_model_from_minimax_h3_state_dict(sd)

    got_paths = set()
    for key, layer in patch.layers.items():
        assert key.startswith(MINIMAX_H3_LORA_TRANSFORMER_PREFIX)
        assert isinstance(layer, LoRALayer)
        got_paths.add(key[len(MINIMAX_H3_LORA_TRANSFORMER_PREFIX) :])
    assert got_paths == EXPECTED_LAYER_PATHS


def test_alpha_defaults_to_rank_scale_one():
    patch = lora_model_from_minimax_h3_state_dict(_make_turbo_style_state_dict())
    for layer in patch.layers.values():
        assert isinstance(layer, LoRALayer)
        assert layer.scale() == 1.0


def test_qkv_fan_out_rows_and_shared_down():
    sd = _make_turbo_style_state_dict()
    key = "blocks.0.attn.qkv_proj"
    down = sd[f"{key}.lora_A.weight"]
    up = sd[f"{key}.lora_B.weight"]
    patch = lora_model_from_minimax_h3_state_dict(sd)

    p = MINIMAX_H3_LORA_TRANSFORMER_PREFIX
    for i, proj in enumerate(("to_q", "to_k", "to_v")):
        layer = patch.layers[f"{p}transformer_blocks.0.attn.{proj}"]
        assert isinstance(layer, LoRALayer)
        assert torch.equal(layer.down, down)
        assert torch.equal(layer.up, up[i * HIDDEN : (i + 1) * HIDDEN])


def test_swiglu_half_swap():
    sd = _make_turbo_style_state_dict()
    key = "blocks.0.mlp.fc1"
    up = sd[f"{key}.lora_B.weight"]
    patch = lora_model_from_minimax_h3_state_dict(sd)

    layer = patch.layers[f"{MINIMAX_H3_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.ff.net.0.proj"]
    assert isinstance(layer, LoRALayer)
    assert torch.equal(layer.up, torch.cat([up[FFN:], up[:FFN]], dim=0))
    assert torch.equal(layer.down, sd[f"{key}.lora_A.weight"])


def test_fused_deltas_match_base_model_converter():
    """Ground truth: patching the converted (split/swapped) weights with the converted LoRA
    must equal converting the fused ``W + up @ down`` with the base-model state-dict
    converter. This ties the LoRA transforms to the exact fan-out/half-swap the loader
    applies to the base weights."""
    torch.manual_seed(0)
    sd = _make_turbo_style_state_dict()
    patch = lora_model_from_minimax_h3_state_dict(sd)

    fused_checkpoint = {
        "blocks.0.attn.qkv_proj.weight": torch.randn(3 * HIDDEN, HIDDEN),
        "blocks.0.mlp.fc1.weight": torch.randn(2 * FFN, HIDDEN),
    }

    # Reference: apply the fused low-rank update, then convert with the base converter.
    fused_patched = {}
    for key, weight in fused_checkpoint.items():
        stem = key[: -len(".weight")]
        delta = sd[f"{stem}.lora_B.weight"] @ sd[f"{stem}.lora_A.weight"]
        fused_patched[key] = weight + delta
    reference, _ = convert_minimax_h3_checkpoint_to_diffusers(fused_patched)

    # Candidate: convert the base weights first, then apply the converted LoRA per module.
    base_converted, _ = convert_minimax_h3_checkpoint_to_diffusers(dict(fused_checkpoint))
    for module_path, base_weight in base_converted.items():
        layer = patch.layers[f"{MINIMAX_H3_LORA_TRANSFORMER_PREFIX}{module_path[: -len('.weight')]}"]
        delta = layer.get_parameters({"weight": base_weight}, weight=1.0)["weight"]
        assert torch.allclose(base_weight + delta, reference[module_path], atol=1e-5)


def test_adaln_layer_path_predicate():
    assert is_minimax_h3_adaln_layer_path("transformer_blocks.0.adaln_proj.linear")
    assert is_minimax_h3_adaln_layer_path("norm_out.linear")
    assert not is_minimax_h3_adaln_layer_path("transformer_blocks.0.attn.to_q")
    assert not is_minimax_h3_adaln_layer_path("token_refiner.refiner_blocks.0.ff.net.2")


def test_format_detection():
    assert is_state_dict_likely_in_minimax_h3_format(_make_turbo_style_state_dict())
    assert is_state_dict_likely_in_minimax_h3_format(_make_turbo_style_state_dict("diffusion_model."))

    wan_native = {
        "diffusion_model.blocks.0.self_attn.q.lora_A.weight": torch.zeros(4, 16),
        "diffusion_model.blocks.0.self_attn.q.lora_B.weight": torch.zeros(16, 4),
    }
    assert not is_state_dict_likely_in_minimax_h3_format(wan_native)

    # An H3-looking dict polluted with another architecture's keys is rejected outright.
    polluted = _make_turbo_style_state_dict()
    polluted["blocks.0.self_attn.q.lora_A.weight"] = torch.zeros(4, 16)
    assert not is_state_dict_likely_in_minimax_h3_format(polluted)


def test_rejects_lycoris_variant_tensors():
    # The probe rejects these at install; the converter guard covers other entry paths. Without
    # it, DoRA's per-output-row magnitudes would be silently mis-applied by the SwiGLU half-swap.
    sd = _make_turbo_style_state_dict()
    sd["blocks.0.mlp.fc1.dora_scale"] = torch.zeros(2 * FFN, 1)
    with pytest.raises(ValueError, match="plain low-rank"):
        lora_model_from_minimax_h3_state_dict(sd)


def test_nested_prefix_is_not_detected():
    # Anchored detection: a nested prefix would survive the single prefix-strip and convert to
    # nonexistent module paths, so it must not be detected as H3 in the first place.
    assert not is_state_dict_likely_in_minimax_h3_format(_make_turbo_style_state_dict("diffusion_model.transformer."))
    assert not is_state_dict_likely_in_minimax_h3_format(_make_turbo_style_state_dict("unet.diffusion_model."))


def test_rejects_malformed_fused_tensors():
    sd = {
        "blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(RANK, HIDDEN),
        "blocks.0.attn.qkv_proj.lora_B.weight": torch.zeros(3 * HIDDEN + 1, RANK),
    }
    with pytest.raises(ValueError, match="not divisible by 3"):
        lora_model_from_minimax_h3_state_dict(sd)

    sd = {
        "blocks.0.mlp.fc1.lora_A.weight": torch.zeros(RANK, HIDDEN),
        "blocks.0.mlp.fc1.lora_B.weight": torch.zeros(2 * FFN + 1, RANK),
    }
    with pytest.raises(ValueError, match="odd output row count"):
        lora_model_from_minimax_h3_state_dict(sd)
