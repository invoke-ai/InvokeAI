"""Tests for the single-file H3 checkpoint converter and the AdaLN-pruned transformer.

The two must agree exactly: `convert_minimax_h3_checkpoint_to_diffusers` output keys ==
`MiniMaxH3PrunedTransformer3DModel.state_dict()` keys, or the loader's strict load fails.
The tiny synthetic checkpoint below is hand-written in the remote-code layout so a converter
regression cannot hide behind a shared rename helper.
"""

import torch

from invokeai.backend.minimax_h3.denoise import denoise
from invokeai.backend.minimax_h3.sampling import build_denoise_state
from invokeai.backend.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from invokeai.backend.minimax_h3.transformer_minimax_h3_pruned import (
    MiniMaxH3AdaLayerNormModulationCurve,
    MiniMaxH3PrunedTransformer3DModel,
)
from invokeai.backend.model_manager.load.model_loaders.minimax_h3_state_dict_utils import (
    convert_minimax_h3_checkpoint_to_diffusers,
)

# hidden 8, 2 heads x head_dim 4 (inner 8), ffn 16, 1 block + 1 refiner block, curve [5, 3]
TINY_PRUNED_CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 4,
    "hidden_size": 8,
    "num_layers": 1,
    "num_refiner_layers": 1,
    "ffn_dim": 16,
    "in_channels": 1,
    "audio_in_channels": 4,
    "patch_size": (1, 2, 2),
    "text_dim": 6,
    "rope_freq_dim": 2,
    "adaln_curve_grid": 5,
    "adaln_curve_dim": 3,
}


def _tiny_remote_code_state_dict() -> dict[str, torch.Tensor]:
    """The TINY_PRUNED_CONFIG checkpoint in MiniMax's remote-code key layout."""
    h, inner, ffn, curve = 8, 8, 16, 3
    sd: dict[str, torch.Tensor] = {
        "video_patch_proj.weight": torch.randn(h, 4),
        "video_patch_proj.bias": torch.randn(h),
        "audio_patch_proj.weight": torch.randn(h, 4),
        "audio_patch_proj.bias": torch.randn(h),
        "condition_proj.weight": torch.randn(h, 6),
        "condition_proj.bias": torch.randn(h),
        "adaln_t_table": torch.randn(5, curve),
        "rope.inv_freq": torch.randn(2),
        "token_refiner.final_norm.weight": torch.randn(h),
        "final_layer.norm.weight": torch.randn(h),
        "final_layer.adaln_proj.linear.weight": torch.randn(2 * h, curve),
        "final_layer.adaln_proj.linear.bias": torch.randn(2 * h),
        "final_layer.video_out.weight": torch.randn(4, h),
        "final_layer.video_out.bias": torch.randn(4),
        "final_layer.audio_out.weight": torch.randn(4, h),
        "final_layer.audio_out.bias": torch.randn(4),
    }
    for prefix in ("token_refiner.blocks.0.", "blocks.0."):
        sd[prefix + "norm1.weight"] = torch.randn(h)
        sd[prefix + "norm2.weight"] = torch.randn(h)
        sd[prefix + "attn.qkv_proj.weight"] = torch.randn(3 * inner, h)
        sd[prefix + "attn.q_norm.weight"] = torch.randn(4)
        sd[prefix + "attn.k_norm.weight"] = torch.randn(4)
        sd[prefix + "attn.out_proj.weight"] = torch.randn(h, inner)
        sd[prefix + "mlp.fc1.weight"] = torch.randn(2 * ffn, h)
        sd[prefix + "mlp.fc2.weight"] = torch.randn(h, ffn)
    sd["blocks.0.adaln_proj.linear.weight"] = torch.randn(6 * h * 3, curve)
    sd["blocks.0.adaln_proj.linear.bias"] = torch.randn(6 * h * 3)
    return sd


def test_converted_keys_match_pruned_model_exactly() -> None:
    torch.manual_seed(0)
    converted, markers = convert_minimax_h3_checkpoint_to_diffusers(_tiny_remote_code_state_dict())
    assert markers == {}

    model = MiniMaxH3PrunedTransformer3DModel(**TINY_PRUNED_CONFIG)
    model_keys = set(model.state_dict().keys())
    converted_keys = set(converted.keys())
    assert converted_keys == model_keys, (
        f"missing from checkpoint: {sorted(model_keys - converted_keys)}; "
        f"unexpected in checkpoint: {sorted(converted_keys - model_keys)}"
    )

    # And the strict load itself must succeed - on plain nn.Linear modules throughout. A
    # markerless (bf16 pruned) checkpoint takes the clean path with no Int8 swap at all.
    model.load_state_dict(converted, strict=True, assign=True)
    assert type(model.transformer_blocks[0].attn.to_q) is torch.nn.Linear
    assert type(model.transformer_blocks[0].ff.net[2]) is torch.nn.Linear


def test_converter_splits_fused_qkv_rows_in_order() -> None:
    sd = _tiny_remote_code_state_dict()
    fused = sd["blocks.0.attn.qkv_proj.weight"]
    converted, _ = convert_minimax_h3_checkpoint_to_diffusers(sd)
    assert torch.equal(converted["transformer_blocks.0.attn.to_q.weight"], fused[0:8])
    assert torch.equal(converted["transformer_blocks.0.attn.to_k.weight"], fused[8:16])
    assert torch.equal(converted["transformer_blocks.0.attn.to_v.weight"], fused[16:24])


def test_converter_quantized_layer_markers_and_scale_split() -> None:
    """A quantized fused qkv fans its marker out to to_q/to_k/to_v, and the per-output-channel
    scale rows split identically to the weight rows."""
    sd = _tiny_remote_code_state_dict()
    marker_json = b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'
    sd["blocks.0.attn.qkv_proj.weight"] = torch.randint(-128, 127, (24, 8), dtype=torch.int8)
    sd["blocks.0.attn.qkv_proj.weight_scale"] = torch.rand(24, 1)
    sd["blocks.0.attn.qkv_proj.comfy_quant"] = torch.frombuffer(marker_json, dtype=torch.uint8).clone()
    sd["blocks.0.mlp.fc2.weight"] = torch.randint(-128, 127, (8, 16), dtype=torch.int8)
    sd["blocks.0.mlp.fc2.weight_scale"] = torch.rand(8, 1)
    sd["blocks.0.mlp.fc2.comfy_quant"] = torch.frombuffer(marker_json, dtype=torch.uint8).clone()

    converted, markers = convert_minimax_h3_checkpoint_to_diffusers(sd)

    expected_marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    assert markers == {
        "transformer_blocks.0.attn.to_q": expected_marker,
        "transformer_blocks.0.attn.to_k": expected_marker,
        "transformer_blocks.0.attn.to_v": expected_marker,
        "transformer_blocks.0.ff.net.2": expected_marker,
    }
    assert not any(k.endswith(".comfy_quant") for k in converted)
    scale = sd["blocks.0.attn.qkv_proj.weight_scale"]
    assert torch.equal(converted["transformer_blocks.0.attn.to_k.weight_scale"], scale[8:16])
    assert converted["transformer_blocks.0.ff.net.2.weight"].dtype == torch.int8


def test_converter_drops_rope_inv_freq() -> None:
    converted, _ = convert_minimax_h3_checkpoint_to_diffusers(_tiny_remote_code_state_dict())
    assert "rope.inv_freq" not in converted


def test_curve_temb_interpolates_table() -> None:
    torch.manual_seed(1)
    model = MiniMaxH3PrunedTransformer3DModel(**TINY_PRUNED_CONFIG)
    table = torch.randn(5, 3)
    model.adaln_t_table.copy_(table)

    t = torch.tensor([0.0, 1.0, 0.375, 2.0])  # 0.375 * 4 = 1.5 -> halfway rows 1..2; 2.0 clamps
    temb = model._curve_temb(t)
    assert torch.allclose(temb[0], table[0], atol=1e-6)
    assert torch.allclose(temb[1], table[4], atol=1e-6)
    assert torch.allclose(temb[2], (table[1] + table[2]) / 2, atol=1e-6)
    assert torch.allclose(temb[3], table[4], atol=1e-6)


def test_curve_adaln_applies_no_silu() -> None:
    torch.manual_seed(2)
    mod = MiniMaxH3AdaLayerNormModulationCurve(time_embed_dim=3, hidden_size=8, output_dtype=torch.float32)
    temb = torch.randn(2, 3)
    chunks = mod(temb)
    manual = mod.linear(temb).view(-1, 6 * 8).chunk(6, dim=-1)
    for got, ref in zip(chunks, manual, strict=True):
        assert torch.allclose(got, ref, atol=1e-6)


def test_pruned_model_runs_the_denoise_loop() -> None:
    """End-to-end: the pruned model is a drop-in for the full model in the denoise loop."""
    torch.manual_seed(3)
    config = {**TINY_PRUNED_CONFIG, "in_channels": 24, "audio_in_channels": 32, "hidden_size": 32}
    config["num_attention_heads"] = 2
    config["attention_head_dim"] = 16
    config["ffn_dim"] = 64
    config["text_dim"] = 8
    model = MiniMaxH3PrunedTransformer3DModel(**config)
    model.eval()

    state = build_denoise_state(
        text_token_tags=torch.tensor([1, 1, 0], dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=8,
        num_inference_steps=3,
        seed=42,
        device=torch.device("cpu"),
    )
    prompt_embeds = torch.randn(1, 3, 8)
    video_rows, audio_rows = denoise(model, state, prompt_embeds)
    assert video_rows.shape == (2 * 2 * 2, 96)
    assert audio_rows.shape == (16, 32)


def test_converted_keys_match_full_model_exactly() -> None:
    """The FULL (non-pruned) single file carries the timestep MLP under time_embedder.proj_in /
    proj_out; those must land on diffusers' TimestepEmbedding attribute names (linear_1 /
    linear_2) or the strict load fails."""
    torch.manual_seed(4)
    sd = _tiny_remote_code_state_dict()
    del sd["adaln_t_table"]
    # freq_dim 4 -> hidden 8 -> time_embed_dim 3 (adaln input dims stay 3, shared with the
    # pruned fixture's curve_dim)
    sd["time_embedder.proj_in.weight"] = torch.randn(8, 4)
    sd["time_embedder.proj_in.bias"] = torch.randn(8)
    sd["time_embedder.proj_out.weight"] = torch.randn(3, 8)
    sd["time_embedder.proj_out.bias"] = torch.randn(3)

    converted, markers = convert_minimax_h3_checkpoint_to_diffusers(sd)
    assert markers == {}
    assert "time_embedder.linear_1.weight" in converted
    assert "time_embedder.linear_2.bias" in converted

    full_config = {k: v for k, v in TINY_PRUNED_CONFIG.items() if k not in ("adaln_curve_grid", "adaln_curve_dim")}
    model = MiniMaxH3Transformer3DModel(**full_config, freq_dim=4, time_embed_hidden_dim=8, time_embed_dim=3)
    model_keys = set(model.state_dict().keys())
    converted_keys = set(converted.keys())
    assert converted_keys == model_keys, (
        f"missing from checkpoint: {sorted(model_keys - converted_keys)}; "
        f"unexpected in checkpoint: {sorted(converted_keys - model_keys)}"
    )
    model.load_state_dict(converted, strict=True, assign=True)


def test_read_comfy_quant_markers_from_header_only(tmp_path) -> None:
    from safetensors.torch import save_file

    from invokeai.backend.model_manager.load.model_loaders.minimax_h3_state_dict_utils import (
        read_comfy_quant_markers,
    )

    marker_json = b'{"format": "fp8_scaled", "convrot": false}'
    path = tmp_path / "tiny.safetensors"
    save_file(
        {
            "blocks.0.mlp.fc2.weight": torch.zeros(2, 2),
            "blocks.0.mlp.fc2.comfy_quant": torch.frombuffer(marker_json, dtype=torch.uint8).clone(),
            "unrelated.weight": torch.zeros(1),
        },
        str(path),
    )

    markers = read_comfy_quant_markers(path)
    assert markers == {"blocks.0.mlp.fc2": {"format": "fp8_scaled", "convrot": False}}


def test_converter_swaps_fused_swiglu_halves() -> None:
    """The remote-code fused fc1 stores [gate; value]; diffusers SwiGLU expects [value; gate]
    (silu on the SECOND half). Verified bit-exactly against the diffusers folder release:
    file[:H] == folder[H:] and file[H:] == folder[:H]. The converter must swap the halves of
    both the weight and (for quantized layers) the per-output-row scales. Regression test for
    the bug where every block's MLP computed silu(value)*gate and the model emitted noise."""
    from invokeai.backend.model_manager.load.model_loaders.minimax_h3_state_dict_utils import (
        convert_minimax_h3_checkpoint_to_diffusers,
    )

    ffn = 4
    gate = torch.full((ffn, 6), 1.0)
    value = torch.full((ffn, 6), 2.0)
    gate_scale = torch.full((ffn, 1), 3.0)
    value_scale = torch.full((ffn, 1), 4.0)
    sd = {
        "blocks.0.mlp.fc1.weight": torch.cat([gate, value], dim=0),
        "blocks.0.mlp.fc1.weight_scale": torch.cat([gate_scale, value_scale], dim=0),
        "token_refiner.blocks.0.mlp.fc1.weight": torch.cat([gate, value], dim=0),
    }
    converted, _ = convert_minimax_h3_checkpoint_to_diffusers(sd)

    w = converted["transformer_blocks.0.ff.net.0.proj.weight"]
    assert torch.equal(w[:ffn], value) and torch.equal(w[ffn:], gate)
    s = converted["transformer_blocks.0.ff.net.0.proj.weight_scale"]
    assert torch.equal(s[:ffn], value_scale) and torch.equal(s[ffn:], gate_scale)
    rw = converted["token_refiner.refiner_blocks.0.ff.net.0.proj.weight"]
    assert torch.equal(rw[:ffn], value) and torch.equal(rw[ffn:], gate)
