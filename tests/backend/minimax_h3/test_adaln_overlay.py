"""The hybrid AdaLN overlay: reading one selection of a checkpoint's AdaLN projections, re-expressing
them in the base's curve basis, and swapping them onto a loaded transformer for the duration of a context.

The observable contract is that a transformer running under the overlay computes exactly what a
transformer loaded from the merged checkpoint would - the overlay is a run-time stand-in for the
merged file - and that nothing of the overlay survives the context.
"""

from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.minimax_h3.adaln_overlay import MiniMaxH3AdaLNOverlay, apply_minimax_h3_adaln_overlay
from invokeai.backend.minimax_h3.denoise import denoise
from invokeai.backend.minimax_h3.sampling import build_denoise_state
from invokeai.backend.minimax_h3.transformer_minimax_h3_pruned import MiniMaxH3PrunedTransformer3DModel
from invokeai.backend.model_manager.load.model_loaders.minimax_h3_state_dict_utils import (
    convert_minimax_h3_checkpoint_to_diffusers,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.minimax_h3_lora_constants import MINIMAX_H3_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

# Two DiT blocks so a block range can exclude one; latent channel counts are the real 24/32 so
# the denoise loop's packing accepts the model.
NUM_BLOCKS = 2
HIDDEN, HEADS, HEAD_DIM, FFN, CURVE, TEXT_DIM = 32, 2, 16, 64, 3, 8
VIDEO_IN, AUDIO_IN = 24 * 4, 32
CONFIG = {
    "num_attention_heads": HEADS,
    "attention_head_dim": HEAD_DIM,
    "hidden_size": HIDDEN,
    "num_layers": NUM_BLOCKS,
    "num_refiner_layers": 1,
    "ffn_dim": FFN,
    "in_channels": 24,
    "audio_in_channels": AUDIO_IN,
    "patch_size": (1, 2, 2),
    "text_dim": TEXT_DIM,
    "rope_freq_dim": 2,
    "adaln_curve_grid": 5,
    "adaln_curve_dim": CURVE,
}


def _tiny_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    """A CONFIG-shaped AdaLN-pruned checkpoint in MiniMax's remote-code key layout."""
    g = torch.Generator().manual_seed(seed)
    inner = HEADS * HEAD_DIM

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    sd = {
        "video_patch_proj.weight": rand(HIDDEN, VIDEO_IN),
        "video_patch_proj.bias": rand(HIDDEN),
        "audio_patch_proj.weight": rand(HIDDEN, AUDIO_IN),
        "audio_patch_proj.bias": rand(HIDDEN),
        "condition_proj.weight": rand(HIDDEN, TEXT_DIM),
        "condition_proj.bias": rand(HIDDEN),
        "adaln_t_table": rand(5, CURVE),
        "rope.inv_freq": rand(2),
        "token_refiner.final_norm.weight": rand(HIDDEN),
        "final_layer.norm.weight": rand(HIDDEN),
        "final_layer.adaln_proj.linear.weight": rand(2 * HIDDEN, CURVE),
        "final_layer.adaln_proj.linear.bias": rand(2 * HIDDEN),
        "final_layer.video_out.weight": rand(VIDEO_IN, HIDDEN),
        "final_layer.video_out.bias": rand(VIDEO_IN),
        "final_layer.audio_out.weight": rand(AUDIO_IN, HIDDEN),
        "final_layer.audio_out.bias": rand(AUDIO_IN),
    }
    for prefix in ["token_refiner.blocks.0."] + [f"blocks.{i}." for i in range(NUM_BLOCKS)]:
        sd[prefix + "norm1.weight"] = rand(HIDDEN)
        sd[prefix + "norm2.weight"] = rand(HIDDEN)
        sd[prefix + "attn.qkv_proj.weight"] = rand(3 * inner, HIDDEN)
        sd[prefix + "attn.q_norm.weight"] = rand(HEAD_DIM)
        sd[prefix + "attn.k_norm.weight"] = rand(HEAD_DIM)
        sd[prefix + "attn.out_proj.weight"] = rand(HIDDEN, inner)
        sd[prefix + "mlp.fc1.weight"] = rand(2 * FFN, HIDDEN)
        sd[prefix + "mlp.fc2.weight"] = rand(HIDDEN, FFN)
    for i in range(NUM_BLOCKS):
        sd[f"blocks.{i}.adaln_proj.linear.weight"] = rand(6 * HIDDEN * 3, CURVE)
        sd[f"blocks.{i}.adaln_proj.linear.bias"] = rand(6 * HIDDEN * 3)
    return sd


def _model_from_checkpoint(sd: dict[str, torch.Tensor]) -> MiniMaxH3PrunedTransformer3DModel:
    converted, markers = convert_minimax_h3_checkpoint_to_diffusers(sd)
    assert markers == {}
    model = MiniMaxH3PrunedTransformer3DModel(**CONFIG)
    model.load_state_dict(converted, strict=True, assign=True)
    return model.eval()


def _save(path: Path, sd: dict[str, torch.Tensor]) -> Path:
    save_file({k: v.contiguous() for k, v in sd.items()}, str(path))
    return path


def _run(model: MiniMaxH3PrunedTransformer3DModel) -> tuple[torch.Tensor, torch.Tensor]:
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
    prompt_embeds = torch.randn(1, 3, TEXT_DIM, generator=torch.Generator().manual_seed(7))
    return denoise(model, state, prompt_embeds)


@pytest.fixture
def base_sd() -> dict[str, torch.Tensor]:
    return _tiny_checkpoint(seed=1)


@pytest.fixture
def overlay_sd() -> dict[str, torch.Tensor]:
    return _tiny_checkpoint(seed=2)


def _overlay(
    tmp_path: Path, sd: dict[str, torch.Tensor], start: int, end: int, final: bool = False, name: str = "ref2va"
) -> MiniMaxH3AdaLNOverlay:
    path = _save(tmp_path / f"{name}.safetensors", sd)
    return MiniMaxH3AdaLNOverlay.load_model(MiniMaxH3AdaLNOverlay.selection_path(path, start, end, final))


def test_load_reads_only_the_selected_projections(tmp_path: Path, overlay_sd) -> None:
    # A range past the file's blocks selects what exists (the released files have 50 blocks; a
    # smaller file just contributes fewer), and the final layer is opt-in.
    overlay = _overlay(tmp_path, overlay_sd, 1, 49)

    tensors = overlay.tensors_for(None)
    assert set(tensors) == {
        "transformer_blocks.1.adaln_proj.linear.weight",
        "transformer_blocks.1.adaln_proj.linear.bias",
    }
    assert torch.equal(
        tensors["transformer_blocks.1.adaln_proj.linear.bias"], overlay_sd["blocks.1.adaln_proj.linear.bias"]
    )
    # Block 0, the attention/MLP weights and the output heads never leave the file; the curve
    # table rides along for the change of basis.
    assert sum(t.numel() for t in overlay.buffers()) == (
        overlay_sd["blocks.1.adaln_proj.linear.weight"].numel()
        + overlay_sd["blocks.1.adaln_proj.linear.bias"].numel()
        + overlay_sd["adaln_t_table"].numel()
    )

    with_final = _overlay(tmp_path, overlay_sd, 1, 0, final=True, name="final_only")
    assert set(with_final.tensors_for(None)) == {"norm_out.linear.weight", "norm_out.linear.bias"}
    assert torch.equal(
        with_final.tensors_for(None)["norm_out.linear.weight"], overlay_sd["final_layer.adaln_proj.linear.weight"]
    )


def test_selection_paths_round_trip_and_an_empty_selection_is_refused(tmp_path: Path, overlay_sd) -> None:
    path = _save(tmp_path / "ref2va.safetensors", overlay_sd)
    assert MiniMaxH3AdaLNOverlay.selection_path(path, 25, 49, False).name == "ref2va.safetensors#adaln-25-49"
    assert MiniMaxH3AdaLNOverlay.selection_path(path, 0, 49, True).name == "ref2va.safetensors#adaln-0-49-final"

    with pytest.raises(ValueError, match="selects nothing: blocks 5..49 of the 2-block"):
        MiniMaxH3AdaLNOverlay.load_model(MiniMaxH3AdaLNOverlay.selection_path(path, 5, 49, False))
    with pytest.raises(ValueError, match="selection path"):
        MiniMaxH3AdaLNOverlay.load_model(path)


def test_load_rejects_files_without_h3_adaln_projections(tmp_path: Path) -> None:
    sd = {"blocks.0.attn.qkv_proj.weight": torch.zeros(4, 4)}
    with pytest.raises(ValueError, match="not an H3 transformer"):
        _overlay(tmp_path, sd, 0, 49, name="other")


def test_load_rejects_quantized_adaln_projections(tmp_path: Path, overlay_sd) -> None:
    marker = b'{"format": "int8_tensorwise"}'
    overlay_sd["blocks.0.adaln_proj.linear.comfy_quant"] = torch.frombuffer(marker, dtype=torch.uint8).clone()
    with pytest.raises(ValueError, match="quantizes its AdaLN projections"):
        _overlay(tmp_path, overlay_sd, 1, 1, name="quant")


def test_load_rejects_a_selected_block_missing_its_bias(tmp_path: Path, overlay_sd) -> None:
    del overlay_sd["blocks.1.adaln_proj.linear.bias"]
    with pytest.raises(ValueError, match="missing blocks.1.adaln_proj.linear.bias"):
        _overlay(tmp_path, overlay_sd, 1, 1, name="partial")


def test_overlay_matches_the_merged_checkpoint_and_leaves_no_trace(tmp_path: Path, base_sd, overlay_sd) -> None:
    """Running FL2VA under a block-1 + final-layer overlay of Ref2VA must equal a transformer
    loaded from the merged state dict, and the base must compute its own output again afterwards.
    (Same curve table on both sides: the swap is then a plain tensor swap, bit for bit.)"""
    overlay_sd["adaln_t_table"] = base_sd["adaln_t_table"]
    base = _model_from_checkpoint(base_sd)
    base_video, base_audio = _run(base)

    merged_sd = dict(base_sd)
    for key in ("blocks.1.adaln_proj.linear.weight", "blocks.1.adaln_proj.linear.bias"):
        merged_sd[key] = overlay_sd[key]
    for key in ("final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias"):
        merged_sd[key] = overlay_sd[key]
    merged_video, merged_audio = _run(_model_from_checkpoint(merged_sd))
    assert not torch.allclose(merged_video, base_video), "fixture must make the overlay observable"

    params = _overlay(tmp_path, overlay_sd, 1, 1, final=True).tensors_for(base.adaln_t_table)
    with apply_minimax_h3_adaln_overlay(base, params):
        hybrid_video, hybrid_audio = _run(base)
        # Block 0 keeps the base projection.
        assert torch.equal(
            base.transformer_blocks[0].adaln_proj.linear.weight, base_sd["blocks.0.adaln_proj.linear.weight"]
        )
    assert torch.equal(hybrid_video, merged_video)
    assert torch.equal(hybrid_audio, merged_audio)

    after_video, after_audio = _run(base)
    assert torch.equal(after_video, base_video)
    assert torch.equal(after_audio, base_audio)
    assert torch.equal(base.norm_out.linear.bias, base_sd["final_layer.adaln_proj.linear.bias"])


def test_pruned_overlay_reproduces_the_overlay_models_modulation_on_the_base_curve(
    tmp_path: Path, base_sd, overlay_sd
) -> None:
    """The pruned files' AdaLN weights are coefficients over their OWN curve table. With the
    overlay's table a rotation of the base's, the re-expressed weights driven by the base's
    curve must produce exactly what the overlay model produces with its own curve - which a
    raw tensor swap does not."""
    change = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]) * 1.5
    overlay_sd["adaln_t_table"] = base_sd["adaln_t_table"] @ change
    base = _model_from_checkpoint(base_sd)
    overlay_model = _model_from_checkpoint(overlay_sd)
    params = _overlay(tmp_path, overlay_sd, 1, 1, final=True).tensors_for(base.adaln_t_table)

    t = torch.tensor([0.0, 0.31, 0.5, 0.999])
    with apply_minimax_h3_adaln_overlay(base, params):
        hybrid_block = torch.cat(base.transformer_blocks[1].adaln_proj(base._curve_temb(t)), dim=-1)
        hybrid_final = base.norm_out.linear(base._curve_temb(t))
    overlay_block = torch.cat(overlay_model.transformer_blocks[1].adaln_proj(overlay_model._curve_temb(t)), dim=-1)
    overlay_final = overlay_model.norm_out.linear(overlay_model._curve_temb(t))
    assert torch.allclose(hybrid_block, overlay_block, atol=1e-4)
    assert torch.allclose(hybrid_final, overlay_final, atol=1e-4)

    # Bit-for-bit copies of the overlay's coefficients on the base's curve would NOT match.
    raw = _overlay(tmp_path, overlay_sd, 1, 1, final=True, name="raw").tensors_for(None)
    with apply_minimax_h3_adaln_overlay(base, raw):
        raw_block = torch.cat(base.transformer_blocks[1].adaln_proj(base._curve_temb(t)), dim=-1)
    assert not torch.allclose(raw_block, overlay_block, atol=1e-2)


def test_overlay_is_cast_to_the_parameter_dtype_and_never_written_in_place(tmp_path: Path, base_sd, overlay_sd) -> None:
    """The pruned repacks store AdaLN tensors as f16 while the loaded model runs them float32; the
    swap must cast, and must leave the original tensor object untouched (it can be the model
    cache's shared CPU copy)."""
    base = _model_from_checkpoint(base_sd)
    original = base.transformer_blocks[0].adaln_proj.linear.weight.data
    original_snapshot = original.clone()
    f16_sd = {k: (v.half() if ".adaln_proj.linear." in k else v) for k, v in overlay_sd.items()}
    params = _overlay(tmp_path, f16_sd, 0, 0, name="ref2va_f16").tensors_for(None)

    with apply_minimax_h3_adaln_overlay(base, params):
        weight = base.transformer_blocks[0].adaln_proj.linear.weight
        assert weight.dtype == torch.float32
        assert torch.equal(weight, f16_sd["blocks.0.adaln_proj.linear.weight"].float())
        assert torch.equal(original, original_snapshot)
    assert torch.equal(base.transformer_blocks[0].adaln_proj.linear.weight, original_snapshot)


def test_overlay_restores_cpu_resident_weights_by_re_pointing_at_the_cached_tensor(
    tmp_path: Path, base_sd, overlay_sd
) -> None:
    """A parameter the partial loader leaves on the CPU is the cache's canonical tensor; restoring
    it must re-alias that tensor, not allocate a second CPU copy outside the cache's accounting."""
    base = _model_from_checkpoint(base_sd)
    path = "transformer_blocks.1.adaln_proj.linear.bias"
    cached = {path: torch.full_like(base.get_parameter(path), 0.5)}
    params = _overlay(tmp_path, overlay_sd, 1, 1).tensors_for(None)

    with apply_minimax_h3_adaln_overlay(base, params, cached_weights=cached):
        assert torch.equal(base.get_parameter(path), overlay_sd["blocks.1.adaln_proj.linear.bias"])
    assert base.get_parameter(path).data_ptr() == cached[path].data_ptr()

    # Without a cache copy the original CPU tensor itself is kept and put back, uncopied.
    original = base.get_parameter("norm_out.linear.weight").data
    with apply_minimax_h3_adaln_overlay(
        base, _overlay(tmp_path, overlay_sd, 1, 0, final=True, name="f").tensors_for(None)
    ):
        assert base.get_parameter("norm_out.linear.weight").data_ptr() != original.data_ptr()
    assert base.get_parameter("norm_out.linear.weight").data_ptr() == original.data_ptr()


def test_shape_mismatch_fails_and_restores_the_tensors_already_swapped(base_sd, overlay_sd) -> None:
    base = _model_from_checkpoint(base_sd)
    snapshot = {k: v.clone() for k, v in base.state_dict().items()}
    params = {
        "transformer_blocks.0.adaln_proj.linear.bias": overlay_sd["blocks.0.adaln_proj.linear.bias"],
        # A full (non-pruned) overlay's 2688-wide projection on a pruned base.
        "transformer_blocks.1.adaln_proj.linear.weight": torch.zeros(6 * HIDDEN * 3, 2688),
    }
    with pytest.raises(ValueError, match="both be AdaLN-pruned or both be full"):
        with apply_minimax_h3_adaln_overlay(base, params):
            pass
    for key, value in base.state_dict().items():
        assert torch.equal(value, snapshot[key]), key


def test_lora_deltas_land_on_the_overlaid_weights_in_the_denoise_nodes_order(
    tmp_path: Path, base_sd, overlay_sd
) -> None:
    """The denoise node enters the overlay BEFORE the LoRA patches. In that order a LoRA touching
    an overlaid AdaLN projection adds onto the hybrid weight and everything unwinds to the base;
    in the reverse order the patcher's restore (from the cache's copy) would silently discard the
    LoRA delta on those blocks. Direct patching is forced because on CPU the patcher would
    otherwise sidecar, hiding the weight-writing path the GPU takes."""
    base = _model_from_checkpoint(base_sd)
    cached = {k: v.clone() for k, v in base.state_dict().items()}
    path = "transformer_blocks.1.adaln_proj.linear"
    linear = base.get_submodule(path)
    down = torch.randn(2, linear.in_features, generator=torch.Generator().manual_seed(11))
    up = torch.randn(linear.out_features, 2, generator=torch.Generator().manual_seed(12))
    lora = ModelPatchRaw({MINIMAX_H3_LORA_TRANSFORMER_PREFIX + path: LoRALayer(up, None, down, None, None)})
    params = _overlay(tmp_path, overlay_sd, 1, 1).tensors_for(None)
    overlay_weight = overlay_sd["blocks.1.adaln_proj.linear.weight"]

    with ExitStack() as stack:
        stack.enter_context(apply_minimax_h3_adaln_overlay(base, params, cached))
        stack.enter_context(
            LayerPatcher.apply_smart_model_patches(
                model=base,
                patches=[(lora, 1.0)],
                prefix=MINIMAX_H3_LORA_TRANSFORMER_PREFIX,
                dtype=torch.float32,
                cached_weights=cached,
                force_direct_patching=True,
            )
        )
        assert torch.allclose(linear.weight, overlay_weight + up @ down, atol=1e-5)
    assert torch.equal(linear.weight, base_sd["blocks.1.adaln_proj.linear.weight"])
    assert torch.equal(cached[path + ".weight"], base_sd["blocks.1.adaln_proj.linear.weight"])


def test_unknown_parameter_path_is_an_error(base_sd) -> None:
    base = _model_from_checkpoint(base_sd)
    with pytest.raises(ValueError, match="has no parameter"):
        with apply_minimax_h3_adaln_overlay(base, {"transformer_blocks.7.adaln_proj.linear.bias": torch.zeros(1)}):
            pass
