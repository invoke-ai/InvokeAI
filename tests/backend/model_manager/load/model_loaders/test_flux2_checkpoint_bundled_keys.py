"""Tests for filtering non-transformer keys from bundled FLUX.2 checkpoints.

Some third-party FLUX.2 .safetensors files are combined checkpoints that bundle
text encoder (text_encoders.*) and VAE (vae.*) weights alongside the transformer
(model.diffusion_model.*) weights. The loader must filter these out before calling
load_state_dict, or it will raise RuntimeError for unexpected keys.
"""

import torch


def _filter_bundled_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reproduce the filtering logic from Flux2CheckpointModel._load_from_singlefile."""
    # Step 1: Strip ComfyUI-style prefix
    prefix_to_strip = None
    for prefix in ["model.diffusion_model.", "diffusion_model."]:
        if any(k.startswith(prefix) for k in sd.keys()):
            prefix_to_strip = prefix
            break

    if prefix_to_strip:
        sd = {(k[len(prefix_to_strip) :] if k.startswith(prefix_to_strip) else k): v for k, v in sd.items()}

    # Step 2: Filter non-transformer keys
    non_transformer_prefixes = ("text_encoders.", "vae.")
    keys_to_remove = [k for k in sd if isinstance(k, str) and k.startswith(non_transformer_prefixes)]
    for k in keys_to_remove:
        del sd[k]

    return sd


def _make_bundled_state_dict() -> dict[str, torch.Tensor]:
    """Create a synthetic bundled state dict mimicking a combined FLUX.2 checkpoint."""
    dummy = torch.zeros(1)
    sd: dict[str, torch.Tensor] = {}

    # Transformer keys (under model.diffusion_model.* prefix, BFL format)
    transformer_keys = [
        "model.diffusion_model.img_in.weight",
        "model.diffusion_model.txt_in.weight",
        "model.diffusion_model.time_in.in_layer.weight",
        "model.diffusion_model.time_in.out_layer.weight",
        "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
        "model.diffusion_model.single_blocks.0.linear1.weight",
        "model.diffusion_model.final_layer.linear.weight",
    ]
    for k in transformer_keys:
        sd[k] = dummy.clone()

    # Text encoder keys (should be filtered out)
    text_encoder_keys = [
        "text_encoders.qwen3_8b.transformer.model.embed_tokens.weight",
        "text_encoders.qwen3_8b.transformer.model.layers.0.input_layernorm.weight",
        "text_encoders.qwen3_8b.transformer.model.layers.0.mlp.down_proj.weight",
        "text_encoders.qwen3_8b.logit_scale",
    ]
    for k in text_encoder_keys:
        sd[k] = dummy.clone()

    # VAE keys (should be filtered out)
    vae_keys = [
        "vae.decoder.conv_in.weight",
        "vae.decoder.conv_out.weight",
        "vae.encoder.conv_in.weight",
        "vae.bn.running_mean",
    ]
    for k in vae_keys:
        sd[k] = dummy.clone()

    return sd


def test_bundled_checkpoint_filters_text_encoder_and_vae_keys() -> None:
    """Bundled checkpoints should have text_encoders.* and vae.* keys removed."""
    sd = _make_bundled_state_dict()
    total_keys = len(sd)
    assert total_keys == 15  # 7 transformer + 4 text encoder + 4 vae

    filtered = _filter_bundled_keys(sd)

    # Only transformer keys should remain (with prefix stripped)
    assert len(filtered) == 7
    assert not any(k.startswith("text_encoders.") for k in filtered)
    assert not any(k.startswith("vae.") for k in filtered)
    assert not any(k.startswith("model.diffusion_model.") for k in filtered)

    # Verify transformer keys had their prefix stripped
    assert "img_in.weight" in filtered
    assert "txt_in.weight" in filtered
    assert "double_blocks.0.img_attn.qkv.weight" in filtered


def test_non_bundled_checkpoint_unaffected() -> None:
    """Transformer-only checkpoints (no text_encoders/vae keys) should pass through unchanged."""
    dummy = torch.zeros(1)
    sd = {
        "model.diffusion_model.img_in.weight": dummy,
        "model.diffusion_model.txt_in.weight": dummy,
        "model.diffusion_model.double_blocks.0.img_attn.qkv.weight": dummy,
    }

    filtered = _filter_bundled_keys(sd)

    assert len(filtered) == 3
    assert "img_in.weight" in filtered
    assert "txt_in.weight" in filtered


def test_checkpoint_without_prefix_unaffected() -> None:
    """Checkpoints already in unprefixed BFL format should pass through unchanged."""
    dummy = torch.zeros(1)
    sd = {
        "img_in.weight": dummy,
        "txt_in.weight": dummy,
        "double_blocks.0.img_attn.qkv.weight": dummy,
    }

    filtered = _filter_bundled_keys(sd)

    assert len(filtered) == 3
    assert "img_in.weight" in filtered
