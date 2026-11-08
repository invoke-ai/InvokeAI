"""Tests for Qwen Image LoRA conversion utilities."""

import logging

import pytest
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.qwen_image_lora_constants import (
    QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX,
    QWEN_IMAGE_LORA_QWEN_VL_PREFIX,
)
from invokeai.backend.patches.lora_conversions.qwen_image_lora_conversion_utils import (
    _convert_kohya_key,
    is_state_dict_likely_kohya_qwen_image,
    lora_model_from_qwen_image_state_dict,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.qwen_image_lora_diffusers_format import (
    state_dict_keys as diffusers_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.qwen_image_lora_kohya_format import (
    state_dict_keys as kohya_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.qwen_image_lora_kohya_with_te_format import (
    state_dict_keys as kohya_with_te_state_dict_keys,
)
from tests.backend.patches.lora_conversions.lora_state_dicts.utils import keys_to_mock_state_dict

# ---- Format detection tests ----


def test_is_kohya_format_true():
    """Kohya-format state dict is correctly identified."""
    state_dict = keys_to_mock_state_dict(kohya_state_dict_keys)
    assert is_state_dict_likely_kohya_qwen_image(state_dict)


def test_is_kohya_format_false_diffusers():
    """Diffusers-format state dict is not identified as Kohya."""
    state_dict = keys_to_mock_state_dict(diffusers_state_dict_keys)
    assert not is_state_dict_likely_kohya_qwen_image(state_dict)


def test_is_kohya_format_false_empty():
    """Empty state dict is not identified as Kohya."""
    assert not is_state_dict_likely_kohya_qwen_image({})


# ---- Kohya key conversion tests ----


def test_convert_kohya_key_attention():
    """Kohya attention projection keys convert correctly."""
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_attn_to_k") == "transformer_blocks.0.attn.to_k"
    assert _convert_kohya_key("lora_unet_transformer_blocks_5_attn_to_q") == "transformer_blocks.5.attn.to_q"
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_attn_to_v") == "transformer_blocks.0.attn.to_v"
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_attn_to_out_0") == "transformer_blocks.0.attn.to_out.0"
    assert (
        _convert_kohya_key("lora_unet_transformer_blocks_0_attn_add_k_proj") == "transformer_blocks.0.attn.add_k_proj"
    )


def test_convert_kohya_key_mlp():
    """Kohya MLP keys convert correctly."""
    assert (
        _convert_kohya_key("lora_unet_transformer_blocks_0_img_mlp_net_0_proj")
        == "transformer_blocks.0.img_mlp.net.0.proj"
    )
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_txt_mlp_net_2") == "transformer_blocks.0.txt_mlp.net.2"


def test_convert_kohya_key_unknown_returns_none():
    """Unknown Kohya sub-module returns None."""
    assert _convert_kohya_key("lora_unet_transformer_blocks_0_unknown_projection") is None


def test_convert_kohya_key_non_matching_returns_none():
    """Key that doesn't match the regex returns None."""
    assert _convert_kohya_key("some_random_key") is None


# ---- Full model conversion tests ----


def test_kohya_conversion_produces_correct_layer_keys():
    """Kohya state dict converts to ModelPatchRaw with correct prefixed layer keys."""
    state_dict = keys_to_mock_state_dict(kohya_state_dict_keys)
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    # Build expected keys: convert each Kohya layer name to model path, add prefix
    expected_keys: set[str] = set()
    for k in kohya_state_dict_keys:
        layer_name = k.split(".", 1)[0]  # e.g. lora_unet_transformer_blocks_0_attn_to_k
        model_path = _convert_kohya_key(layer_name)
        if model_path is not None:
            expected_keys.add(f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}{model_path}")

    assert set(model.layers.keys()) == expected_keys
    assert len(model.layers) > 0


def test_diffusers_conversion_produces_correct_layer_keys():
    """Diffusers state dict converts to ModelPatchRaw with correct prefixed layer keys."""
    state_dict = keys_to_mock_state_dict(diffusers_state_dict_keys)
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    expected_keys = {
        f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k",
        f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_q",
        f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.1.attn.to_k",
    }

    assert set(model.layers.keys()) == expected_keys


@pytest.mark.parametrize("prefix", ["transformer.", "diffusion_model.", "base_model.model."])
def test_diffusers_key_prefix_is_stripped(prefix: str):
    """The diffusers (`transformer.`), ComfyUI (`diffusion_model.`) and raw PEFT (`base_model.model.`) prefixes
    are stripped, so the layer names a module of the transformer."""
    state_dict = {
        f"{prefix}transformer_blocks.0.attn.to_k.lora_down.weight": torch.empty(64, 3072),
        f"{prefix}transformer_blocks.0.attn.to_k.lora_up.weight": torch.empty(3072, 64),
    }
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    assert set(model.layers) == {f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k"}


# ---- Unknown key handling tests ----


def test_kohya_unknown_submodule_is_skipped():
    """Unknown Kohya sub-modules are skipped, producing no layers for them."""
    state_dict = {
        # Known key — should produce a layer
        "lora_unet_transformer_blocks_0_attn_to_k.lokr_w1": torch.empty(3072, 16),
        "lora_unet_transformer_blocks_0_attn_to_k.lokr_w2": torch.empty(16, 3072),
        # Unknown key — should be skipped
        "lora_unet_transformer_blocks_0_unknown_projection.lokr_w1": torch.empty(3072, 16),
        "lora_unet_transformer_blocks_0_unknown_projection.lokr_w2": torch.empty(16, 3072),
    }
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    # Only the known key should produce a layer
    assert len(model.layers) == 1
    expected_key = f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k"
    assert expected_key in model.layers


def test_kohya_all_unknown_submodules_produces_empty_model():
    """State dict with only unknown Kohya sub-modules produces an empty ModelPatchRaw."""
    state_dict = {
        "lora_unet_transformer_blocks_0_totally_unknown.lokr_w1": torch.empty(3072, 16),
        "lora_unet_transformer_blocks_0_totally_unknown.lokr_w2": torch.empty(16, 3072),
    }
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    assert len(model.layers) == 0


def test_peft_dora_magnitude_is_kept_as_a_dora_layer():
    """A PEFT DoRA adapter must load as DoRA: dropping `lora_magnitude_vector` would silently apply a plain LoRA."""
    from invokeai.backend.patches.layers.dora_layer import DoRALayer

    magnitude = torch.rand(3072)
    state_dict = {
        "base_model.model.transformer_blocks.0.attn.to_k.lora_A.weight": torch.empty(64, 3072),
        "base_model.model.transformer_blocks.0.attn.to_k.lora_B.weight": torch.empty(3072, 64),
        "base_model.model.transformer_blocks.0.attn.to_k.lora_magnitude_vector.weight": magnitude,
    }
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    layer = model.layers[f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k"]
    assert isinstance(layer, DoRALayer)
    assert layer.magnitude_is_out_dim
    assert torch.equal(layer.dora_scale, magnitude)


# ---- Text encoder layers ----


def test_kohya_text_encoder_layers_are_routed_to_the_qwen_vl_prefix():
    """`lora_te_*` layers name Qwen2.5-VL modules (language tower, vision tower, merger, lm_head); the transformer's
    layers in the same file keep their prefix."""
    state_dict = keys_to_mock_state_dict(kohya_with_te_state_dict_keys)
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    t = QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX
    te = QWEN_IMAGE_LORA_QWEN_VL_PREFIX
    assert set(model.layers) == {
        f"{t}transformer_blocks.0.attn.to_k",
        f"{t}transformer_blocks.0.img_mlp.net.2",
        f"{te}model.language_model.layers.0.self_attn.q_proj",
        f"{te}model.language_model.layers.0.self_attn.k_proj",
        f"{te}model.language_model.layers.0.self_attn.v_proj",
        f"{te}model.language_model.layers.0.self_attn.o_proj",
        f"{te}model.language_model.layers.0.mlp.gate_proj",
        f"{te}model.language_model.layers.0.mlp.up_proj",
        f"{te}model.language_model.layers.0.mlp.down_proj",
        f"{te}model.visual.blocks.0.attn.qkv",
        f"{te}model.visual.blocks.0.attn.proj",
        f"{te}model.visual.blocks.0.mlp.gate_proj",
        f"{te}model.visual.blocks.0.mlp.down_proj",
        f"{te}model.visual.merger.mlp.2",
        f"{te}lm_head",
    }


@pytest.mark.parametrize("prefix", ["text_encoder.", "base_model.model.text_encoder."])
def test_diffusers_text_encoder_layers_are_routed_to_the_qwen_vl_prefix(prefix: str):
    """`base_model.model.` is also a transformer prefix, so it must not swallow the text-encoder layers."""
    state_dict = {
        f"{prefix}model.language_model.layers.3.mlp.down_proj.lora_A.weight": torch.empty(8, 18944),
        f"{prefix}model.language_model.layers.3.mlp.down_proj.lora_B.weight": torch.empty(3584, 8),
        f"{prefix}model.visual.blocks.1.attn.qkv.lora_A.weight": torch.empty(8, 1280),
        f"{prefix}model.visual.blocks.1.attn.qkv.lora_B.weight": torch.empty(3840, 8),
        "base_model.model.transformer_blocks.0.attn.to_k.lora_A.weight": torch.empty(8, 3072),
        "base_model.model.transformer_blocks.0.attn.to_k.lora_B.weight": torch.empty(3072, 8),
    }
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    assert set(model.layers) == {
        f"{QWEN_IMAGE_LORA_QWEN_VL_PREFIX}model.language_model.layers.3.mlp.down_proj",
        f"{QWEN_IMAGE_LORA_QWEN_VL_PREFIX}model.visual.blocks.1.attn.qkv",
        f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k",
    }


def test_unmapped_kohya_layers_produce_one_warning_with_the_count(caplog: pytest.LogCaptureFixture):
    """Layers that name no known module (an unknown transformer module, a text-encoder Linear outside the tree, a bare
    prefix of a real path) are skipped with a single warning, however many there are."""
    state_dict = {
        "lora_unet_transformer_blocks_0_attn_to_k.lora_down.weight": torch.empty(8, 3072),
        "lora_unet_transformer_blocks_0_attn_to_k.lora_up.weight": torch.empty(3072, 8),
    }
    unmapped = [
        "lora_unet_transformer_blocks_0_unknown_projection",
        "lora_te_model_visual_patch_embed_proj",
        "lora_te_model_language_model_layers_0_mlp",
        "lora_te_model_language_model_layers_0_mlp_down_proj_extra",
    ]
    for layer in unmapped:
        state_dict[f"{layer}.lora_down.weight"] = torch.empty(8, 16)
        state_dict[f"{layer}.lora_up.weight"] = torch.empty(16, 8)

    with caplog.at_level(logging.WARNING):
        model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    assert set(model.layers) == {f"{QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_k"}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Skipped 4 " in warnings[0]


def test_every_linear_of_a_qwen2_5_vl_encoder_is_reachable_from_its_onetrainer_key():
    """The Kohya key of each Linear in the real module tree converts to a path the patcher resolves to that very
    module. Built from the model's own names, so a module missing from the parsing tree fails here."""
    config = Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 64,
            "fullatt_block_indexes": [1],
        },
    )
    with torch.device("meta"):
        encoder = Qwen2_5_VLForConditionalGeneration(config)
    linears = {name: module for name, module in encoder.named_modules() if isinstance(module, torch.nn.Linear)}
    assert any(name.startswith("model.visual.merger") for name in linears)

    state_dict = {}
    for name in linears:
        layer = "lora_te_" + name.replace(".", "_")
        state_dict[f"{layer}.lora_down.weight"] = torch.empty(4, 8)
        state_dict[f"{layer}.lora_up.weight"] = torch.empty(8, 4)
    state_dict["lora_unet_transformer_blocks_0_attn_to_k.lora_down.weight"] = torch.empty(4, 8)
    state_dict["lora_unet_transformer_blocks_0_attn_to_k.lora_up.weight"] = torch.empty(8, 4)
    model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    resolved = {}
    for layer_key in model.layers:
        if layer_key.startswith(QWEN_IMAGE_LORA_QWEN_VL_PREFIX):
            module_key, module = LayerPatcher._get_submodule(
                encoder, layer_key[len(QWEN_IMAGE_LORA_QWEN_VL_PREFIX) :], layer_key_is_flattened=False
            )
            resolved[module_key] = module
    assert resolved.keys() == linears.keys()
    assert all(resolved[name] is linears[name] for name in linears)


@pytest.mark.parametrize("prefix", ["text_encoder.", "base_model.model.text_encoder."])
def test_diffusers_text_encoder_layers_that_are_not_encoder_linears_are_skipped_with_one_warning(
    prefix: str, caplog: pytest.LogCaptureFixture
):
    """A Conv3d (`patch_embed`) cannot hold a patch and would crash the patcher, and the transformers < 4.52 layout
    (`model.layers.N`) names nothing in the current encoder. Both join the single counted warning."""
    state_dict = {
        f"{prefix}model.language_model.layers.0.mlp.down_proj.lora_A.weight": torch.empty(8, 18944),
        f"{prefix}model.language_model.layers.0.mlp.down_proj.lora_B.weight": torch.empty(3584, 8),
        f"{prefix}model.visual.patch_embed.proj.lora_A.weight": torch.empty(8, 1176),
        f"{prefix}model.visual.patch_embed.proj.lora_B.weight": torch.empty(1280, 8),
        f"{prefix}model.layers.0.mlp.down_proj.lora_A.weight": torch.empty(8, 18944),
        f"{prefix}model.layers.0.mlp.down_proj.lora_B.weight": torch.empty(3584, 8),
        f"{prefix}model.language_model.layers.0.mlp.lora_A.weight": torch.empty(8, 16),
        f"{prefix}model.language_model.layers.0.mlp.lora_B.weight": torch.empty(16, 8),
    }

    with caplog.at_level(logging.WARNING):
        model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    assert set(model.layers) == {f"{QWEN_IMAGE_LORA_QWEN_VL_PREFIX}model.language_model.layers.0.mlp.down_proj"}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Skipped 3 " in warnings[0]


def test_onetrainer_comfy_text_encoder_keys_are_routed_beside_diffusion_model_transformer_keys(
    caplog: pytest.LogCaptureFixture,
):
    """OneTrainer's COMFY_LORA export writes encoder layers as `text_encoders.qwen25_7b.transformer.<module>`."""
    te = "text_encoders.qwen25_7b.transformer."
    modules = {
        f"{te}model.language_model.layers.0.mlp.down_proj": (18944, 3584),
        f"{te}model.visual.blocks.0.attn.qkv": (1280, 3840),
        f"{te}model.visual.merger.mlp.0": (5120, 5120),
        f"{te}lm_head": (3584, 152064),
        "diffusion_model.transformer_blocks.0.attn.to_k": (3072, 3072),
    }
    state_dict = {}
    for layer, (in_features, out_features) in modules.items():
        state_dict[f"{layer}.lora_down.weight"] = torch.empty(8, in_features)
        state_dict[f"{layer}.lora_up.weight"] = torch.empty(out_features, 8)

    with caplog.at_level(logging.WARNING):
        model = lora_model_from_qwen_image_state_dict(state_dict, alpha=None)

    t, v = QWEN_IMAGE_EDIT_LORA_TRANSFORMER_PREFIX, QWEN_IMAGE_LORA_QWEN_VL_PREFIX
    assert set(model.layers) == {
        f"{t}transformer_blocks.0.attn.to_k",
        f"{v}model.language_model.layers.0.mlp.down_proj",
        f"{v}model.visual.blocks.0.attn.qkv",
        f"{v}model.visual.merger.mlp.0",
        f"{v}lm_head",
    }
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
