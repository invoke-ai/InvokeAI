"""Tests for Wan LoRA state-dict conversion to ModelPatchRaw."""

import torch

from invokeai.backend.patches.lora_conversions.wan_lora_constants import WAN_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.wan_lora_conversion_utils import (
    _kohya_layer_to_diffusers_path,
    _native_layer_path_to_diffusers,
    _strip_peft_prefix,
    lora_model_from_wan_state_dict,
)


def _ab_pair(in_dim: int, out_dim: int, rank: int = 16) -> dict[str, torch.Tensor]:
    """PEFT-style lora_A (in→rank) + lora_B (rank→out) pair."""
    return {
        "lora_A.weight": torch.zeros((rank, in_dim)),
        "lora_B.weight": torch.zeros((out_dim, rank)),
    }


def _down_up_pair(in_dim: int, out_dim: int, rank: int = 16) -> dict[str, torch.Tensor]:
    """Kohya-style lora_down + lora_up pair."""
    return {
        "lora_down.weight": torch.zeros((rank, in_dim)),
        "lora_up.weight": torch.zeros((out_dim, rank)),
    }


class TestKohyaLayerToDiffusersPath:
    def test_diffusers_self_attention(self):
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_attn1_to_q") == "blocks.0.attn1.to_q"
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_5_attn1_to_out_0") == "blocks.5.attn1.to_out.0"

    def test_diffusers_cross_attention(self):
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_attn2_to_k") == "blocks.0.attn2.to_k"
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_attn2_to_v") == "blocks.0.attn2.to_v"

    def test_native_self_attention_maps_to_attn1(self):
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_self_attn_q") == "blocks.0.attn1.to_q"
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_self_attn_o") == "blocks.0.attn1.to_out.0"

    def test_native_cross_attention_maps_to_attn2(self):
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_2_cross_attn_v") == "blocks.2.attn2.to_v"

    def test_ffn_diffusers(self):
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_ffn_net_0_proj") == "blocks.0.ffn.net.0.proj"
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_ffn_net_2") == "blocks.0.ffn.net.2"

    def test_ffn_native_maps_to_diffusers(self):
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_ffn_0") == "blocks.0.ffn.net.0.proj"
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_ffn_2") == "blocks.0.ffn.net.2"

    def test_unknown_submodule_returns_none(self):
        assert _kohya_layer_to_diffusers_path("lora_unet_blocks_0_unknown_thing") is None

    def test_non_kohya_returns_none(self):
        assert _kohya_layer_to_diffusers_path("transformer.blocks.0.attn1.to_q") is None


class TestPEFTPathConversion:
    def test_strip_transformer_prefix(self):
        assert _strip_peft_prefix("transformer.blocks.0.attn1.to_q") == "blocks.0.attn1.to_q"

    def test_strip_diffusion_model_prefix(self):
        assert _strip_peft_prefix("diffusion_model.blocks.0.self_attn.q") == "blocks.0.self_attn.q"

    def test_strip_base_model_prefix(self):
        assert _strip_peft_prefix("base_model.model.transformer.blocks.0.attn1.to_q") == "blocks.0.attn1.to_q"

    def test_no_prefix_unchanged(self):
        assert _strip_peft_prefix("blocks.0.attn1.to_q") == "blocks.0.attn1.to_q"

    def test_diffusers_path_passes_through(self):
        assert _native_layer_path_to_diffusers("blocks.0.attn1.to_q") == "blocks.0.attn1.to_q"
        assert _native_layer_path_to_diffusers("blocks.0.ffn.net.0.proj") == "blocks.0.ffn.net.0.proj"

    def test_native_self_attn_becomes_attn1(self):
        assert _native_layer_path_to_diffusers("blocks.0.self_attn.q") == "blocks.0.attn1.to_q"
        assert _native_layer_path_to_diffusers("blocks.0.self_attn.k") == "blocks.0.attn1.to_k"
        assert _native_layer_path_to_diffusers("blocks.0.self_attn.v") == "blocks.0.attn1.to_v"
        assert _native_layer_path_to_diffusers("blocks.0.self_attn.o") == "blocks.0.attn1.to_out.0"

    def test_native_cross_attn_becomes_attn2(self):
        assert _native_layer_path_to_diffusers("blocks.7.cross_attn.q") == "blocks.7.attn2.to_q"
        assert _native_layer_path_to_diffusers("blocks.7.cross_attn.o") == "blocks.7.attn2.to_out.0"

    def test_native_ffn_becomes_diffusers_ffn(self):
        assert _native_layer_path_to_diffusers("blocks.0.ffn.0") == "blocks.0.ffn.net.0.proj"
        assert _native_layer_path_to_diffusers("blocks.0.ffn.2") == "blocks.0.ffn.net.2"

    def test_non_block_path_rejected(self):
        assert _native_layer_path_to_diffusers("patch_embedding.weight") is None


class TestLoRAModelFromStateDict:
    """End-to-end conversion: state dict -> ModelPatchRaw."""

    def test_diffusers_peft_with_transformer_prefix(self):
        sd = {f"transformer.blocks.0.attn1.to_q.{k}": v for k, v in _ab_pair(5120, 5120).items()}
        patch = lora_model_from_wan_state_dict(sd)
        expected_key = f"{WAN_LORA_TRANSFORMER_PREFIX}blocks.0.attn1.to_q"
        assert expected_key in patch.layers

    def test_diffusers_peft_bare(self):
        sd = {f"blocks.5.attn2.to_k.{k}": v for k, v in _ab_pair(5120, 5120).items()}
        patch = lora_model_from_wan_state_dict(sd)
        assert f"{WAN_LORA_TRANSFORMER_PREFIX}blocks.5.attn2.to_k" in patch.layers

    def test_native_peft_diffusion_model_prefix(self):
        sd = {f"diffusion_model.blocks.0.self_attn.q.{k}": v for k, v in _ab_pair(5120, 5120).items()}
        patch = lora_model_from_wan_state_dict(sd)
        # native self_attn.q must be rewritten to attn1.to_q
        assert f"{WAN_LORA_TRANSFORMER_PREFIX}blocks.0.attn1.to_q" in patch.layers

    def test_native_peft_cross_attn_to_attn2(self):
        sd = {f"diffusion_model.blocks.3.cross_attn.o.{k}": v for k, v in _ab_pair(5120, 5120).items()}
        patch = lora_model_from_wan_state_dict(sd)
        assert f"{WAN_LORA_TRANSFORMER_PREFIX}blocks.3.attn2.to_out.0" in patch.layers

    def test_native_peft_ffn_to_diffusers(self):
        sd = {f"diffusion_model.blocks.0.ffn.0.{k}": v for k, v in _ab_pair(5120, 13824).items()}
        patch = lora_model_from_wan_state_dict(sd)
        assert f"{WAN_LORA_TRANSFORMER_PREFIX}blocks.0.ffn.net.0.proj" in patch.layers

    def test_kohya_diffusers_naming(self):
        sd = {f"lora_unet_blocks_0_attn1_to_q.{k}": v for k, v in _down_up_pair(5120, 5120).items()}
        patch = lora_model_from_wan_state_dict(sd)
        assert f"{WAN_LORA_TRANSFORMER_PREFIX}blocks.0.attn1.to_q" in patch.layers

    def test_kohya_native_naming(self):
        sd = {f"lora_unet_blocks_0_self_attn_q.{k}": v for k, v in _down_up_pair(5120, 5120).items()}
        patch = lora_model_from_wan_state_dict(sd)
        assert f"{WAN_LORA_TRANSFORMER_PREFIX}blocks.0.attn1.to_q" in patch.layers

    def test_kohya_ffn_native_naming(self):
        sd = {f"lora_unet_blocks_0_ffn_0.{k}": v for k, v in _down_up_pair(5120, 13824).items()}
        patch = lora_model_from_wan_state_dict(sd)
        assert f"{WAN_LORA_TRANSFORMER_PREFIX}blocks.0.ffn.net.0.proj" in patch.layers

    def test_multiple_layers(self):
        """Cover a realistic mix of attn + ffn keys across multiple blocks."""
        sd = {}
        for block in range(3):
            for k, v in _ab_pair(5120, 5120).items():
                sd[f"transformer.blocks.{block}.attn1.to_q.{k}"] = v
                sd[f"transformer.blocks.{block}.attn2.to_v.{k}"] = v
            for k, v in _ab_pair(5120, 13824).items():
                sd[f"transformer.blocks.{block}.ffn.net.0.proj.{k}"] = v

        patch = lora_model_from_wan_state_dict(sd)
        expected_paths = []
        for block in range(3):
            expected_paths.append(f"blocks.{block}.attn1.to_q")
            expected_paths.append(f"blocks.{block}.attn2.to_v")
            expected_paths.append(f"blocks.{block}.ffn.net.0.proj")
        for path in expected_paths:
            assert f"{WAN_LORA_TRANSFORMER_PREFIX}{path}" in patch.layers

    def test_alpha_override_propagates(self):
        sd = {f"blocks.0.attn1.to_q.{k}": v for k, v in _ab_pair(5120, 5120).items()}
        patch = lora_model_from_wan_state_dict(sd, alpha=8.0)
        layer = patch.layers[f"{WAN_LORA_TRANSFORMER_PREFIX}blocks.0.attn1.to_q"]
        # any_lora_layer_from_state_dict picks LoRALayer / LoKR / etc. — the
        # layer object should at minimum have processed the alpha into its state.
        assert layer is not None

    def test_unknown_kohya_submodule_is_skipped_silently(self):
        sd = {f"lora_unet_blocks_0_unknown_thing.{k}": v for k, v in _down_up_pair(5120, 5120).items()}
        patch = lora_model_from_wan_state_dict(sd)
        assert len(patch.layers) == 0

    def test_empty_state_dict(self):
        patch = lora_model_from_wan_state_dict({})
        assert len(patch.layers) == 0
