"""Unit tests for the pure state-dict helpers in the Qwen-Image / Qwen-VL loader.

These freeze the checkpoint key-surgery that the loaders perform before instantiating a model,
so a regression like the transformers-5.x one (where `_checkpoint_conversion_mapping` became
`{}` and the `visual.* -> model.visual.*` remap was silently skipped) fails here instead of at
the user's first load.
"""

import torch

from invokeai.backend.model_manager.load.model_loaders.qwen_image import (
    _build_qwen_image_transformer_config,
    _dequantize_comfyui_fp8,
    _remap_qwen_vl_checkpoint_keys,
    _strip_comfyui_prefix,
    _strip_quantization_metadata,
)
from tests.backend.model_manager.load.state_dicts.qwen_vl_encoder_comfyui_keys import (
    state_dict_keys as qwen_vl_keys,
)
from tests.backend.model_manager.load.state_dicts.utils import keys_to_mock_state_dict

# Prefixes the Qwen2.5-VL architecture (transformers >=4.50) actually exposes. Frozen here on
# purpose: if a remap regresses, converted keys stop matching this set.
_VALID_QWEN_VL_PREFIXES = ("model.visual.", "model.language_model.", "lm_head")


class TestRemapQwenVlCheckpointKeys:
    def test_every_key_maps_to_the_transformers_layout(self):
        """Every legacy ComfyUI key must land under `model.visual.*` / `model.language_model.*`."""
        sd = keys_to_mock_state_dict(qwen_vl_keys)
        # The loader strips fp8 metadata (`scaled_fp8` etc.) before remapping; mirror that order.
        _strip_quantization_metadata(sd)

        remapped = _remap_qwen_vl_checkpoint_keys(sd)

        assert len(remapped) == len(sd)
        for key in remapped:
            assert key.startswith(_VALID_QWEN_VL_PREFIXES), f"key not remapped to a known layout: {key}"
        # No key may survive in the legacy layout.
        assert not any(k.startswith("visual.") for k in remapped)
        assert not any(k.startswith(("model.layers.", "model.embed_tokens", "model.norm")) for k in remapped)

    def test_specific_keys_from_the_bug_report(self):
        """The exact keys that failed to load in the original bug report are remapped."""
        sd = {
            "visual.blocks.0.attn.qkv.weight": torch.empty(1),
            "visual.patch_embed.proj.weight": torch.empty(1),
            "model.layers.0.self_attn.q_proj.weight": torch.empty(1),
            "model.embed_tokens.weight": torch.empty(1),
            "lm_head.weight": torch.empty(1),
        }

        remapped = _remap_qwen_vl_checkpoint_keys(sd)

        assert "model.visual.blocks.0.attn.qkv.weight" in remapped
        assert "model.visual.patch_embed.proj.weight" in remapped
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in remapped
        assert "model.language_model.embed_tokens.weight" in remapped
        assert "lm_head.weight" in remapped  # unchanged

    def test_idempotent_on_already_converted_layout(self):
        """Re-running the remap on new-layout keys must not double-prefix them."""
        sd = keys_to_mock_state_dict(qwen_vl_keys)

        once = _remap_qwen_vl_checkpoint_keys(sd)
        twice = _remap_qwen_vl_checkpoint_keys(once)

        assert set(once.keys()) == set(twice.keys())

    def test_fallback_when_transformers_mapping_is_empty(self, monkeypatch):
        """Even if transformers stops providing `_checkpoint_conversion_mapping`, the remap fires.

        transformers 5.x returns `{}` here; forcing that value pins the fallback that fixes the
        original bug.
        """
        from transformers import Qwen2_5_VLForConditionalGeneration

        monkeypatch.setattr(Qwen2_5_VLForConditionalGeneration, "_checkpoint_conversion_mapping", {})

        remapped = _remap_qwen_vl_checkpoint_keys(
            {
                "visual.blocks.0.norm1.weight": torch.empty(1),
                "model.layers.0.input_layernorm.weight": torch.empty(1),
            }
        )

        assert "model.visual.blocks.0.norm1.weight" in remapped
        assert "model.language_model.layers.0.input_layernorm.weight" in remapped


class TestStripQuantizationMetadata:
    def test_drops_fp8_metadata_keeps_weights(self):
        sd = keys_to_mock_state_dict(qwen_vl_keys)
        # The captured checkpoint is fp8_scaled, so it really does ship this metadata.
        assert any(k.endswith((".scale_weight", ".scale_input")) or k == "scaled_fp8" for k in sd)
        n_weights_before = sum(1 for k in sd if k.endswith(".weight"))

        _strip_quantization_metadata(sd)

        assert not any(
            k.endswith((".scale_weight", ".scale_input")) or "comfy_quant" in k or k == "scaled_fp8" for k in sd
        )
        # Real weights are untouched.
        assert sum(1 for k in sd if k.endswith(".weight")) == n_weights_before


class TestDequantizeComfyuiFp8:
    def test_scalar_scale(self):
        sd = {
            "l.weight": torch.full((2, 2), 2.0),
            "l.scale_weight": torch.tensor(3.0),
            "l.scale_input": torch.tensor(9.0),  # activation scale, must be ignored
        }

        count = _dequantize_comfyui_fp8(sd, torch.float32)

        assert count == 1
        assert torch.allclose(sd["l.weight"], torch.full((2, 2), 6.0))

    def test_block_wise_scale_is_broadcast(self):
        # Per-block scale [2, 1] must be repeat_interleaved up to the weight shape [4, 2].
        sd = {
            "l.weight": torch.ones(4, 2),
            "l.weight_scale": torch.tensor([[10.0], [20.0]]),
        }

        count = _dequantize_comfyui_fp8(sd, torch.float32)

        assert count == 1
        expected = torch.tensor([[10.0, 10.0], [10.0, 10.0], [20.0, 20.0], [20.0, 20.0]])
        assert torch.allclose(sd["l.weight"], expected)


class TestStripComfyuiPrefix:
    def test_strips_diffusion_model_prefix(self):
        sd = {
            "model.diffusion_model.transformer_blocks.0.img_mod.1.weight": torch.empty(1),
            "model.diffusion_model.img_in.weight": torch.empty(1),
        }
        out = _strip_comfyui_prefix(sd)
        assert set(out.keys()) == {"transformer_blocks.0.img_mod.1.weight", "img_in.weight"}

    def test_no_prefix_is_a_noop(self):
        sd = {"transformer_blocks.0.x": torch.empty(1)}
        assert _strip_comfyui_prefix(sd) is sd


class TestBuildQwenImageTransformerConfig:
    def test_infers_layer_count_and_dims_from_shapes(self):
        # torch-order (logical) shapes, as the GGMLTensor.tensor_shape / safetensors path exposes.
        sd = {
            "img_in.weight": torch.empty(3072, 64),
            "txt_in.weight": torch.empty(3072, 3584),
            "transformer_blocks.0.img_mod.1.weight": torch.empty(1),
            "transformer_blocks.1.img_mod.1.weight": torch.empty(1),
            "transformer_blocks.5.img_mod.1.weight": torch.empty(1),
        }

        cfg = _build_qwen_image_transformer_config(sd, is_edit=False)

        assert cfg["num_layers"] == 6  # max block index (5) + 1
        assert cfg["in_channels"] == 64
        assert cfg["num_attention_heads"] == 24  # 3072 // 128
        assert cfg["joint_attention_dim"] == 3584

    def test_empty_state_dict_falls_back_to_defaults(self):
        cfg = _build_qwen_image_transformer_config({}, is_edit=False)
        assert cfg["num_layers"] == 60
