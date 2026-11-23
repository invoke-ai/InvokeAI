"""Unit tests for the pure state-dict helpers in the Qwen-Image / Qwen-VL loader.

These freeze the checkpoint key-surgery that the loaders perform before instantiating a model,
so a regression like the transformers-5.x one (where `_checkpoint_conversion_mapping` became
`{}` and the `visual.* -> model.visual.*` remap was silently skipped) fails here instead of at
the user's first load. The mapping now lives in the loader, since transformers 5.10 removed it.
"""

import torch

from invokeai.backend.model_manager.load.model_loaders.qwen_image import (
    _build_qwen_image_transformer_config,
    _remap_qwen_vl_checkpoint_keys,
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
        # `scaled_fp8` is the file's quantization marker, which the fp8 extraction removes; it names no module.
        sd = {key: value for key, value in keys_to_mock_state_dict(qwen_vl_keys).items() if key != "scaled_fp8"}

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


class TestRemapKeepsScalesBesideTheirWeights:
    def test_every_scale_of_the_captured_fp8_scaled_file_pairs_with_a_weight_after_the_remap(self):
        """The loader extracts scales after the remap, so each scale must travel with its weight."""
        sd = keys_to_mock_state_dict(qwen_vl_keys)
        scales = [k for k in sd if k.endswith((".scale_weight", ".scale_input"))]
        assert scales, "the captured checkpoint is fp8_scaled, so it ships scales"

        remapped = _remap_qwen_vl_checkpoint_keys(sd)

        for key in scales:
            moved = _remap_qwen_vl_checkpoint_keys({key: None})
            (new_key,) = moved
            module = new_key.rsplit(".", 1)[0]
            assert f"{module}.weight" in remapped, new_key


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
