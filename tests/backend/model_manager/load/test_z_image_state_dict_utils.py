"""Unit tests for the Z-Image GGUF/ComfyUI -> diffusers state-dict converter."""

import pytest
import torch

from invokeai.backend.model_manager.load.model_loaders.z_image import (
    _convert_z_image_gguf_to_diffusers,
    _remap_z_image_layer_paths,
)
from tests.backend.model_manager.load.state_dicts.utils import keys_to_mock_state_dict
from tests.backend.model_manager.load.state_dicts.z_image_transformer_comfyui_keys import (
    state_dict_keys as z_image_keys,
)


class TestConvertZImageGgufToDiffusers:
    def test_fused_qkv_split(self):
        sd = keys_to_mock_state_dict(z_image_keys)
        n_qkv = sum(1 for k in sd if k.endswith(".attention.qkv.weight"))
        assert n_qkv > 0

        out = _convert_z_image_gguf_to_diffusers(sd)

        # Each fused qkv weight becomes three separate projections.
        assert sum(1 for k in out if k.endswith(".attention.to_q.weight")) == n_qkv
        assert sum(1 for k in out if k.endswith(".attention.to_k.weight")) == n_qkv
        assert sum(1 for k in out if k.endswith(".attention.to_v.weight")) == n_qkv
        assert not any(".attention.qkv." in k for k in out)

    def test_key_renames(self):
        out = _convert_z_image_gguf_to_diffusers(keys_to_mock_state_dict(z_image_keys))
        # q_norm/k_norm -> norm_q/norm_k, attention.out -> attention.to_out.0
        assert any(k.endswith(".attention.norm_q.weight") for k in out)
        assert any(k.endswith(".attention.norm_k.weight") for k in out)
        assert any(k.endswith(".attention.to_out.0.weight") for k in out)
        assert not any(".q_norm." in k or ".k_norm." in k for k in out)
        assert not any(".attention.out." in k for k in out)

    def test_embedder_and_final_layer_renamed(self):
        out = _convert_z_image_gguf_to_diffusers(keys_to_mock_state_dict(z_image_keys))
        assert any(k.startswith("all_x_embedder.2-1.") for k in out)
        assert any(k.startswith("all_final_layer.2-1.") for k in out)
        assert not any(k.startswith("x_embedder.") or k.startswith("final_layer.") for k in out)

    def test_norm_final_is_dropped(self):
        # The diffusers model uses a non-learnable final LayerNorm, so norm_final.* is skipped.
        assert any(k.startswith("norm_final.") for k in z_image_keys)
        out = _convert_z_image_gguf_to_diffusers(keys_to_mock_state_dict(z_image_keys))
        assert not any(k.startswith("norm_final.") for k in out)

    def test_pad_tokens_are_2d_after_conversion(self):
        # The diffusers model expects a leading batch dim on the pad tokens. The checkpoint
        # already stores them 2D; GGUF ships them 1D (see the reshape test below).
        out = _convert_z_image_gguf_to_diffusers(keys_to_mock_state_dict(z_image_keys))
        for pad in ("x_pad_token", "cap_pad_token"):
            assert out[pad].dim() == 2
            assert out[pad].shape[0] == 1

    def test_1d_pad_token_gains_batch_dim(self):
        # GGUF stores pad tokens as [dim]; they must be reshaped to [1, dim].
        out = _convert_z_image_gguf_to_diffusers({"x_pad_token": torch.arange(4.0)})
        assert out["x_pad_token"].shape == (1, 4)

    def test_qkv_split_preserves_values(self):
        # A [6, 2] fused qkv splits into three [2, 2] chunks in order q, k, v.
        qkv = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        out = _convert_z_image_gguf_to_diffusers({"blk.attention.qkv.weight": qkv})
        assert torch.allclose(out["blk.attention.to_q.weight"], qkv[0:2])
        assert torch.allclose(out["blk.attention.to_k.weight"], qkv[2:4])
        assert torch.allclose(out["blk.attention.to_v.weight"], qkv[4:6])


class TestQkvQuantizationSideChannel:
    """A scaled-fp8 checkpoint puts a `scale_weight` next to the fused `qkv.weight`.

    Left on `...attention.qkv`, the recovered scale is keyed on a module path the diffusers model
    does not have, so `attach_fp8_scales` finds nothing and the three split weights stay quantized
    but *unscaled* — off by 1/weight_scale, with no error anywhere.
    """

    def test_per_tensor_scale_reaches_all_three_projections(self):
        out = _convert_z_image_gguf_to_diffusers(
            {
                "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                "blk.attention.qkv.scale_weight": torch.tensor(0.25),
            }
        )
        assert not any(".attention.qkv." in k for k in out)
        for name in ("to_q", "to_k", "to_v"):
            assert torch.equal(out[f"blk.attention.{name}.scale_weight"], torch.tensor(0.25))

    def test_per_channel_scale_is_split_like_the_weight(self):
        out = _convert_z_image_gguf_to_diffusers(
            {
                "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                "blk.attention.qkv.weight_scale": torch.arange(6, dtype=torch.float32),
            }
        )
        assert torch.equal(out["blk.attention.to_q.weight_scale"], torch.tensor([0.0, 1.0]))
        assert torch.equal(out["blk.attention.to_k.weight_scale"], torch.tensor([2.0, 3.0]))
        assert torch.equal(out["blk.attention.to_v.weight_scale"], torch.tensor([4.0, 5.0]))

    def test_marker_blob_is_copied_not_split(self):
        # `.comfy_quant` is a 1-D JSON byte string describing the layer, not a per-channel vector.
        blob = torch.frombuffer(b'{"format":"float8_e4m3fn"}', dtype=torch.uint8).clone()
        out = _convert_z_image_gguf_to_diffusers(
            {
                "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                "blk.attention.qkv.comfy_quant": blob,
            }
        )
        for name in ("to_q", "to_k", "to_v"):
            assert torch.equal(out[f"blk.attention.{name}.comfy_quant"], blob)

    def test_unknown_suffix_is_left_alone(self):
        out = _convert_z_image_gguf_to_diffusers(
            {
                "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                "blk.attention.qkv.something_else": torch.tensor(1.0),
            }
        )
        assert "blk.attention.qkv.something_else" in out

    def test_undivisible_scale_is_rejected_rather_than_mis_split(self):
        with pytest.raises(ValueError, match="Cannot split fused QKV quantization data"):
            _convert_z_image_gguf_to_diffusers(
                {
                    "blk.attention.qkv.weight": torch.arange(12, dtype=torch.float32).reshape(6, 2),
                    "blk.attention.qkv.weight_scale": torch.arange(4, dtype=torch.float32),
                }
            )


class TestMetadataPathRemap:
    """`_quantization_metadata` names layers in the checkpoint's scheme; the scales are recovered
    after the rename, so the per-layer hints have to follow the same route."""

    def test_renamed_layers_map_one_to_one(self):
        mapping = _remap_z_image_layer_paths(["x_embedder", "final_layer.linear", "layers.0.attention.out"])
        assert mapping["x_embedder"] == ["all_x_embedder.2-1"]
        assert mapping["final_layer.linear"] == ["all_final_layer.2-1.linear"]
        assert mapping["layers.0.attention.out"] == ["layers.0.attention.to_out.0"]

    def test_fused_qkv_maps_to_all_three_projections(self):
        mapping = _remap_z_image_layer_paths(["layers.0.attention.qkv"])
        assert mapping["layers.0.attention.qkv"] == [
            "layers.0.attention.to_q",
            "layers.0.attention.to_k",
            "layers.0.attention.to_v",
        ]
