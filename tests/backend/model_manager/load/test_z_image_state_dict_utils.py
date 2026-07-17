"""Unit tests for the Z-Image GGUF/ComfyUI -> diffusers state-dict converter."""

import torch

from invokeai.backend.model_manager.load.model_loaders.z_image import _convert_z_image_gguf_to_diffusers
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
