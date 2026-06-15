"""Tests for Wan loader helpers (native -> diffusers key conversion)."""

import gguf
import torch

from invokeai.backend.model_manager.load.model_loaders.wan import (
    _convert_wan_native_to_diffusers,
    _unwrap_unquantized_to_compute_dtype,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


def test_converts_text_and_time_embedders():
    sd = {
        "text_embedding.0.weight": "a",
        "text_embedding.0.bias": "b",
        "text_embedding.2.weight": "c",
        "time_embedding.0.weight": "d",
        "time_embedding.2.weight": "e",
        "time_projection.1.weight": "f",
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert "condition_embedder.text_embedder.linear_1.weight" in out
    assert "condition_embedder.text_embedder.linear_1.bias" in out
    assert "condition_embedder.text_embedder.linear_2.weight" in out
    assert "condition_embedder.time_embedder.linear_1.weight" in out
    assert "condition_embedder.time_embedder.linear_2.weight" in out
    assert "condition_embedder.time_proj.weight" in out


def test_converts_attention_blocks():
    sd = {
        "blocks.0.self_attn.q.weight": 1,
        "blocks.0.self_attn.k.weight": 2,
        "blocks.0.self_attn.v.weight": 3,
        "blocks.0.self_attn.o.weight": 4,
        "blocks.0.self_attn.norm_q.weight": 5,
        "blocks.0.self_attn.norm_k.weight": 6,
        "blocks.0.cross_attn.q.weight": 7,
        "blocks.0.cross_attn.k.weight": 8,
        "blocks.0.cross_attn.v.weight": 9,
        "blocks.0.cross_attn.o.weight": 10,
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert "blocks.0.attn1.to_q.weight" in out
    assert "blocks.0.attn1.to_k.weight" in out
    assert "blocks.0.attn1.to_v.weight" in out
    assert "blocks.0.attn1.to_out.0.weight" in out
    assert "blocks.0.attn1.norm_q.weight" in out
    assert "blocks.0.attn1.norm_k.weight" in out
    assert "blocks.0.attn2.to_q.weight" in out
    assert "blocks.0.attn2.to_out.0.weight" in out


def test_converts_ffn_and_modulation():
    sd = {
        "blocks.0.ffn.0.weight": 1,
        "blocks.0.ffn.0.bias": 2,
        "blocks.0.ffn.2.weight": 3,
        "blocks.0.modulation": 4,
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert "blocks.0.ffn.net.0.proj.weight" in out
    assert "blocks.0.ffn.net.0.proj.bias" in out
    assert "blocks.0.ffn.net.2.weight" in out
    assert "blocks.0.scale_shift_table" in out


def test_swaps_norm2_and_norm3():
    """Native norm3 has params (cross-attn norm in diffusers norm2 slot)
    while native norm2 is the elementwise-affine-False norm. The swap
    via placeholder must not collide."""
    sd = {
        "blocks.0.norm2.weight": "native_norm2",
        "blocks.0.norm3.weight": "native_norm3",
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert out["blocks.0.norm3.weight"] == "native_norm2"
    assert out["blocks.0.norm2.weight"] == "native_norm3"


def test_converts_head_keys():
    sd = {
        "head.head.weight": 1,
        "head.head.bias": 2,
        "head.modulation": 3,
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert "proj_out.weight" in out
    assert "proj_out.bias" in out
    assert "scale_shift_table" in out


def test_diffusers_keys_pass_through_unchanged():
    """If a state dict is already in diffusers form, the substring rules
    must be no-ops — none of the native fingerprints are present."""
    sd = {
        "patch_embedding.weight": 1,
        "condition_embedder.text_embedder.linear_1.weight": 2,
        "blocks.0.attn1.to_q.weight": 3,
        "blocks.0.ffn.net.0.proj.weight": 4,
        "scale_shift_table": 5,
        "proj_out.weight": 6,
    }
    out = _convert_wan_native_to_diffusers(sd)
    assert set(out.keys()) == set(sd.keys())
    assert all(out[k] == sd[k] for k in sd)


def test_does_not_mutate_input():
    sd = {"text_embedding.0.weight": 1}
    snapshot = dict(sd)
    _convert_wan_native_to_diffusers(sd)
    assert sd == snapshot


def test_non_string_keys_pass_through():
    sd = {0: "ignored", "text_embedding.0.weight": "renamed"}
    out = _convert_wan_native_to_diffusers(sd)
    assert out[0] == "ignored"
    assert "condition_embedder.text_embedder.linear_1.weight" in out


def _ggml(data: torch.Tensor, qtype: gguf.GGMLQuantizationType, compute_dtype: torch.dtype) -> GGMLTensor:
    return GGMLTensor(
        data=data,
        ggml_quantization_type=qtype,
        tensor_shape=data.shape,
        compute_dtype=compute_dtype,
    )


class TestUnwrapUnquantized:
    """The QuantStack GGUFs store ``patch_embedding.bias`` as F16 while latents
    flow through the model as bf16. Conv3d isn't in GGMLTensor's dispatch table,
    so without unwrapping the F16 wrapper goes into conv3d as-is and crashes
    with ``Input type (c10::BFloat16) and bias type (c10::Half) should be the same``.
    These tests guard the unwrap step that prevents that."""

    def test_f16_compatible_qtype_is_unwrapped_and_cast(self):
        # F16 storage that should become bf16 plain tensor.
        f16_data = torch.zeros((4,), dtype=torch.float16)
        sd = {"bias": _ggml(f16_data, gguf.GGMLQuantizationType.F16, torch.bfloat16)}
        out = _unwrap_unquantized_to_compute_dtype(sd)

        result = out["bias"]
        assert not isinstance(result, GGMLTensor)
        assert result.dtype == torch.bfloat16

    def test_f32_compatible_qtype_is_unwrapped_and_cast(self):
        # patch_embedding.weight in QuantStack is F32 — same path.
        f32_data = torch.zeros((4,), dtype=torch.float32)
        sd = {"weight": _ggml(f32_data, gguf.GGMLQuantizationType.F32, torch.bfloat16)}
        out = _unwrap_unquantized_to_compute_dtype(sd)

        result = out["weight"]
        assert not isinstance(result, GGMLTensor)
        assert result.dtype == torch.bfloat16

    def test_quantized_tensor_stays_wrapped(self):
        # Q4_K and friends must remain GGMLTensor so on-demand dequant works
        # via the linear/addmm dispatch path. The byte storage shape is fake
        # but irrelevant for this test.
        q4_data = torch.zeros((1,), dtype=torch.uint8)
        sd = {"linear.weight": _ggml(q4_data, gguf.GGMLQuantizationType.Q4_K, torch.bfloat16)}
        out = _unwrap_unquantized_to_compute_dtype(sd)

        assert isinstance(out["linear.weight"], GGMLTensor)
        assert out["linear.weight"]._ggml_quantization_type == gguf.GGMLQuantizationType.Q4_K

    def test_plain_torch_tensor_passes_through(self):
        plain = torch.zeros((4,), dtype=torch.bfloat16)
        sd = {"plain": plain}
        out = _unwrap_unquantized_to_compute_dtype(sd)
        assert out["plain"] is plain
