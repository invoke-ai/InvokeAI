"""Tests for the Comfy int8_tensorwise(+convrot) dequantization runtime.

The Hadamard/rotation semantics were verified against Comfy-Org/comfy-quants and
comfy-kitchen: ``W_rot = grouped(W) @ H^T`` with the normalized regular Hadamard.
These tests quantize a known weight the same way and pin that our dequantization
recovers it, so the loader can never silently mis-rotate.
"""

import torch

from invokeai.backend.minimax_h3.int8_convrot import (
    CONVROT_GROUP_SIZE,
    Int8ConvrotLinear,
    build_regular_hadamard,
    dequantize_convrot_weight,
    parse_comfy_quant_marker,
)


def _quantize_reference(w: torch.Tensor, convrot: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference quantizer mirroring comfy-quants: optional grouped rotation by H^T,
    then symmetric per-output-channel int8."""
    if convrot:
        out_f, in_f = w.shape
        h = build_regular_hadamard(CONVROT_GROUP_SIZE, dtype=w.dtype)
        w = (w.view(out_f, in_f // CONVROT_GROUP_SIZE, CONVROT_GROUP_SIZE) @ h.T).view(out_f, in_f)
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def test_regular_hadamard_is_symmetric_orthonormal() -> None:
    h = build_regular_hadamard(CONVROT_GROUP_SIZE)
    assert torch.equal(h, h.T)
    eye = h @ h
    assert torch.allclose(eye, torch.eye(CONVROT_GROUP_SIZE), atol=1e-5)


def test_dequantize_recovers_unrotated_weight() -> None:
    """Quantize with rotation, dequantize with derotation: the result must match the
    original weight to within int8 quantization error (and be much closer than the
    still-rotated weight is)."""
    torch.manual_seed(0)
    w = torch.randn(64, 2 * CONVROT_GROUP_SIZE)
    q, scale = _quantize_reference(w, convrot=True)

    recovered = dequantize_convrot_weight(q, scale, convrot=True, dtype=torch.float32)
    quant_err = (recovered - w).abs().max().item()
    # Per-element error is scale/2 in the ROTATED domain; derotation is orthonormal so the
    # l2 norm carries over exactly, but the max-abs bound is only statistical (each element
    # mixes 256 errors at +-1/16) - allow 2x the per-row scale.
    assert quant_err < 2 * scale.max().item()

    still_rotated = q.float() * scale
    assert (still_rotated - w).abs().max() > 10 * quant_err


def test_dequantize_without_convrot() -> None:
    torch.manual_seed(1)
    w = torch.randn(32, CONVROT_GROUP_SIZE)
    q, scale = _quantize_reference(w, convrot=False)
    recovered = dequantize_convrot_weight(q, scale, convrot=False, dtype=torch.float32)
    assert (recovered - w).abs().max().item() < scale.max().item()


def test_int8_convrot_linear_matches_dequantized_f_linear() -> None:
    torch.manual_seed(2)
    w = torch.randn(48, CONVROT_GROUP_SIZE)
    q, scale = _quantize_reference(w, convrot=True)
    lin = Int8ConvrotLinear(q, scale, convrot=True)

    x = torch.randn(5, CONVROT_GROUP_SIZE)
    ref = torch.nn.functional.linear(x, dequantize_convrot_weight(q, scale, convrot=True, dtype=torch.float32))
    got = lin(x)
    assert torch.allclose(ref, got, atol=1e-5)

    # End-to-end sanity: output approximates the full-precision linear.
    full = torch.nn.functional.linear(x, w)
    assert (got - full).abs().max().item() < 1.0


def test_int8_convrot_linear_state_dict_contract() -> None:
    """Persistent buffers must be named exactly `weight` / `weight_scale` (+ optional `bias`)
    so the converted checkpoint's keys load directly and strict load_state_dict holds; the
    Hadamard is computed, never loaded, but must still move with the module."""
    torch.manual_seed(3)
    w = torch.randn(16, CONVROT_GROUP_SIZE)
    q, scale = _quantize_reference(w, convrot=True)
    bias = torch.randn(16)
    lin = Int8ConvrotLinear(q, scale, convrot=True, bias=bias)
    assert set(lin.state_dict().keys()) == {"weight", "weight_scale", "bias"}
    assert {n for n, _ in lin.named_buffers()} == {"weight", "weight_scale", "hadamard", "bias"}
    assert lin.state_dict()["weight"].dtype == torch.int8


def test_parse_comfy_quant_marker() -> None:
    blob = torch.frombuffer(
        b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}', dtype=torch.uint8
    ).clone()
    marker = parse_comfy_quant_marker(blob)
    assert marker == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def test_int8_convrot_linear_bf16_path_tracks_fp32_reference() -> None:
    """Dequant + derotation run in the input dtype; on bf16 the scale-multiply rounding must
    stay within ~2% relative of the exact fp32 dequantization path."""
    torch.manual_seed(4)
    w = torch.randn(48, 2 * CONVROT_GROUP_SIZE)
    q, scale = _quantize_reference(w, convrot=True)
    lin = Int8ConvrotLinear(q, scale, convrot=True)

    x = torch.randn(5, 2 * CONVROT_GROUP_SIZE)
    ref = torch.nn.functional.linear(x, dequantize_convrot_weight(q, scale, convrot=True, dtype=torch.float32))
    got = lin(x.to(torch.bfloat16)).to(torch.float32)
    rel_err = (got - ref).abs().max() / ref.abs().max()
    assert rel_err < 0.02, f"bf16 dequant path diverges from fp32 reference: rel_err={rel_err:.4f}"
