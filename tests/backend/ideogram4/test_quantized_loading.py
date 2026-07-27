"""Tests for the Ideogram 4 weight-only fp8 loading mechanism.

The Ideogram 4 fp8 text encoder is loaded by building the empty architecture, swapping its
quantized ``nn.Linear`` layers for ``Fp8Linear`` (gated on a saved per-row scale), then loading the
prequantized state dict with ``assign=True`` / ``strict=False`` — the exact pattern the model loader
uses in ``model_loaders/ideogram4.py::_load_text_encoder``. These tests exercise that mechanism on a
tiny CPU model so the fp8 path has regression coverage without a multi-GB checkpoint.
"""

import accelerate
import pytest
import torch
import torch.nn as nn

from invokeai.backend.ideogram4.quantized_loading import (
    FP8_TEXT_ENCODER_CONFIG_FLAG,
    Fp8Linear,
    is_fp8_state_dict,
    load_fp8_state_dict,
    quantize_weight_to_fp8,
    swap_linears_to_fp8,
)


class _TinyEncoder(nn.Module):
    """A stand-in for the text encoder: two Linears (fp8-quantized) around a non-quantized norm,
    plus a non-persistent buffer that mimics the rotary caches transformers models compute in
    ``__init__`` (and which must survive the meta-device build)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.norm = nn.LayerNorm(16)
        self.lin2 = nn.Linear(16, 4)
        self.register_buffer("rope_cache", torch.arange(4, dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.norm(self.lin1(x)))


def _make_fp8_state_dict(ref: _TinyEncoder, compute_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Build a prequantized state dict: the two Linears become fp8 weight + per-row scale, everything
    else stays a normal float tensor. Mirrors the on-disk fp8 checkpoint layout."""
    sd: dict[str, torch.Tensor] = {}
    for name in ("lin1", "lin2"):
        lin: nn.Linear = getattr(ref, name)
        q, scale = quantize_weight_to_fp8(lin.weight)
        sd[f"{name}.weight"] = q
        sd[f"{name}.weight_scale"] = scale
        sd[f"{name}.bias"] = lin.bias.detach().to(compute_dtype)
    sd["norm.weight"] = ref.norm.weight.detach().to(compute_dtype)
    sd["norm.bias"] = ref.norm.bias.detach().to(compute_dtype)
    return sd


def _dequant_reference(ref: _TinyEncoder, sd: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Forward pass using the dequantized fp8 weights — the exact math ``Fp8Linear.forward`` runs, so
    the loaded model must match this to within dtype rounding (not the lossy original weights)."""
    dtype = x.dtype

    def deq(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        w = sd[f"{name}.weight"].to(dtype) * sd[f"{name}.weight_scale"].to(dtype).unsqueeze(1)
        return w, sd[f"{name}.bias"].to(dtype)

    w1, b1 = deq("lin1")
    w2, b2 = deq("lin2")
    h = torch.nn.functional.linear(x, w1, b1)
    h = torch.nn.functional.layer_norm(h, (16,), sd["norm.weight"].to(dtype), sd["norm.bias"].to(dtype))
    return torch.nn.functional.linear(h, w2, b2)


def test_fp8_load_matches_loader_pattern() -> None:
    """Build empty -> swap -> load(assign, strict=False), exactly as the ideogram4 loader does, and
    verify no meta tensors survive and the forward matches the dequantized reference."""
    torch.manual_seed(0)
    compute_dtype = torch.float32

    ref = _TinyEncoder().to(compute_dtype).eval()
    sd = _make_fp8_state_dict(ref, compute_dtype)

    assert is_fp8_state_dict(sd)

    with accelerate.init_empty_weights():
        model = _TinyEncoder()
        swap_linears_to_fp8(model, sd, compute_dtype=compute_dtype)

    # Only the two Linears carry a saved scale, so exactly two get swapped.
    assert sum(1 for m in model.modules() if isinstance(m, Fp8Linear)) == 2

    load_fp8_state_dict(model, sd, device=torch.device("cpu"), dtype=compute_dtype, assign=True, strict=False)
    model.eval()

    assert not any(p.is_meta for p in model.parameters()), "meta params remained after fp8 load"
    assert not any(b.is_meta for b in model.buffers()), "meta buffers remained after fp8 load"
    # The non-persistent rope buffer must have been rebuilt with real data by the meta-device init.
    assert torch.equal(model.rope_cache, torch.arange(4, dtype=torch.float32))

    x = torch.randn(2, 8, dtype=compute_dtype)
    with torch.no_grad():
        out = model(x)
        expected = _dequant_reference(ref, sd, x)
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4)


def test_fp8_load_rejects_unexpected_keys() -> None:
    """A key the model has no home for must fail loudly rather than load silently."""
    torch.manual_seed(1)
    compute_dtype = torch.float32
    ref = _TinyEncoder().to(compute_dtype).eval()
    sd = _make_fp8_state_dict(ref, compute_dtype)
    sd["lin1.bogus_extra"] = torch.zeros(3)

    model = _TinyEncoder().to(compute_dtype)
    swap_linears_to_fp8(model, sd, compute_dtype=compute_dtype)
    with pytest.raises(RuntimeError, match="unexpected keys"):
        load_fp8_state_dict(model, sd, device=torch.device("cpu"), dtype=compute_dtype, strict=False)


def test_fp8_missing_key_strictness() -> None:
    """strict=True raises on a missing weight; strict=False downgrades it to a warning."""
    torch.manual_seed(2)
    compute_dtype = torch.float32
    ref = _TinyEncoder().to(compute_dtype).eval()
    sd = _make_fp8_state_dict(ref, compute_dtype)
    del sd["norm.bias"]

    def build() -> _TinyEncoder:
        m = _TinyEncoder().to(compute_dtype)
        swap_linears_to_fp8(m, sd, compute_dtype=compute_dtype)
        return m

    with pytest.raises(RuntimeError, match="missing keys"):
        load_fp8_state_dict(build(), sd, device=torch.device("cpu"), dtype=compute_dtype, strict=True)

    with pytest.warns(UserWarning, match="missing keys"):
        load_fp8_state_dict(build(), sd, device=torch.device("cpu"), dtype=compute_dtype, strict=False)


def test_fp8_config_flag_constant() -> None:
    """The loader keys the fp8 path off this exact config.json marker; pin it so a rename can't
    silently reintroduce the 'importable but fails at encode time' bug."""
    assert FP8_TEXT_ENCODER_CONFIG_FLAG == "ideogram_fp8_weight_only"
