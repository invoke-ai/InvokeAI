"""Runtime support for Comfy "int8_tensorwise + convrot" quantized linears.

The Comfy-Org single-file H3 transformers store their four big per-block linears
(qkv/out/fc1/fc2) as symmetric per-output-channel int8:

- ``<layer>.weight``: int8 ``[out, in]``
- ``<layer>.weight_scale``: float32 ``[out, 1]``
- ``<layer>.comfy_quant``: a uint8 JSON marker, e.g.
  ``{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}``

With ``convrot``, the weights were rotated along the input dimension before
quantization: ``W_rot = grouped_256(W) @ H^T`` where ``H`` is the normalized
REGULAR Hadamard matrix built from the 4x4 seed ``[[1,1,1,-1],[1,1,-1,1],
[1,-1,1,1],[-1,1,1,1]]`` by Kronecker powers (sizes are powers of 4) and scaled
by ``1/sqrt(size)``. This ``H`` is symmetric and orthonormal (``H == H.T``,
``H @ H == I``), so recovering the un-rotated weight is the SAME grouped matmul:
``W = grouped_256(W_rot) @ H``. (Semantics verified against
Comfy-Org/comfy-quants and comfy-kitchen ``tensor/int8_utils.py`` — reimplemented
here, not copied.)

Comfy's W8A8 kernels instead rotate activations at runtime and run int8 GEMMs.
We target bf16 compute with int8 *storage*: ``Int8ConvrotLinear`` keeps the int8
weight and scale resident (4.6x smaller than bf16) and materializes the
dequantized, derotated bf16 weight per forward call. The derotation is a
``[out, in/256, 256] @ [256, 256]`` matmul — a rounding error next to the
transformer forward itself — and the transient bf16 weight (<= ~310 MB for the
largest layer) lives inside the denoise node's working-memory reservation.
"""

import json

import torch
import torch.nn.functional as F

CONVROT_GROUP_SIZE = 256

_HADAMARD_SEED = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))


def build_regular_hadamard(size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized regular Hadamard matrix of a power-of-4 size (CPU tensor)."""
    if size < 4 or (size & (size - 1)) != 0 or (size.bit_length() - 1) % 2 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(_HADAMARD_SEED, dtype=torch.float64)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return (h / (size**0.5)).to(dtype)


def parse_comfy_quant_marker(blob: torch.Tensor) -> dict:
    """Decode a ``<layer>.comfy_quant`` uint8 tensor into its JSON dict."""
    return json.loads(bytes(blob.cpu().numpy().tobytes()).decode("utf-8"))


def dequantize_convrot_weight(
    weight_q: torch.Tensor,
    weight_scale: torch.Tensor,
    convrot: bool,
    dtype: torch.dtype,
    group_size: int = CONVROT_GROUP_SIZE,
) -> torch.Tensor:
    """Recover the bf16/fp16 weight from int8 storage (and undo convrot if applied)."""
    w = weight_q.to(torch.float32) * weight_scale.to(torch.float32)
    if convrot:
        out_features, in_features = w.shape
        if in_features % group_size != 0:
            raise ValueError(f"convrot weight in_features {in_features} not divisible by {group_size}")
        h = build_regular_hadamard(group_size).to(device=w.device)
        w = (w.view(out_features, in_features // group_size, group_size) @ h).view(out_features, in_features)
    return w.to(dtype)


class Int8ConvrotLinear(torch.nn.Module):
    """A linear layer storing Comfy int8_tensorwise(+convrot) weights, dequantized per forward.

    The int8 weight and fp32 scale are registered as PERSISTENT buffers named ``weight`` and
    ``weight_scale`` — exactly the converted checkpoint's key names — so ``load_state_dict``
    consumes the quantized tensors directly and the model cache moves them between devices
    like any other weight. The Hadamard matrix is computed, not loaded (non-persistent).

    The model cache wraps this module as ``CustomInt8ConvrotLinear`` (see
    ``AUTOCAST_MODULE_TYPE_MAPPING``), which enables sidecar LoRA patches and lets a partial
    load leave some int8 buffers on the CPU — ``forward``'s per-call ``.to(device)`` then
    streams them (at half the bf16 byte count) instead of failing outright. Fully-resident
    operation (~20 GiB free VRAM for the pruned transformer) remains the intended regime;
    streamed layers pay a per-forward PCIe cost, and the diffusers-folder bf16 model is still
    the better citizen on small cards.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        convrot: bool,
        bias: torch.Tensor | None = None,
        group_size: int = CONVROT_GROUP_SIZE,
    ) -> None:
        super().__init__()
        if weight.dtype != torch.int8:
            raise ValueError(f"expected int8 weight, got {weight.dtype}")
        self.out_features, self.in_features = weight.shape
        self.convrot = convrot
        self.group_size = group_size
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        if convrot:
            if self.in_features % group_size != 0:
                raise ValueError(f"convrot weight in_features {self.in_features} not divisible by {group_size}")
            self.register_buffer("hadamard", build_regular_hadamard(group_size), persistent=False)
        else:
            self.hadamard = None
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def _dequantized_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # `.to(device)` is a no-op in the intended fully-resident regime; under partial load
        # the model cache may leave these buffers on the CPU, in which case this call streams
        # the int8 weight to the compute device per forward (see the class docstring).
        #
        # Dequant + derotation run directly in the compute dtype: int8 values are exact in
        # bf16/fp16, the scale multiply adds ~0.2% relative rounding vs the ~0.4-0.8% int8
        # quantization floor, and the matmul accumulates fp32 internally. This keeps the
        # per-call transient at ~two weight-sized tensors (~620 MB peak for the fused-SwiGLU
        # fc1 in bf16) instead of tripling through an fp32 intermediate (~1.5 GiB). An fp32
        # compute dtype still gets the exact fp32 path for free.
        w = self.weight.to(device=device, dtype=dtype) * self.weight_scale.to(device=device, dtype=dtype)
        if self.convrot:
            assert self.hadamard is not None
            w = (
                w.view(self.out_features, self.in_features // self.group_size, self.group_size)
                @ self.hadamard.to(device=device, dtype=dtype)
            ).view(self.out_features, self.in_features)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantized_weight(x.device, x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, convrot={self.convrot}"
