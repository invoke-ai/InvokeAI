"""Weight-only fp8 Linears with a per-row scale, as InvokeAI's own Ideogram 4 fp8 builds store them.

Kept apart from ``backend/ideogram4`` because the model cache maps this module to its sidecar wrapper
(``AUTOCAST_MODULE_TYPE_MAPPING``) and must not import an architecture package to do so.
"""

import torch

from invokeai.backend.quantization.dequantizing_linear import DequantizingLinear

FP8_ROWWISE_WEIGHT_DTYPE = torch.float8_e4m3fn


class Fp8Linear(DequantizingLinear):
    """Linear layer holding an e4m3 float8 weight + per-row float32 scale, dequantized per forward.

    The weight and scale are registered as buffers (not parameters) so they load via ``load_state_dict``
    and are excluded from optimizer/grad machinery. The dequantized matmul runs in the input's dtype;
    ``compute_dtype`` is what the surrounding model casts its inputs to (see ``compute_dtype_of``).

    As a :class:`DequantizingLinear` it is wrapped by the model cache for sidecar LoRA patches. The wrapper keeps
    its weights required on the device (``CustomFp8Linear.streams_weights``), as they were before it was wrapped:
    the forward could stream buffers left on the CPU, but the cache does not leave them there.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool, compute_dtype: torch.dtype) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8_ROWWISE_WEIGHT_DTYPE))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def _dequantized_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # The cast always allocates (the weight is float8), so the scale can be applied in place: one weight-sized
        # tensor instead of two. Same rounding as `weight.to(dtype) * scale.to(dtype)`.
        weight = self.weight.to(device=device, dtype=dtype)
        return weight.mul_(self.weight_scale.to(device=device, dtype=dtype).unsqueeze(1))

    def dequant_transient_bytes(self, compute_dtype: torch.dtype) -> int:
        """One weight-sized tensor in the compute dtype (the scale is applied in place)."""
        return self.out_features * self.in_features * compute_dtype.itemsize
