"""Which tensors of a GGUF state dict stay packed, and turning the rest into plain tensors at load."""

from typing import Callable

import gguf
import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

# GGUF's unquantized storage types. `gguf_sd_loader` wraps BF16 like a quantized type (torch has no
# GGML-side view for it), so here it is as packed as Q4_0 -- but dequantizing it changes no byte count.
UNQUANTIZED_GGML_TYPES = frozenset(
    {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}
)


def dequantize_ggml_at_load(
    sd: dict[str, torch.Tensor],
    model: torch.nn.Module,
    skip_patterns: tuple[str, ...],
    reserve: Callable[[int], None],
) -> int:
    """Replace, in `sd`, the GGUF tensors that must not -- or need not -- stay packed. Returns the count.

    Must not: `GGMLTensor` only works where torch dispatches the op to it (a Linear's matmul, `mul`,
    `add`). Ideogram 4 shows the rest: `F.embedding` (`embed_image_indicator`) is not dispatched at all,
    `F.rms_norm` validates the packed buffer in C++, and `compute_dtype_of` reads the *storage* dtype --
    `uint8` -- off `input_proj` and `t_embedding`, then casts every activation to it; Qwen-Image-2.1's
    `txt_in.text_norm` reads `weight.float()`, a dtype change `GGMLTensor` refuses. So everything that is not a
    Linear weight or bias, plus the model's own skip patterns. Published GGUFs differ in which of these
    they quantize (some keep the norms in BF16, others pack them to Q4_K), hence a rule by module.

    Need not: an unquantized tensor is the same size either way, but kept packed it is dequantized on
    every forward -- for the BF16 `llm_cond_proj` most releases ship, a 245M-element weight, that is a
    measured 1.87 GB transient per call.

    The reservation is absolute, because `make_room` makes that much room rather than adding to the
    file-size reservation the framework made before the loader ran: everything that stays packed at
    its packed size, everything replaced at its unpacked size, and one replacement in flight. BF16 is
    reinterpreted rather than run through the GGML kernel, which widens through int32 and float32 on
    the way -- about 8 bytes per element, 2 GB for `llm_cond_proj` alone.
    """
    linear_params = {
        f"{module_name}.{param_name}"
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        for param_name, _ in module.named_parameters(recurse=False)
    }
    keys = [
        key
        for key, value in sd.items()
        if isinstance(value, GGMLTensor)
        and (
            value._ggml_quantization_type in UNQUANTIZED_GGML_TYPES
            or key not in linear_params
            or any(pattern in key for pattern in skip_patterns)
        )
    ]
    unpacked = {key: sd[key].tensor_shape.numel() * sd[key].compute_dtype.itemsize for key in keys}
    kept = sum(
        value.quantized_data.nbytes if isinstance(value, GGMLTensor) else value.nbytes
        for key, value in sd.items()
        if key not in unpacked
    )
    reserve(kept + sum(unpacked.values()) + max(unpacked.values(), default=0))
    # One at a time, so each packed original is released as its replacement lands.
    for key in keys:
        sd[key] = _unpack(sd[key])
    return len(keys)


def _unpack(value: GGMLTensor) -> torch.Tensor:
    """`value` as a plain tensor in its compute dtype; BF16 as a view of its bytes, not a decode."""
    if value._ggml_quantization_type is gguf.GGMLQuantizationType.BF16:
        return value.quantized_data.view(torch.bfloat16).reshape(value.tensor_shape).to(value.compute_dtype)
    return value.get_dequantized_tensor()
