"""Helpers for models loaded with FP8 layerwise-casting storage.

See `ModelLoader._apply_fp8_layerwise_casting`: eligible layers keep their weights in
`float8_e4m3fn` and forward hooks cast them up to the model's compute dtype (fp16/bf16) for the
duration of each forward pass.

The consequence for callers is that `model.dtype` — which diffusers derives from the first
parameter — reports `float8_e4m3fn`. That is a *storage* dtype: torch has no CUDA kernels for
arithmetic on it (`"pow_cuda" not implemented for 'Float8_e4m3fn'`). So any code that reads a
model's dtype in order to build or cast *tensors* (latents, noise, conditioning, control images,
LoRA patch weights) must use `get_model_compute_dtype()` instead of `model.dtype`.
"""

import torch

from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

# Storage-only float8 dtypes. Weights may be held in these, but no math may be done in them.
FP8_STORAGE_DTYPES: tuple[torch.dtype, ...] = (torch.float8_e4m3fn, torch.float8_e5m2)

# Attribute set on a model by the loader when FP8 layerwise casting is applied. It lives in the
# module's `__dict__` (torch.dtype is not a Parameter/Module), so it survives the deepcopy of the
# meta shell in the shared-CPU-weights adoption path.
FP8_COMPUTE_DTYPE_ATTR = "_invokeai_fp8_compute_dtype"


def set_fp8_compute_dtype(model: torch.nn.Module, compute_dtype: torch.dtype) -> None:
    """Record the dtype that `model`'s fp8-cast layers compute in."""
    if compute_dtype in FP8_STORAGE_DTYPES:
        # A float8 compute dtype is never valid, and recording one would silently reintroduce the
        # very crash this module exists to prevent: `get_model_compute_dtype` trusts the marker, so
        # every downstream tensor would be built in a dtype torch has no arithmetic kernels for.
        # The realistic way to get here is deriving the compute dtype from a model that is already
        # cast (i.e. casting twice) — fail loudly at the source instead.
        raise ValueError(
            f"Refusing to record {compute_dtype} as an FP8 compute dtype; it is a storage-only dtype. "
            "This usually means the compute dtype was derived from an already-fp8-cast model."
        )
    setattr(model, FP8_COMPUTE_DTYPE_ATTR, compute_dtype)


def get_model_compute_dtype(model: torch.nn.Module) -> torch.dtype:
    """Return the dtype that `model` actually computes in.

    Equivalent to `model.dtype` for normally-loaded models. For models with FP8 storage it returns
    the compute dtype (fp16/bf16) rather than the float8 storage dtype.
    """
    marked = getattr(model, FP8_COMPUTE_DTYPE_ATTR, None)
    if isinstance(marked, torch.dtype):
        return marked

    dtype = getattr(model, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        dtype = next((p.dtype for p in model.parameters()), None)
    if isinstance(dtype, torch.dtype) and dtype not in FP8_STORAGE_DTYPES:
        return dtype

    # Marker missing but the model is in fp8 storage. The cast skips precision-sensitive layers
    # (norms, position/patch embeddings), so the first non-fp8 float param reveals the compute
    # dtype. Fall back to the global torch dtype if every param happens to be fp8.
    for param in model.parameters():
        if param.is_floating_point() and param.dtype not in FP8_STORAGE_DTYPES:
            return param.dtype

    fallback = TorchDevice.choose_torch_dtype()
    logger.warning(
        f"{type(model).__name__} is in FP8 storage but carries no compute-dtype marker and has no non-fp8 float "
        f"parameter to infer one from; falling back to {fallback}. If the model computes in a different dtype, "
        "expect a dtype mismatch during the forward pass."
    )
    return fallback
