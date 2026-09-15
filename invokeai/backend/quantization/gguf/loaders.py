import gc
import math
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()


class WrappedGGUFReader:
    """Wrapper around GGUFReader that adds a close() method."""

    def __init__(self, path: Path):
        self.reader = gguf.GGUFReader(path)

    def __enter__(self):
        return self.reader

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        """Explicitly close the memory-mapped file."""
        if hasattr(self.reader, "data"):
            try:
                self.reader.data.flush()
                del self.reader.data
            except (AttributeError, OSError, ValueError) as e:
                logger.warning(f"Wasn't able to close GGUF memory map: {e}")
        del self.reader
        gc.collect()


ORIG_SHAPE_KEY_PREFIX = "comfy.gguf.orig_shape."


def _coerce_dim(value: Any) -> int | None:
    """Coerce a single value from a GGUF metadata array to a positive dimension, or None if it isn't one.

    Metadata is untrusted input, so anything that is not a finite, integral, positive number is rejected rather
    than silently truncated (``int(2.5) == 2``) or allowed to raise (``int(float("inf"))``).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        dim = int(value)
    elif isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value) or not value.is_integer():
            return None
        dim = int(value)
    else:
        return None
    return dim if dim > 0 else None


def _read_comfy_orig_shapes(reader: gguf.GGUFReader) -> dict[str, torch.Size]:
    """Read ComfyUI's ``comfy.gguf.orig_shape.<tensor name>`` metadata.

    ComfyUI's GGUF converter can only quantize 2-D tensors, so it reshapes any tensor whose native
    rank/shape the quantizer rejects (e.g. Krea-2's ``first.weight`` of (6144, 64)) into a workable
    2-D shape and records the native shape under this key. Without honoring it, the tensor loads with
    the reshaped shape and ``load_state_dict`` fails with a size mismatch.
    """
    orig_shapes: dict[str, torch.Size] = {}
    for key, field in reader.fields.items():
        if not key.startswith(ORIG_SHAPE_KEY_PREFIX):
            continue
        tensor_name = key[len(ORIG_SHAPE_KEY_PREFIX) :]
        try:
            contents = field.contents()
        except Exception as e:
            logger.warning(f"Ignoring malformed GGUF metadata key {key!r}: {e}")
            continue
        if not isinstance(contents, (list, tuple)):
            logger.warning(f"Ignoring malformed GGUF metadata key {key!r}: expected an array, got {contents!r}")
            continue
        dims = tuple(_coerce_dim(v) for v in contents)
        if not dims or any(d is None for d in dims):
            logger.warning(f"Ignoring malformed GGUF metadata key {key!r}: {tuple(contents)!r}")
            continue
        orig_shapes[tensor_name] = torch.Size(dims)
    return orig_shapes


def gguf_sd_loader(path: Path, compute_dtype: torch.dtype) -> dict[str, GGMLTensor]:
    with WrappedGGUFReader(path) as reader:
        sd: dict[str, GGMLTensor] = {}
        orig_shapes = _read_comfy_orig_shapes(reader)
        for tensor in reader.tensors:
            # Use .copy() to create a true copy of the data, not a view.
            # This is critical on Windows where the memory-mapped file cannot be deleted
            # while tensors still hold references to the mapped memory.
            torch_tensor = torch.from_numpy(tensor.data.copy())

            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            orig_shape = orig_shapes.get(tensor.name)
            if orig_shape is not None:
                if orig_shape.numel() != shape.numel():
                    raise ValueError(
                        f"GGUF tensor {tensor.name!r} declares original shape {tuple(orig_shape)}, which has a "
                        f"different element count than its stored shape {tuple(shape)}."
                    )
                shape = orig_shape
            if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
                torch_tensor = torch_tensor.view(*shape)
            sd[tensor.name] = GGMLTensor(
                torch_tensor,
                ggml_quantization_type=tensor.tensor_type,
                tensor_shape=shape,
                compute_dtype=compute_dtype,
            )
        return sd
