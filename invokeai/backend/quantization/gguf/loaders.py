import gc
from pathlib import Path

import gguf
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


def gguf_sd_loader(path: Path, compute_dtype: torch.dtype) -> dict[str, GGMLTensor]:
    with WrappedGGUFReader(path) as reader:
        sd: dict[str, GGMLTensor] = {}
        for tensor in reader.tensors:
            # Use .copy() to create a true copy of the data, not a view.
            # This is critical on Windows where the memory-mapped file cannot be deleted
            # while tensors still hold references to the mapped memory.
            torch_tensor = torch.from_numpy(tensor.data.copy())

            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
                torch_tensor = torch_tensor.view(*shape)
            sd[tensor.name] = GGMLTensor(
                torch_tensor,
                ggml_quantization_type=tensor.tensor_type,
                tensor_shape=shape,
                compute_dtype=compute_dtype,
            )
        return sd
