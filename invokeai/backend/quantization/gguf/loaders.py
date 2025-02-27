from pathlib import Path

import gguf
import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES


def gguf_sd_loader(path: Path, compute_dtype: torch.dtype) -> dict[str, GGMLTensor]:
    reader = gguf.GGUFReader(path)

    sd: dict[str, GGMLTensor] = {}
    for tensor in reader.tensors:
        torch_tensor = torch.from_numpy(tensor.data)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
            torch_tensor = torch_tensor.view(*shape)
        sd[tensor.name] = GGMLTensor(
            torch_tensor, ggml_quantization_type=tensor.tensor_type, tensor_shape=shape, compute_dtype=compute_dtype
        )
    return sd
