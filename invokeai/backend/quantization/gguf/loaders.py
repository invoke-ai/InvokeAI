# Largely based on https://github.com/city96/ComfyUI-GGUF

from pathlib import Path

import gguf
import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.layers import GGUFTensor
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES

TORCH_COMPATIBLE_QTYPES = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}


def gguf_sd_loader(path: Path) -> dict[str, GGUFTensor]:
    reader = gguf.GGUFReader(path)

    sd: dict[str, GGUFTensor] = {}
    for tensor in reader.tensors:
        torch_tensor = torch.from_numpy(tensor.data)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
            torch_tensor = torch_tensor.view(*shape)
        sd[tensor.name] = GGMLTensor(torch_tensor, ggml_quantization_type=tensor.tensor_type, tensor_shape=shape)
    return sd


# def gguf_sd_loader(
#     path: Path, handle_prefix: str = "model.diffusion_model.", data_type: torch.dtype = torch.bfloat16
# ) -> dict[str, GGUFTensor]:
#     """
#     Read state dict as fake tensors
#     """
#     reader = gguf.GGUFReader(path)

#     prefix_len = len(handle_prefix)
#     tensor_names = {tensor.name for tensor in reader.tensors}
#     has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

#     tensors: list[tuple[str, gguf.ReaderTensor]] = []
#     for tensor in reader.tensors:
#         sd_key = tensor_name = tensor.name
#         if has_prefix:
#             if not tensor_name.startswith(handle_prefix):
#                 continue
#             sd_key = tensor_name[prefix_len:]
#         tensors.append((sd_key, tensor))

#     # detect and verify architecture
#     compat = None
#     arch_str = None
#     arch_field = reader.get_field("general.architecture")
#     if arch_field is not None:
#         if len(arch_field.types) != 1 or arch_field.types[0] != gguf.GGUFValueType.STRING:
#             raise TypeError(f"Bad type for GGUF general.architecture key: expected string, got {arch_field.types!r}")
#         arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")
#         if arch_str not in {"flux"}:
#             raise ValueError(f"Unexpected architecture type in GGUF file, expected flux, but got {arch_str!r}")
#     else:
#         arch_str = detect_arch({val[0] for val in tensors})
#         compat = "sd.cpp"

#     # main loading loop
#     state_dict: dict[str, GGUFTensor] = {}
#     qtype_dict: dict[str, int] = {}
#     for sd_key, tensor in tensors:
#         tensor_name = tensor.name
#         tensor_type_str = str(tensor.tensor_type)
#         torch_tensor = torch.from_numpy(tensor.data)  # mmap

#         shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
#         # Workaround for stable-diffusion.cpp SDXL detection.
#         if compat == "sd.cpp" and arch_str == "sdxl":
#             if tensor_name.endswith((".proj_in.weight", ".proj_out.weight")):
#                 while len(shape) > 2 and shape[-1] == 1:
#                     shape = shape[:-1]

#         # add to state dict
#         if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
#             torch_tensor = torch_tensor.view(*shape)
#         state_dict[sd_key] = GGUFTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
#         qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

#     return state_dict
