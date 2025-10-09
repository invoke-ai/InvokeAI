import json
from pathlib import Path
from typing import Any, Optional

import gguf
import torch

from invokeai.backend.model_manager.model_on_disk import ModelOnDisk, StateDict
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


class StrippedModelOnDisk(ModelOnDisk):
    METADATA_KEY = "metadata_key_for_stripped_models"
    STR_TO_DTYPE = {str(dtype): dtype for dtype in torch.__dict__.values() if isinstance(dtype, torch.dtype)}

    def load_state_dict(self, path: Optional[Path] = None) -> StateDict:
        path = self.resolve_weight_file(path)
        return self.load_stripped_model(path)

    def metadata(self, path: Optional[Path] = None) -> dict[str, str]:
        path = self.resolve_weight_file(path)
        with open(path, "r") as f:
            contents = json.load(f)
        return contents.get(self.METADATA_KEY, {})

    @classmethod
    def strip(cls, v: Any):
        match v:
            case GGMLTensor():
                # GGMLTensor needs special handling to preserve quantization metadata. It is a subclass of torch.Tensor,
                # so we need to check for it before checking for torch.Tensor.
                return {
                    "quantized_data": cls.strip(v.quantized_data),
                    "ggml_quantization_type": v._ggml_quantization_type.name,
                    "tensor_shape": list(v.tensor_shape),
                    "compute_dtype": str(v.compute_dtype),
                    "fakeGGMLTensor": True,
                }
            case torch.Tensor():
                return {"shape": v.shape, "dtype": str(v.dtype), "fakeTensor": True}
            case dict():
                return {k: cls.strip(v) for k, v in v.items()}
            case list() | tuple():
                return [cls.strip(x) for x in v]
            case _:
                return v

    @classmethod
    def dress(cls, v: Any):
        match v:
            case {
                "quantized_data": quantized_data,
                "ggml_quantization_type": qtype_name,
                "tensor_shape": tensor_shape,
                "compute_dtype": compute_dtype_str,
                "fakeGGMLTensor": True,
            }:
                # Reconstruct the GGMLTensor from stripped data
                qtype = gguf.GGMLQuantizationType[qtype_name]
                compute_dtype = cls.STR_TO_DTYPE[compute_dtype_str]
                dressed_quantized_data = cls.dress(quantized_data)
                return GGMLTensor(
                    data=dressed_quantized_data,
                    ggml_quantization_type=qtype,
                    tensor_shape=torch.Size(tensor_shape),
                    compute_dtype=compute_dtype,
                )
            case {"shape": shape, "dtype": dtype_str, "fakeTensor": True}:
                dtype = cls.STR_TO_DTYPE[dtype_str]
                return torch.empty(shape, dtype=dtype)
            case dict():
                return {k: cls.dress(v) for k, v in v.items()}
            case list() | tuple():
                return [cls.dress(x) for x in v]
            case _:
                return v

    @classmethod
    def load_stripped_model(cls, path: Path, *args, **kwargs):
        with open(path, "r") as f:
            contents = json.load(f)
            contents.pop(cls.METADATA_KEY, None)
        return cls.dress(contents)
