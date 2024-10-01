import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.layers import GGUFTensor


def test_ggml_tensor():
    """Smoke test that multiplication works on a GGMLTensor."""
    weight: GGUFTensor = torch.load("tests/assets/gguf_qweight.pt")
    tensor_shape = weight.tensor_shape
    tensor_type = weight.tensor_type
    data = torch.Tensor(weight.data)

    ggml_tensor = GGMLTensor(data, tensor_type, tensor_shape)
    ones = torch.ones([1], dtype=torch.float32)

    _ = ggml_tensor * ones
