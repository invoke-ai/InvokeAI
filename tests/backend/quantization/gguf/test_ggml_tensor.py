import gguf
import pytest
import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.util.calc_tensor_size import calc_tensor_size


def quantize_tensor(data: torch.Tensor, ggml_quantization_type: gguf.GGMLQuantizationType) -> GGMLTensor:
    """Quantize a torch.Tensor to a GGMLTensor.

    Uses the gguf library's numpy implementation to quantize the tensor.
    """
    data_np = data.detach().cpu().numpy()
    quantized_np = gguf.quantize(data_np, ggml_quantization_type)
    return GGMLTensor(
        data=torch.from_numpy(quantized_np),
        ggml_quantization_type=ggml_quantization_type,
        tensor_shape=data.shape,
        compute_dtype=data.dtype,
    ).to(device=data.device)  # type: ignore


@pytest.mark.parametrize(
    ["device", "x1_quant_type", "x2_quant_type"],
    [
        # Test with no quantization.
        ("cpu", None, None),
        # Test with Q8_0 quantization.
        ("cpu", gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q8_0),
        ("cpu", None, gguf.GGMLQuantizationType.Q8_0),
        ("cpu", gguf.GGMLQuantizationType.Q8_0, None),
        # Test with F16 quantization (i.e. torch-compmatible quantization).
        ("cpu", gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F16),
        ("cpu", None, gguf.GGMLQuantizationType.F16),
        ("cpu", gguf.GGMLQuantizationType.F16, None),
        # Test all of above cases on CUDA.
        ("cuda", None, None),
        # Test with Q8_0 quantization.
        ("cuda", gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q8_0),
        ("cuda", None, gguf.GGMLQuantizationType.Q8_0),
        ("cuda", gguf.GGMLQuantizationType.Q8_0, None),
        # Test with F16 quantization (i.e. torch-compmatible quantization).
        ("cuda", gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F16),
        ("cuda", None, gguf.GGMLQuantizationType.F16),
        ("cuda", gguf.GGMLQuantizationType.F16, None),
    ],
)
def test_ggml_tensor_multiply(
    device: str, x1_quant_type: gguf.GGMLQuantizationType | None, x2_quant_type: gguf.GGMLQuantizationType | None
):
    # Skip test if CUDA is not available.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    generator = torch.Generator().manual_seed(123)

    x1 = torch.randn(32, 64, generator=generator).to(device=device)
    x2 = torch.randn(32, 64, generator=generator).to(device=device)

    # Quantize the tensors.
    x1_quantized = quantize_tensor(x1, x1_quant_type) if x1_quant_type is not None else x1
    x2_quantized = quantize_tensor(x2, x2_quant_type) if x2_quant_type is not None else x2

    # Check devices.
    for x in [x1, x2, x1_quantized, x2_quantized]:
        assert x.device.type == device

    # Perform the multiplication.
    result = x1 * x2
    result_quantized = x1_quantized * x2_quantized

    assert result.shape == result_quantized.shape
    assert result.dtype == result_quantized.dtype
    assert torch.allclose(result, result_quantized, atol=1e-1)


def test_ggml_tensor_to_dtype_raises_error():
    x = torch.randn(32, 64)
    x_quantized = quantize_tensor(x, gguf.GGMLQuantizationType.Q8_0)

    with pytest.raises(ValueError):
        x_quantized.to(dtype=torch.float32)

    with pytest.raises(ValueError):
        x_quantized.float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
def test_ggml_tensor_to_device():
    x = torch.randn(32, 64)
    x_cpu = quantize_tensor(x, gguf.GGMLQuantizationType.Q8_0)

    x_gpu = x_cpu.to(device=torch.device("cuda"))

    assert x_cpu.device.type == "cpu"
    assert x_gpu.device.type == "cuda"

    assert torch.allclose(x_cpu.quantized_data, x_gpu.quantized_data.cpu(), atol=1e-5)


def test_ggml_tensor_shape():
    x = torch.randn(32, 64)
    x_quantized = quantize_tensor(x, gguf.GGMLQuantizationType.Q8_0)

    assert x_quantized.shape == x.shape
    assert x_quantized.size() == x.size()


def test_ggml_tensor_quantized_shape():
    x = torch.randn(32, 64)
    x_quantized = quantize_tensor(x, gguf.GGMLQuantizationType.Q8_0)

    # This is mainly just a smoke test to confirm that .quantized_shape can be accesses and doesn't hit any weird
    # dispatch errors.
    assert x_quantized.quantized_shape != x.shape


def test_ggml_tensor_calc_size():
    """Test that the calc_tensor_size(...) utility function correctly uses the underlying quantized tensor to calculate
    size rather than the unquantized tensor.
    """
    x = torch.randn(32, 64)
    x_quantized = quantize_tensor(x, gguf.GGMLQuantizationType.Q8_0)

    compression_ratio = calc_tensor_size(x) / calc_tensor_size(x_quantized)
    # Assert that the compression ratio is approximately 4x.
    assert abs(compression_ratio - 4) < 0.5
