from unittest.mock import Mock

import gguf
import pytest
import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.utils import DEQUANTIZE_FUNCTIONS, TORCH_COMPATIBLE_QTYPES
from invokeai.backend.util import level_zero
from invokeai.backend.util.calc_tensor_size import calc_tensor_size
from invokeai.backend.util.devices import TorchDevice


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


def test_ggml_tensor_numpy_fallback_dequantize_uses_logical_shape():
    """Qtypes without a torch dequantize kernel fall back to numpy, which infers the shape from the stored data.

    ComfyUI reshapes tensors before quantizing them, so the stored shape is not necessarily the logical one - the
    fallback must still return the logical shape (see GGMLTensor.tensor_shape).
    """
    qtype = gguf.GGMLQuantizationType.IQ4_NL
    assert qtype not in DEQUANTIZE_FUNCTIONS and qtype not in TORCH_COMPATIBLE_QTYPES

    # One IQ4_NL block is 32 elements in 18 bytes, so this holds 8 rows of 32 elements = 256 elements, which the
    # numpy fallback dequantizes to (8, 32). The logical shape is a different view of the same 256 elements.
    generator = torch.Generator().manual_seed(123)
    quantized = torch.randint(0, 256, (8, 18), dtype=torch.uint8, generator=generator)
    logical_shape = torch.Size((4, 64))

    t = GGMLTensor(
        data=quantized,
        ggml_quantization_type=qtype,
        tensor_shape=logical_shape,
        compute_dtype=torch.float32,
    )

    assert t.get_dequantized_tensor().shape == logical_shape


def _gguf_linear(
    in_features: int,
    out_features: int,
    qtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.Q8_0,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.nn.Linear:
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    packed = torch.zeros((out_features, in_features // block_size * type_size), dtype=torch.uint8)
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    linear.weight = torch.nn.Parameter(
        GGMLTensor(packed, qtype, torch.Size((out_features, in_features)), compute_dtype), requires_grad=False
    )
    return linear


def test_the_shared_dequant_transient_counts_gguf_linears(monkeypatch):
    """Every denoise and encoder node sizes its reservation with `peak_dequant_transient_bytes`, so a GGUF
    Linear -- dequantized into a full copy on every forward -- has to count there, as an int8 one does.
    It is the largest layer that counts, whichever scheme it uses, since the layers run one at a time."""
    from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
    from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear

    monkeypatch.setattr(
        TorchDevice, "choose_torch_device", lambda: pytest.fail("No host fallback needs a device lookup")
    )
    monkeypatch.setattr(level_zero, "xpu_device_is_integrated", lambda _: pytest.fail("No host fallback needs a probe"))
    gguf_model = torch.nn.Sequential(_gguf_linear(64, 128), _gguf_linear(64, 32))
    gguf_peak = peak_dequant_transient_bytes(gguf_model, torch.float32)

    # 8192 elements * (4-byte result + 2-byte Q8 intermediates), plus 256 packed 34-byte blocks.
    assert gguf_peak == 57856
    assert peak_dequant_transient_bytes(torch.nn.Sequential(gguf_model[0]), torch.float32) == 57856

    # Whichever scheme holds the largest layer decides, in either order.
    larger_int8 = Int8ConvrotLinear(
        weight=torch.zeros(128, 128, dtype=torch.int8), weight_scale=torch.ones(128, 1), convrot=False
    )
    mixed = torch.nn.Sequential(gguf_model[0], larger_int8)
    assert peak_dequant_transient_bytes(mixed, torch.float32) == larger_int8.dequant_transient_bytes(torch.float32)
    smaller_int8 = Int8ConvrotLinear(
        weight=torch.zeros(8, 64, dtype=torch.int8), weight_scale=torch.ones(8, 1), convrot=False
    )
    assert smaller_int8.dequant_transient_bytes(torch.float32) < gguf_peak
    assert peak_dequant_transient_bytes(torch.nn.Sequential(smaller_int8, gguf_model[0]), torch.float32) == gguf_peak
    assert peak_dequant_transient_bytes(torch.nn.Sequential(torch.nn.Linear(64, 64)), torch.float32) == 0


@pytest.mark.parametrize("compute_dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
def test_host_dequant_reservation_counts_only_result_and_packed_weight(compute_dtype, monkeypatch):
    """Host decode intermediates do not consume dedicated VRAM, even with the weight still on CPU."""
    from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
    from invokeai.backend.quantization.gguf.ggml_tensor import peak_ggml_linear_dequant_transient_bytes

    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cuda:1"))
    linear = _gguf_linear(256, 2, gguf.GGMLQuantizationType.IQ4_NL, compute_dtype)
    weight = linear.weight
    assert weight.device.type == "cpu"
    decoded = weight.get_dequantized_tensor()
    assert decoded.dtype == compute_dtype

    model = torch.nn.Sequential(linear)
    expected_bytes = decoded.nbytes + weight.quantized_data.nbytes
    assert peak_ggml_linear_dequant_transient_bytes(model) == expected_bytes
    # GGUF carries its own dtype, even if a caller passes a different module-level dtype.
    assert peak_dequant_transient_bytes(model, torch.float64) == expected_bytes


@pytest.mark.parametrize(
    ("execution_device", "integrated", "compute_dtype", "intermediate_bytes"),
    [
        pytest.param("cpu", None, torch.float32, 8, id="cpu-fp32"),
        pytest.param("mps", None, torch.float16, 8, id="mps-fp16"),
        pytest.param("xpu:2", True, torch.float32, 8, id="integrated-xpu-fp32"),
        pytest.param("xpu:2", False, torch.float16, 0, id="discrete-xpu-fp16"),
        pytest.param("xpu:2", None, torch.float16, 0, id="unknown-xpu-fp16"),
    ],
)
def test_host_dequant_reservation_follows_execution_memory_budget(
    execution_device, integrated, compute_dtype, intermediate_bytes, monkeypatch
):
    """Shared execution retains the prior conservative 8 bytes/element for host intermediates."""
    from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes

    device = torch.device(execution_device)
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: device)

    def is_integrated(probed_device):
        assert probed_device == device
        return integrated

    monkeypatch.setattr(level_zero, "xpu_device_is_integrated", is_integrated)
    linear = _gguf_linear(256, 2, gguf.GGMLQuantizationType.IQ4_NL, compute_dtype)
    assert linear.weight.device.type == "cpu"
    # 512 logical elements and sixteen 18-byte IQ4_NL blocks; the weight owns the result dtype.
    expected_bytes = 512 * (compute_dtype.itemsize + intermediate_bytes) + 288
    assert peak_dequant_transient_bytes(linear, torch.float64) == expected_bytes


def test_bf16_device_dequant_reservation_keeps_measured_intermediates():
    from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes

    linear = _gguf_linear(256, 2, gguf.GGMLQuantizationType.BF16, torch.float16)
    # 512 elements * (2-byte result + 6-byte BF16 intermediates), plus 1024 packed bytes.
    assert peak_dequant_transient_bytes(linear, torch.float64) == 5120


def test_dedicated_dequant_reservation_uses_largest_bytes_in_either_order(monkeypatch):
    from invokeai.backend.quantization.gguf.ggml_tensor import peak_ggml_linear_dequant_transient_bytes

    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cuda:1"))
    larger_iq = _gguf_linear(256, 16, gguf.GGMLQuantizationType.IQ4_NL)
    smaller_q8 = _gguf_linear(256, 12)
    # IQ: 4096 * 4 + 2304 = 18688. Q8: 3072 * (4 + 2) + 3264 = 21696.
    for layers in ((larger_iq, smaller_q8), (smaller_q8, larger_iq)):
        assert peak_ggml_linear_dequant_transient_bytes(torch.nn.Sequential(*layers)) == 21696


def test_host_dequant_reservation_resolves_execution_device_once_per_call(monkeypatch):
    from invokeai.backend.quantization.gguf.ggml_tensor import peak_ggml_linear_dequant_transient_bytes

    choose_device = Mock(side_effect=[torch.device("xpu:2"), torch.device("cuda:1")])
    probe = Mock(return_value=True)
    monkeypatch.setattr(TorchDevice, "choose_torch_device", choose_device)
    monkeypatch.setattr(level_zero, "xpu_device_is_integrated", probe)
    model = torch.nn.Sequential(
        _gguf_linear(256, 2, gguf.GGMLQuantizationType.IQ4_NL, torch.float16),
        _gguf_linear(256, 1, gguf.GGMLQuantizationType.IQ4_NL, torch.float16),
    )

    assert peak_ggml_linear_dequant_transient_bytes(model) == 5408  # 512 * (2 + 8) + 288.
    choose_device.assert_called_once_with()
    assert peak_ggml_linear_dequant_transient_bytes(model) == 1312  # 512 * 2 + 288.
    assert choose_device.call_count == 2
    probe.assert_called_once_with(torch.device("xpu:2"))


def test_new_device_dequant_kernel_keeps_conservative_intermediate_reservation(monkeypatch):
    from invokeai.backend.quantization.gguf.ggml_tensor import peak_ggml_linear_dequant_transient_bytes

    qtype = gguf.GGMLQuantizationType.IQ4_NL
    packed = torch.zeros((16, 18), dtype=torch.uint8)
    weight = GGMLTensor(packed, qtype, torch.Size((2, 256)), torch.float16)
    linear = torch.nn.Linear(256, 2, bias=False)
    linear.weight = torch.nn.Parameter(weight, requires_grad=False)
    # A newly registered device kernel has no measured table entry yet. Do not treat it as host-only.
    monkeypatch.setitem(DEQUANTIZE_FUNCTIONS, qtype, DEQUANTIZE_FUNCTIONS[gguf.GGMLQuantizationType.Q8_0])
    assert peak_ggml_linear_dequant_transient_bytes(linear) == 512 * (2 + 8) + packed.nbytes
