import gguf
import pytest
import torch

from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.row_ops import take_rows

_Q8_0 = gguf.GGMLQuantizationType.Q8_0


def _q8_0(rows: int, cols: int) -> tuple[GGMLTensor, torch.Tensor]:
    values = torch.randn(rows, cols, generator=torch.Generator().manual_seed(0))
    packed = torch.from_numpy(gguf.quantize(values.numpy(), _Q8_0))
    tensor = GGMLTensor(packed, _Q8_0, torch.Size((rows, cols)), torch.float32)
    return tensor, tensor.get_dequantized_tensor()


def test_rows_of_a_packed_tensor_are_gathered_without_unpacking() -> None:
    """A subset in a new order, so the result's shape has to follow the index, not the source."""
    tensor, dequantized = _q8_0(6, 64)
    index = torch.tensor([4, 0, 5])

    rows = take_rows(tensor, index)

    assert isinstance(rows, GGMLTensor)
    assert rows._ggml_quantization_type is _Q8_0
    assert rows.compute_dtype is torch.float32
    assert rows.shape == (3, 64)
    assert torch.equal(rows.get_dequantized_tensor(), dequantized[index])


def test_packed_data_not_stored_one_row_per_output_is_refused() -> None:
    """ComfyUI-GGUF reshapes tensors its quantizer rejects and records the native shape; such packed
    rows are not output rows, and gathering them would mix values across outputs."""
    tensor, _ = _q8_0(6, 64)
    reshaped = GGMLTensor(tensor.quantized_data.reshape(3, -1), _Q8_0, tensor.tensor_shape, torch.float32)

    with pytest.raises(ValueError, match="not stored one row per output"):
        take_rows(reshaped, torch.tensor([0, 1]))
