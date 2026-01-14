"""Unit tests for SDNQTensor class."""

import pytest
import torch

from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from invokeai.backend.quantization.sdnq.utils import SDNQQuantizationType


class TestSDNQTensor:
    """Tests for SDNQTensor dequantization and operations."""

    def test_symmetric_dequantization(self):
        """Test symmetric dequantization: result = weight * scale."""
        weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
        scale = torch.tensor([0.1], dtype=torch.float32)

        tensor = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size([2, 2]),
            compute_dtype=torch.float32,
            scale=scale,
        )

        dequantized = tensor.get_dequantized_tensor()
        expected = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
        assert torch.allclose(dequantized, expected, atol=1e-6)

    def test_asymmetric_dequantization(self):
        """Test asymmetric dequantization: result = (weight - zero_point) * scale."""
        weight = torch.tensor([[10, 20], [30, 40]], dtype=torch.int8)
        scale = torch.tensor([0.1], dtype=torch.float32)
        zero_point = torch.tensor([5], dtype=torch.float32)

        tensor = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_ASYM,
            tensor_shape=torch.Size([2, 2]),
            compute_dtype=torch.float32,
            scale=scale,
            zero_point=zero_point,
        )

        dequantized = tensor.get_dequantized_tensor()
        # (weight - zero_point) * scale = ([10-5, 20-5], [30-5, 40-5]) * 0.1
        expected = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)
        assert torch.allclose(dequantized, expected, atol=1e-6)

    def test_svd_correction(self):
        """Test SVD correction: result = dequant + svd_up @ svd_down."""
        weight = torch.tensor([[0, 0], [0, 0]], dtype=torch.int8)
        scale = torch.tensor([1.0], dtype=torch.float32)
        svd_up = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        svd_down = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

        tensor = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size([2, 2]),
            compute_dtype=torch.float32,
            scale=scale,
            svd_up=svd_up,
            svd_down=svd_down,
        )

        assert tensor.has_svd
        dequantized = tensor.get_dequantized_tensor()
        # SVD correction: [[1], [2]] @ [[1, 1]] = [[1, 1], [2, 2]]
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        assert torch.allclose(dequantized, expected, atol=1e-6)

    def test_shape_properties(self):
        """Test tensor shape properties."""
        weight = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int8)
        scale = torch.tensor([0.1], dtype=torch.float32)

        tensor = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size([2, 3]),
            compute_dtype=torch.float32,
            scale=scale,
        )

        assert tensor.shape == torch.Size([2, 3])
        assert tensor.size() == torch.Size([2, 3])
        assert tensor.size(0) == 2
        assert tensor.size(1) == 3
        assert tensor.quantized_shape == torch.Size([2, 3])

    def test_is_asymmetric_property(self):
        """Test is_asymmetric property."""
        weight = torch.tensor([[1]], dtype=torch.int8)
        scale = torch.tensor([0.1], dtype=torch.float32)

        # Symmetric (no zero_point)
        tensor_sym = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size([1, 1]),
            compute_dtype=torch.float32,
            scale=scale,
        )
        assert not tensor_sym.is_asymmetric

        # Asymmetric (has zero_point)
        tensor_asym = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_ASYM,
            tensor_shape=torch.Size([1, 1]),
            compute_dtype=torch.float32,
            scale=scale,
            zero_point=torch.tensor([0.0]),
        )
        assert tensor_asym.is_asymmetric

    def test_has_svd_property(self):
        """Test has_svd property."""
        weight = torch.tensor([[1]], dtype=torch.int8)
        scale = torch.tensor([0.1], dtype=torch.float32)

        # No SVD
        tensor_no_svd = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size([1, 1]),
            compute_dtype=torch.float32,
            scale=scale,
        )
        assert not tensor_no_svd.has_svd

        # With SVD
        tensor_with_svd = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size([1, 1]),
            compute_dtype=torch.float32,
            scale=scale,
            svd_up=torch.tensor([[1.0]]),
            svd_down=torch.tensor([[1.0]]),
        )
        assert tensor_with_svd.has_svd

    def test_repr(self):
        """Test tensor string representation."""
        weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
        scale = torch.tensor([0.1], dtype=torch.float32)

        tensor = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size([2, 2]),
            compute_dtype=torch.float32,
            scale=scale,
        )

        repr_str = repr(tensor)
        assert "SDNQTensor" in repr_str
        assert "int8_sym" in repr_str
        assert "has_svd=False" in repr_str

    def test_compute_dtype_bfloat16(self):
        """Test dequantization with bfloat16 compute dtype."""
        weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
        scale = torch.tensor([0.5], dtype=torch.float32)

        tensor = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size([2, 2]),
            compute_dtype=torch.bfloat16,
            scale=scale,
        )

        dequantized = tensor.get_dequantized_tensor()
        assert dequantized.dtype == torch.bfloat16

    def test_requires_grad_noop(self):
        """Test that requires_grad_ is a no-op for inference-only tensor."""
        weight = torch.tensor([[1]], dtype=torch.int8)
        scale = torch.tensor([0.1], dtype=torch.float32)

        tensor = SDNQTensor(
            data=weight,
            quantization_type=SDNQQuantizationType.INT8_SYM,
            tensor_shape=torch.Size([1, 1]),
            compute_dtype=torch.float32,
            scale=scale,
        )

        # Should return self without error
        result = tensor.requires_grad_(True)
        assert result is tensor
