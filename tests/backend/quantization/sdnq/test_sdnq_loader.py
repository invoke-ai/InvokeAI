"""Integration tests for SDNQ state dict loader."""

import pytest
import torch
from pathlib import Path

from invokeai.backend.quantization.sdnq.loaders import has_sdnq_keys, sdnq_sd_loader
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor


class TestSDNQLoader:
    """Tests for SDNQ state dict loading."""

    @pytest.fixture
    def mock_sdnq_file(self, tmp_path: Path) -> Path:
        """Create a mock SDNQ safetensors file."""
        from safetensors.torch import save_file

        # Create mock quantized weights with scale
        tensors = {
            "layer1.weight": torch.randint(-128, 127, (256, 256), dtype=torch.int8),
            "layer1.scale": torch.tensor([0.01], dtype=torch.float32),
            "layer1.bias": torch.randn(256, dtype=torch.float32),
            "layer2.weight": torch.randint(-128, 127, (128, 256), dtype=torch.int8),
            "layer2.scale": torch.tensor([0.02], dtype=torch.float32),
            "layer2.zero_point": torch.tensor([0.5], dtype=torch.float32),
            "norm.weight": torch.randn(256, dtype=torch.float32),  # Not quantized
        }

        file_path = tmp_path / "model.safetensors"
        save_file(tensors, str(file_path))
        return file_path

    @pytest.fixture
    def mock_sdnq_file_with_svd(self, tmp_path: Path) -> Path:
        """Create a mock SDNQ safetensors file with SVD correction."""
        from safetensors.torch import save_file

        tensors = {
            "layer.weight": torch.randint(-128, 127, (64, 64), dtype=torch.int8),
            "layer.scale": torch.tensor([0.1], dtype=torch.float32),
            "layer.svd_up": torch.randn(64, 16, dtype=torch.float32),
            "layer.svd_down": torch.randn(16, 64, dtype=torch.float32),
        }

        file_path = tmp_path / "model_svd.safetensors"
        save_file(tensors, str(file_path))
        return file_path

    @pytest.fixture
    def mock_non_sdnq_file(self, tmp_path: Path) -> Path:
        """Create a mock non-SDNQ safetensors file (no scale keys)."""
        from safetensors.torch import save_file

        tensors = {
            "layer.weight": torch.randn(256, 256, dtype=torch.float32),
            "layer.bias": torch.randn(256, dtype=torch.float32),
        }

        file_path = tmp_path / "model_regular.safetensors"
        save_file(tensors, str(file_path))
        return file_path

    def test_load_sdnq_file(self, mock_sdnq_file: Path):
        """Test loading an SDNQ quantized file."""
        sd = sdnq_sd_loader(mock_sdnq_file)

        # Check that quantized weights are wrapped in SDNQTensor
        assert "layer1.weight" in sd
        assert isinstance(sd["layer1.weight"], SDNQTensor)
        assert "layer2.weight" in sd
        assert isinstance(sd["layer2.weight"], SDNQTensor)

        # Check that non-quantized tensors are regular tensors
        assert "layer1.bias" in sd
        assert isinstance(sd["layer1.bias"], torch.Tensor)
        assert not isinstance(sd["layer1.bias"], SDNQTensor)

        assert "norm.weight" in sd
        assert isinstance(sd["norm.weight"], torch.Tensor)
        assert not isinstance(sd["norm.weight"], SDNQTensor)

    def test_symmetric_vs_asymmetric(self, mock_sdnq_file: Path):
        """Test that symmetric and asymmetric quantization is detected."""
        sd = sdnq_sd_loader(mock_sdnq_file)

        # layer1 is symmetric (no zero_point)
        layer1_weight = sd["layer1.weight"]
        assert isinstance(layer1_weight, SDNQTensor)
        assert not layer1_weight.is_asymmetric

        # layer2 is asymmetric (has zero_point)
        layer2_weight = sd["layer2.weight"]
        assert isinstance(layer2_weight, SDNQTensor)
        assert layer2_weight.is_asymmetric

    def test_svd_loading(self, mock_sdnq_file_with_svd: Path):
        """Test that SVD correction matrices are loaded."""
        sd = sdnq_sd_loader(mock_sdnq_file_with_svd)

        layer_weight = sd["layer.weight"]
        assert isinstance(layer_weight, SDNQTensor)
        assert layer_weight.has_svd

    def test_has_sdnq_keys_positive(self, mock_sdnq_file: Path):
        """Test has_sdnq_keys returns True for SDNQ files."""
        from safetensors.torch import load_file

        sd = load_file(str(mock_sdnq_file))
        assert has_sdnq_keys(sd)

    def test_has_sdnq_keys_negative(self, mock_non_sdnq_file: Path):
        """Test has_sdnq_keys returns False for non-SDNQ files."""
        from safetensors.torch import load_file

        sd = load_file(str(mock_non_sdnq_file))
        assert not has_sdnq_keys(sd)

    def test_compute_dtype_propagation(self, mock_sdnq_file: Path):
        """Test that compute_dtype is set correctly on SDNQTensor."""
        sd = sdnq_sd_loader(mock_sdnq_file, compute_dtype=torch.bfloat16)

        layer1_weight = sd["layer1.weight"]
        assert isinstance(layer1_weight, SDNQTensor)
        assert layer1_weight.compute_dtype == torch.bfloat16

        # Verify dequantization produces correct dtype
        dequantized = layer1_weight.get_dequantized_tensor()
        assert dequantized.dtype == torch.bfloat16

    def test_dequantization_produces_correct_shape(self, mock_sdnq_file: Path):
        """Test that dequantized tensors have correct shape."""
        sd = sdnq_sd_loader(mock_sdnq_file)

        layer1_weight = sd["layer1.weight"]
        assert isinstance(layer1_weight, SDNQTensor)
        assert layer1_weight.shape == torch.Size([256, 256])

        dequantized = layer1_weight.get_dequantized_tensor()
        assert dequantized.shape == torch.Size([256, 256])

    def test_scale_keys_not_in_output(self, mock_sdnq_file: Path):
        """Test that scale/zero_point keys are not in the output dict."""
        sd = sdnq_sd_loader(mock_sdnq_file)

        # Scale and zero_point should be absorbed into SDNQTensor
        assert "layer1.scale" not in sd
        assert "layer2.scale" not in sd
        assert "layer2.zero_point" not in sd

    def test_empty_directory_raises(self, tmp_path: Path):
        """Test that loading from empty directory raises ValueError."""
        with pytest.raises(ValueError, match="No safetensors files found"):
            sdnq_sd_loader(tmp_path)
