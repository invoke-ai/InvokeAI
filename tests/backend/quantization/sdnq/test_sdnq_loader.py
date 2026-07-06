"""Integration tests for SDNQ state dict loader."""

from pathlib import Path

import pytest
import torch

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

    def test_uint4_group_size_from_scale_overrides_config(self, tmp_path: Path):
        """A uint4 tensor whose actual group size differs from the config value must dequantize.

        SDNQ dynamic-mixed models (e.g. FLUX.2 Klein 4B) quantize some layers with a group size
        that differs from the nominal ``group_size`` in quantization_config.json. Here the config
        advertises group_size=128, but the scale tensor has 4 groups over 256 in-features, i.e. a
        real group size of 64. The loader must trust the scale tensor's group dimension; otherwise
        ``num_groups = in_features // 128 = 2`` disagrees with scale.shape[1] == 4 and the per-group
        reshape/broadcast inside dequantize_uint4_per_group fails.
        """
        import json

        from safetensors.torch import save_file

        out_features = 32
        in_features = 256
        actual_group_size = 64
        num_groups = in_features // actual_group_size  # 4

        # uint4 packs 2 values per byte -> packed last dim is in_features // 2.
        packed_weight = torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8)
        # Per-group scale/zero_point shaped [out_features, num_groups, 1] (64-wide groups).
        scale = torch.rand(out_features, num_groups, 1, dtype=torch.float32) * 0.01
        zero_point = torch.rand(out_features, num_groups, 1, dtype=torch.float32) * 0.01

        model_dir = tmp_path / "mixed_group_model"
        model_dir.mkdir()
        save_file(
            {
                "layer.weight": packed_weight,
                "layer.scale": scale,
                "layer.zero_point": zero_point,
            },
            str(model_dir / "model.safetensors"),
        )
        # Config advertises group_size=128, which is WRONG for this 64-wide-group tensor.
        (model_dir / "quantization_config.json").write_text(
            json.dumps({"weights_dtype": "uint4", "group_size": 128}),
            encoding="utf-8",
        )

        sd = sdnq_sd_loader(model_dir)

        layer_weight = sd["layer.weight"]
        assert isinstance(layer_weight, SDNQTensor)
        # Group size must be derived from the scale tensor (256 / 4 = 64), not the config's 128.
        assert layer_weight._group_size == actual_group_size

        # Dequantization must succeed and produce the original (unpacked) shape.
        dequantized = layer_weight.get_dequantized_tensor()
        assert dequantized.shape == torch.Size([out_features, in_features])
