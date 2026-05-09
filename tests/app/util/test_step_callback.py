"""Tests for diffusion step callback preview image generation."""

import torch
from PIL import Image

from invokeai.app.util.step_callback import (
    QWEN_IMAGE_LATENT_RGB_BIAS,
    QWEN_IMAGE_LATENT_RGB_FACTORS,
    sample_to_lowres_estimated_image,
)


class TestSampleToLowresEstimatedImage:
    """Test the latent-to-preview-image conversion used during denoising."""

    def test_qwen_image_preview_produces_valid_image(self):
        """A synthetic Qwen latent tensor produces a valid RGB preview image."""
        # Create a small 1x16x4x4 latent tensor (batch=1, channels=16, 4x4 spatial)
        torch.manual_seed(42)
        sample = torch.randn(1, 16, 4, 4)

        factors = torch.tensor(QWEN_IMAGE_LATENT_RGB_FACTORS, dtype=sample.dtype)
        bias = torch.tensor(QWEN_IMAGE_LATENT_RGB_BIAS, dtype=sample.dtype)

        image = sample_to_lowres_estimated_image(
            samples=sample,
            latent_rgb_factors=factors,
            latent_rgb_bias=bias,
        )

        assert isinstance(image, Image.Image)
        assert image.size == (4, 4)
        assert image.mode == "RGB"

    def test_qwen_image_preview_deterministic(self):
        """The same input tensor always produces the same preview image."""
        sample = torch.ones(1, 16, 2, 2)

        factors = torch.tensor(QWEN_IMAGE_LATENT_RGB_FACTORS, dtype=sample.dtype)
        bias = torch.tensor(QWEN_IMAGE_LATENT_RGB_BIAS, dtype=sample.dtype)

        image1 = sample_to_lowres_estimated_image(samples=sample, latent_rgb_factors=factors, latent_rgb_bias=bias)
        image2 = sample_to_lowres_estimated_image(samples=sample, latent_rgb_factors=factors, latent_rgb_bias=bias)

        assert list(image1.getdata()) == list(image2.getdata())

    def test_qwen_image_preview_known_value(self):
        """Verify the preview computation against a hand-calculated expected value.

        With a 1x16x1x1 tensor of all ones:
        - latent_image = [1,1,...,1] @ factors = sum of each column of factors
        - R = sum(col 0) = 0.3677, G = sum(col 1) = 0.4577, B = sum(col 2) = 0.9101
        - After bias: R = 0.1842, G = 0.3709, B = 0.5741
        - After scale ((x+1)/2): R = 0.5921, G = 0.6855, B = 0.7871
        - After quantize (*255): R = 151, G = 175, B = 201
        """
        sample = torch.ones(1, 16, 1, 1)

        factors = torch.tensor(QWEN_IMAGE_LATENT_RGB_FACTORS, dtype=sample.dtype)
        bias = torch.tensor(QWEN_IMAGE_LATENT_RGB_BIAS, dtype=sample.dtype)

        image = sample_to_lowres_estimated_image(samples=sample, latent_rgb_factors=factors, latent_rgb_bias=bias)

        assert image.size == (1, 1)
        pixel = image.getpixel((0, 0))

        # Compute expected values
        col_sums = [sum(row[c] for row in QWEN_IMAGE_LATENT_RGB_FACTORS) for c in range(3)]
        expected = []
        for c in range(3):
            val = col_sums[c] + QWEN_IMAGE_LATENT_RGB_BIAS[c]
            val = (val + 1) / 2  # scale from [-1,1] to [0,1]
            val = max(0.0, min(1.0, val))  # clamp
            expected.append(int(val * 255))

        assert pixel == tuple(expected), f"Expected {tuple(expected)}, got {pixel}"

    def test_qwen_image_preview_zeros_tensor(self):
        """A zero tensor with bias produces a valid image reflecting just the bias."""
        sample = torch.zeros(1, 16, 2, 2)

        factors = torch.tensor(QWEN_IMAGE_LATENT_RGB_FACTORS, dtype=sample.dtype)
        bias = torch.tensor(QWEN_IMAGE_LATENT_RGB_BIAS, dtype=sample.dtype)

        image = sample_to_lowres_estimated_image(samples=sample, latent_rgb_factors=factors, latent_rgb_bias=bias)

        assert isinstance(image, Image.Image)
        assert image.size == (2, 2)

        # All pixels should be identical (uniform zero input)
        pixels = list(image.get_flattened_data())
        assert all(p == pixels[0] for p in pixels)

        # With zero input, result = bias, scaled: ((bias + 1) / 2) * 255
        expected = []
        for c in range(3):
            val = (QWEN_IMAGE_LATENT_RGB_BIAS[c] + 1) / 2
            val = max(0.0, min(1.0, val))
            expected.append(int(val * 255))
        assert pixels[0] == tuple(expected)

    def test_qwen_image_factors_have_correct_shape(self):
        """Qwen Image uses 16 latent channels, so factors should be 16x3."""
        assert len(QWEN_IMAGE_LATENT_RGB_FACTORS) == 16
        for row in QWEN_IMAGE_LATENT_RGB_FACTORS:
            assert len(row) == 3
        assert len(QWEN_IMAGE_LATENT_RGB_BIAS) == 3

    def test_3d_input_accepted(self):
        """sample_to_lowres_estimated_image accepts 3D input (no batch dim)."""
        sample = torch.randn(16, 4, 4)  # no batch dimension

        factors = torch.tensor(QWEN_IMAGE_LATENT_RGB_FACTORS, dtype=sample.dtype)
        bias = torch.tensor(QWEN_IMAGE_LATENT_RGB_BIAS, dtype=sample.dtype)

        image = sample_to_lowres_estimated_image(samples=sample, latent_rgb_factors=factors, latent_rgb_bias=bias)

        assert isinstance(image, Image.Image)
        assert image.size == (4, 4)
