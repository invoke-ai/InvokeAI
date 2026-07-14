"""Test that Qwen Image VAE invocations properly estimate and request working memory."""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

from invokeai.app.invocations.qwen_image_image_to_latents import QwenImageImageToLatentsInvocation
from invokeai.app.invocations.qwen_image_latents_to_image import QwenImageLatentsToImageInvocation
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_qwen_image


class TestQwenImageWorkingMemoryEstimate:
    """Lock in the per-backend scaling constants calibrated in scripts/calibrate_qwen_vae_working_memory.py.

    These differ by backend because the Qwen VAE is attention-heavy: ROCm falls back to math attention
    (O(area^2), much higher memory) while CUDA uses Flash/efficient attention. A regression that swaps
    the constants would reintroduce the ROCm OOM (under-estimate) or needlessly over-budget CUDA.
    """

    # (operation, latent_h, latent_w) -> the estimator scales pixel area (latent * 8 for decode,
    # raw for encode) by element_size and the constant.
    @pytest.mark.parametrize(
        "operation, is_rocm, expected_constant",
        [
            ("decode", True, 5500),
            ("decode", False, 2900),
            ("encode", True, 6300),
            ("encode", False, 1600),
        ],
    )
    def test_constant_selected_per_backend(self, operation, is_rocm, expected_constant):
        mock_vae = MagicMock(spec=AutoencoderKLQwenImage)
        mock_vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.float16)])  # element_size == 2

        # decode receives latents (pixel area = latent area * 8^2); encode receives a pixel image.
        if operation == "decode":
            image_tensor = torch.zeros(1, 16, 1, 64, 64)
            h = w = 64 * 8
        else:
            image_tensor = torch.zeros(1, 3, 1, 512, 512)
            h = w = 512

        hip_value = "7.1.0" if is_rocm else None
        with patch("torch.version.hip", hip_value):
            result = estimate_vae_working_memory_qwen_image(
                operation=operation, image_tensor=image_tensor, vae=mock_vae
            )

        assert result == h * w * 2 * expected_constant


class TestQwenImageWorkingMemory:
    """Test that Qwen Image VAE invocations request working memory before decode/encode."""

    def _mock_vae_info(self):
        """Build a mocked AutoencoderKLQwenImage and its LoadedModel wrapper."""
        mock_vae = MagicMock(spec=AutoencoderKLQwenImage)

        # Create mock parameter for dtype detection
        mock_param = torch.zeros(1)
        mock_vae.parameters.return_value = iter([mock_param])

        # Create mock vae_info with a model_on_device context manager yielding (None, vae)
        mock_vae_info = MagicMock()
        mock_vae_info.model = mock_vae

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=(None, mock_vae))
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_vae_info.model_on_device = MagicMock(return_value=mock_cm)

        return mock_vae, mock_vae_info

    def test_qwen_latents_to_image_requests_working_memory(self):
        """QwenImageLatentsToImageInvocation estimates decode memory and passes it to the cache."""
        mock_vae, mock_vae_info = self._mock_vae_info()

        # Mock the context
        mock_context = MagicMock()
        mock_context.models.load.return_value = mock_vae_info

        # Mock latents (5D: B, C, num_frames, H, W)
        mock_latents = torch.zeros(1, 16, 1, 64, 64)
        mock_context.tensors.load.return_value = mock_latents

        estimation_path = "invokeai.app.invocations.qwen_image_latents_to_image.estimate_vae_working_memory_qwen_image"
        seamless_path = "invokeai.app.invocations.qwen_image_latents_to_image.SeamlessExt.static_patch_model"

        with (
            patch(estimation_path) as mock_estimate,
            patch(seamless_path, return_value=nullcontext()),
        ):
            expected_memory = 1024 * 1024 * 10000  # 10GB
            mock_estimate.return_value = expected_memory

            invocation = QwenImageLatentsToImageInvocation.model_construct(
                latents=MagicMock(latents_name="test_latents"),
                vae=MagicMock(vae=MagicMock(), seamless_axes=["x", "y"]),
            )

            try:
                invocation.invoke(mock_context)
            except Exception:
                # Downstream decode math fails under mocking; we only care that the cache was
                # asked to reserve the estimated working memory before entering the device context.
                pass

            mock_estimate.assert_called_once()
            assert mock_estimate.call_args.kwargs["operation"] == "decode"
            mock_vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected_memory)

    def test_seamless_patch_is_applied_to_converted_anima_vae(self):
        original_vae, mock_vae_info = self._mock_vae_info()
        converted_vae = MagicMock(spec=AutoencoderKLQwenImage)
        converted_vae.parameters.return_value = iter([torch.zeros(1)])
        converted_vae.dtype = torch.float32
        converted_vae.config.z_dim = 16
        converted_vae.config.latents_mean = [0.0] * 16
        converted_vae.config.latents_std = [1.0] * 16
        converted_vae.decode.side_effect = RuntimeError("stop after seamless patch")
        mock_context = MagicMock()
        mock_context.models.load.return_value = mock_vae_info
        mock_context.tensors.load.return_value = torch.zeros(1, 16, 1, 2, 2)

        with (
            patch(
                "invokeai.app.invocations.qwen_image_latents_to_image.estimate_vae_working_memory_qwen_image",
                return_value=1,
            ),
            patch(
                "invokeai.app.invocations.qwen_image_latents_to_image.as_qwen_image_vae",
                return_value=converted_vae,
            ),
            patch(
                "invokeai.app.invocations.qwen_image_latents_to_image.SeamlessExt.static_patch_model",
                return_value=nullcontext(),
            ) as patch_seamless,
            patch(
                "invokeai.app.invocations.qwen_image_latents_to_image.TorchDevice.choose_torch_device",
                return_value=torch.device("cpu"),
            ),
        ):
            invocation = QwenImageLatentsToImageInvocation.model_construct(
                latents=MagicMock(latents_name="test_latents"),
                vae=MagicMock(vae=MagicMock(), seamless_axes=["x"]),
            )
            with pytest.raises(RuntimeError, match="stop after seamless patch"):
                invocation.invoke(mock_context)

        assert original_vae is not converted_vae
        patch_seamless.assert_called_once_with(converted_vae, ["x"])

    def test_qwen_image_to_latents_requests_working_memory(self):
        """QwenImageImageToLatentsInvocation estimates encode memory and passes it to the cache."""
        mock_vae, mock_vae_info = self._mock_vae_info()

        mock_image_tensor = torch.zeros(1, 3, 512, 512)

        estimation_path = "invokeai.app.invocations.qwen_image_image_to_latents.estimate_vae_working_memory_qwen_image"

        with patch(estimation_path) as mock_estimate:
            expected_memory = 1024 * 1024 * 5000  # 5GB
            mock_estimate.return_value = expected_memory

            try:
                QwenImageImageToLatentsInvocation.vae_encode(mock_vae_info, mock_image_tensor)
            except Exception:
                # Downstream encode math fails under mocking; we only care that the cache was
                # asked to reserve the estimated working memory before entering the device context.
                pass

            mock_estimate.assert_called_once()
            assert mock_estimate.call_args.kwargs["operation"] == "encode"
            mock_vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected_memory)
