"""Test that Z-Image VAE invocations properly estimate and request working memory."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.app.invocations.z_image_image_to_latents import ZImageImageToLatentsInvocation
from invokeai.backend.flux.modules.autoencoder import AutoEncoder as FluxAutoEncoder


class TestZImageWorkingMemory:
    """Test that Z-Image VAE invocations request working memory."""

    @pytest.mark.parametrize("vae_type", [AutoencoderKL, FluxAutoEncoder])
    def test_z_image_latents_to_image_requests_working_memory(self, vae_type):
        """Test that ZImageLatentsToImageInvocation estimates and requests working memory."""
        # Create mock VAE
        mock_vae = MagicMock(spec=vae_type)

        # Only set config for AutoencoderKL (FluxAutoEncoder doesn't use config)
        if vae_type == AutoencoderKL:
            mock_vae.config.scaling_factor = 1.0
            mock_vae.config.shift_factor = None

        # Create mock parameter for dtype detection
        mock_param = torch.zeros(1)
        mock_vae.parameters.return_value = iter([mock_param])

        # Create mock vae_info
        mock_vae_info = MagicMock()
        mock_vae_info.model = mock_vae

        # Create mock context manager return value
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=(None, mock_vae))
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_vae_info.model_on_device = MagicMock(return_value=mock_cm)

        # Mock the context
        mock_context = MagicMock()
        mock_context.models.load.return_value = mock_vae_info

        # Mock latents
        mock_latents = torch.zeros(1, 16, 64, 64)
        mock_context.tensors.load.return_value = mock_latents

        estimation_path = "invokeai.app.invocations.z_image_latents_to_image.estimate_vae_working_memory_flux"

        with patch(estimation_path) as mock_estimate:
            expected_memory = 1024 * 1024 * 500  # 500MB
            mock_estimate.return_value = expected_memory

            # Mock VAE decode to avoid actual computation
            if vae_type == FluxAutoEncoder:
                mock_vae.decode.return_value = torch.zeros(1, 3, 512, 512)
            else:
                mock_vae.decode.return_value = (torch.zeros(1, 3, 512, 512),)

            # Mock image save
            mock_image_dto = MagicMock()
            mock_context.images.save.return_value = mock_image_dto

            # Import and create invocation using model_construct to bypass validation
            from invokeai.app.invocations.z_image_latents_to_image import ZImageLatentsToImageInvocation

            invocation = ZImageLatentsToImageInvocation.model_construct(
                latents=MagicMock(latents_name="test_latents"),
                vae=MagicMock(vae=MagicMock(), seamless_axes=["x", "y"]),
            )

            try:
                invocation.invoke(mock_context)
            except Exception:
                # We expect some errors due to mocking, but we just want to verify the working memory was requested
                pass

            # Verify that working memory estimation was called
            mock_estimate.assert_called_once()
            # Verify that model_on_device was called with the estimated working memory
            mock_vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected_memory)

    @pytest.mark.parametrize("vae_type", [AutoencoderKL, FluxAutoEncoder])
    def test_z_image_image_to_latents_requests_working_memory(self, vae_type):
        """Test that ZImageImageToLatentsInvocation estimates and requests working memory."""
        # Create mock VAE
        mock_vae = MagicMock(spec=vae_type)

        # Only set config for AutoencoderKL (FluxAutoEncoder doesn't use config)
        if vae_type == AutoencoderKL:
            mock_vae.config.scaling_factor = 1.0
            mock_vae.config.shift_factor = None

        # Create mock parameter for dtype detection
        mock_param = torch.zeros(1)
        mock_vae.parameters.return_value = iter([mock_param])

        # Create mock vae_info
        mock_vae_info = MagicMock()
        mock_vae_info.model = mock_vae

        # Create mock context manager return value
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=(None, mock_vae))
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_vae_info.model_on_device = MagicMock(return_value=mock_cm)

        # Mock image tensor
        mock_image_tensor = torch.zeros(1, 3, 512, 512)

<<<<<<< HEAD
        # Mock the appropriate estimation function
        if vae_type == FluxAutoEncoder:
            estimation_path = "invokeai.app.invocations.z_image_image_to_latents.estimate_vae_working_memory_flux"
        else:
            estimation_path = "invokeai.app.invocations.z_image_image_to_latents.estimate_vae_working_memory_sd3"
=======
        # Mock the estimation function
        estimation_path = "invokeai.app.invocations.z_image_image_to_latents.estimate_vae_working_memory_flux"
>>>>>>> main

        with patch(estimation_path) as mock_estimate:
            expected_memory = 1024 * 1024 * 250  # 250MB
            mock_estimate.return_value = expected_memory

            # Mock VAE encode to avoid actual computation
            if vae_type == FluxAutoEncoder:
                mock_vae.encode.return_value = torch.zeros(1, 16, 64, 64)
            else:
                mock_latent_dist = MagicMock()
                mock_latent_dist.sample.return_value = torch.zeros(1, 16, 64, 64)
                mock_encode_result = MagicMock()
                mock_encode_result.latent_dist = mock_latent_dist
                mock_vae.encode.return_value = mock_encode_result

            # Call the static method directly
            try:
                ZImageImageToLatentsInvocation.vae_encode(mock_vae_info, mock_image_tensor)
            except Exception:
                # We expect some errors due to mocking, but we just want to verify the working memory was requested
                pass

            # Verify that working memory estimation was called
            mock_estimate.assert_called_once()
            # Verify that model_on_device was called with the estimated working memory
            mock_vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected_memory)
