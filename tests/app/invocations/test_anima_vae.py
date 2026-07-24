"""Tests for the Anima VAE invocations: working-memory estimation, the tiled-decode
decision, and the tiled retry on out-of-memory."""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan

from invokeai.app.invocations.anima_image_to_latents import AnimaImageToLatentsInvocation
from invokeai.app.invocations.anima_latents_to_image import (
    ANIMA_VAE_TILE_SIZE,
    ANIMA_VAE_TILE_STRIDE,
    AnimaLatentsToImageInvocation,
)
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_anima


def _mock_wan_vae(dtype: torch.dtype = torch.float16) -> MagicMock:
    vae = MagicMock(spec=AutoencoderKLWan)
    param = torch.zeros(1, dtype=dtype)
    # Return a fresh iterator on every call so the estimator can be called repeatedly.
    vae.parameters.side_effect = lambda: iter([param])
    return vae


class TestEstimateVaeWorkingMemoryAnima:
    def test_untiled_decode_uses_decode_constant_and_scales_latent_dims(self):
        latents = torch.zeros(1, 16, 1, 128, 128)
        result = estimate_vae_working_memory_anima(
            operation="decode", image_tensor=latents, vae=_mock_wan_vae(torch.float16), tile_size=None
        )
        out_h = out_w = 128 * LATENT_SCALE_FACTOR
        assert result == int(out_h * out_w * 2 * 2900)

    def test_untiled_encode_uses_encode_constant_and_pixel_dims(self):
        image = torch.zeros(1, 3, 1, 1024, 1024)
        result = estimate_vae_working_memory_anima(
            operation="encode", image_tensor=image, vae=_mock_wan_vae(torch.float16), tile_size=None
        )
        assert result == int(1024 * 1024 * 2 * 1450)

    @pytest.mark.parametrize("latent_hw", [(64, 64), (160, 160)])
    def test_tiled_decode_estimate_is_independent_of_image_size(self, latent_hw):
        latents = torch.zeros(1, 16, 1, *latent_hw)
        result = estimate_vae_working_memory_anima(
            operation="decode", image_tensor=latents, vae=_mock_wan_vae(torch.float16), tile_size=512
        )
        assert result == int(512 * 512 * 2 * 2900 * 1.25)

    def test_estimate_scales_with_element_size(self):
        latents = torch.zeros(1, 16, 1, 128, 128)
        fp16 = estimate_vae_working_memory_anima(
            operation="decode", image_tensor=latents, vae=_mock_wan_vae(torch.float16), tile_size=None
        )
        fp32 = estimate_vae_working_memory_anima(
            operation="decode", image_tensor=latents, vae=_mock_wan_vae(torch.float32), tile_size=None
        )
        assert fp32 == 2 * fp16


class TestUseTiledDecode:
    @pytest.mark.parametrize("device_type", ["cpu", "mps"])
    def test_non_cuda_never_tiles(self, device_type):
        assert AnimaLatentsToImageInvocation._use_tiled_decode(torch.device(device_type), 10**12) is False

    def test_cuda_flips_at_70_percent_of_total_vram(self):
        total_vram = 8 * 2**30
        boundary = 0.7 * total_vram
        device = torch.device("cuda")
        with patch("torch.cuda.get_device_properties", return_value=MagicMock(total_memory=total_vram)) as mock_props:
            assert AnimaLatentsToImageInvocation._use_tiled_decode(device, math.floor(boundary)) is False
            assert AnimaLatentsToImageInvocation._use_tiled_decode(device, math.ceil(boundary) + 1) is True
            mock_props.assert_called_with(device)


def _build_decode_mocks(latents: torch.Tensor, decoded: torch.Tensor):
    """Mock the Wan VAE decode path: a spec'd AutoencoderKLWan, its LoadedModel wrapper, and the
    invocation context, wired so `AnimaLatentsToImageInvocation.invoke` runs end-to-end on CPU."""
    vae = _mock_wan_vae(torch.float32)
    vae.config.latents_mean = [0.0] * 16
    vae.config.latents_std = [1.0] * 16
    vae.decode.return_value = (decoded,)

    vae_info = MagicMock()
    vae_info.model = vae
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device.return_value = cm

    context = MagicMock()
    context.models.load.return_value = vae_info
    context.tensors.load.return_value = latents
    image_dto = MagicMock()
    image_dto.image_name = "test.png"
    image_dto.width = decoded.shape[-1]
    image_dto.height = decoded.shape[-2]
    context.images.save.return_value = image_dto

    return vae, vae_info, context


def _build_l2i_invocation() -> AnimaLatentsToImageInvocation:
    return AnimaLatentsToImageInvocation.model_construct(
        latents=MagicMock(latents_name="test_latents"),
        vae=MagicMock(vae=MagicMock()),
    )


class TestAnimaLatentsToImageOomFallback:
    @pytest.mark.parametrize(
        "oom_error",
        [
            torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 5.9 GiB"),
            RuntimeError("CUDA error: out of memory"),
            RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"),
        ],
    )
    def test_untiled_decode_oom_retries_with_tiling(self, oom_error):
        decoded = torch.zeros(1, 3, 1, 64, 64)
        vae, _, context = _build_decode_mocks(latents=torch.zeros(1, 16, 32, 32), decoded=decoded)
        vae.decode.side_effect = [oom_error, (decoded,)]

        with patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")):
            result = _build_l2i_invocation().invoke(context)

        assert vae.decode.call_count == 2
        vae.enable_tiling.assert_called_once_with(
            tile_sample_min_height=ANIMA_VAE_TILE_SIZE,
            tile_sample_min_width=ANIMA_VAE_TILE_SIZE,
            tile_sample_stride_height=ANIMA_VAE_TILE_STRIDE,
            tile_sample_stride_width=ANIMA_VAE_TILE_STRIDE,
        )
        assert result.width == 64

    def test_non_oom_runtime_error_propagates_without_retry(self):
        vae, _, context = _build_decode_mocks(latents=torch.zeros(1, 16, 32, 32), decoded=torch.zeros(1, 3, 1, 64, 64))
        vae.decode.side_effect = RuntimeError("Input type (float) and weight type (half) should be the same")

        with patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")):
            with pytest.raises(RuntimeError, match="weight type"):
                _build_l2i_invocation().invoke(context)

        assert vae.decode.call_count == 1
        vae.enable_tiling.assert_not_called()

    def test_oom_while_already_tiled_reraises(self):
        vae, _, context = _build_decode_mocks(latents=torch.zeros(1, 16, 32, 32), decoded=torch.zeros(1, 3, 1, 64, 64))
        vae.decode.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")

        with (
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
            patch.object(AnimaLatentsToImageInvocation, "_use_tiled_decode", return_value=True),
        ):
            with pytest.raises(torch.cuda.OutOfMemoryError):
                _build_l2i_invocation().invoke(context)

        # No second attempt: the initial enable_tiling is the only one, and decode is not retried.
        assert vae.decode.call_count == 1
        vae.enable_tiling.assert_called_once()

    def test_decode_requests_estimated_working_memory(self):
        decoded = torch.zeros(1, 3, 1, 64, 64)
        vae, vae_info, context = _build_decode_mocks(latents=torch.zeros(1, 16, 32, 32), decoded=decoded)

        estimation_path = "invokeai.app.invocations.anima_latents_to_image.estimate_vae_working_memory_anima"
        expected_memory = 1024 * 1024 * 500
        with (
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
            patch(estimation_path, return_value=expected_memory) as mock_estimate,
        ):
            _build_l2i_invocation().invoke(context)

        # Called once for the full-decode estimate (tiling decision) and once for the actual request.
        assert mock_estimate.call_count == 2
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected_memory)


class TestAnimaImageToLatentsEncode:
    def test_encode_disables_tiling_and_requests_working_memory(self):
        vae = _mock_wan_vae(torch.float32)
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        mock_dist = MagicMock()
        mock_dist.sample.return_value = torch.zeros(1, 16, 1, 4, 4)
        vae.encode.return_value = (mock_dist,)

        vae_info = MagicMock()
        vae_info.model = vae
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=(None, vae))
        cm.__exit__ = MagicMock(return_value=None)
        vae_info.model_on_device.return_value = cm

        estimation_path = "invokeai.app.invocations.anima_image_to_latents.estimate_vae_working_memory_anima"
        expected_memory = 1024 * 1024 * 250
        with (
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
            patch(estimation_path, return_value=expected_memory) as mock_estimate,
        ):
            latents = AnimaImageToLatentsInvocation.vae_encode(
                vae_info=vae_info, image_tensor=torch.zeros(1, 3, 32, 32)
            )

        # The shared cached VAE may have tiling enabled from a previous decode; encode must reset it.
        vae.disable_tiling.assert_called_once()
        vae.enable_tiling.assert_not_called()
        mock_estimate.assert_called_once()
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected_memory)
        assert latents.shape == (1, 16, 4, 4)
