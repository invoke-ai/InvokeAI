"""Wan VAE invocations: working-memory estimation and cpu_only device handling."""

from unittest.mock import MagicMock, patch

import PIL.Image
import torch
from diffusers.models.autoencoders import AutoencoderKLWan

from invokeai.app.invocations.wan_image_to_latents import WanImageToLatentsInvocation
from invokeai.app.invocations.wan_latents_to_image import WanLatentsToImageInvocation
from invokeai.app.invocations.wan_latents_to_video import WanLatentsToVideoInvocation
from invokeai.app.invocations.wan_ref_image_encoder import WanRefImageEncoderInvocation
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_wan


def _mock_wan_vae(
    z_dim: int = 16, spatial_scale: int = 8, temporal_scale: int = 4, dtype: torch.dtype = torch.float16
) -> MagicMock:
    vae = MagicMock(spec=AutoencoderKLWan)
    param = torch.zeros(1, dtype=dtype)
    # parameters() is consumed several times (estimator, dtype probe, effective device).
    vae.parameters.side_effect = lambda: iter([param])
    vae.config = MagicMock()
    vae.config.z_dim = z_dim
    vae.config.scale_factor_spatial = spatial_scale
    vae.config.scale_factor_temporal = temporal_scale
    vae.config.latents_mean = [0.0] * z_dim
    vae.config.latents_std = [1.0] * z_dim
    vae.tile_sample_min_height = 256
    return vae


def _mock_vae_info(vae: MagicMock, cpu_only: bool | None = None) -> MagicMock:
    vae_info = MagicMock()
    vae_info.model = vae
    vae_info.config = MagicMock()
    vae_info.config.cpu_only = cpu_only
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device = MagicMock(return_value=cm)
    return vae_info


class TestEstimateVaeWorkingMemoryWan:
    """Lock in the generalized Wan estimator: per-frame conv working set plus the full
    RGB clip that stays resident on the execution device."""

    def test_decode_single_frame(self):
        vae = _mock_wan_vae()
        est = estimate_vae_working_memory_wan(
            operation="decode", vae=vae, pixel_height=256, pixel_width=256, pixel_frames=1
        )
        area_bytes = 256 * 256 * 2
        # Decode counts the clip twice: torch.cat frame accumulation transiently holds
        # the clip and its copy at peak.
        assert est == area_bytes * 2900 + 2 * 3 * 1 * area_bytes

    def test_encode_uses_half_the_decode_constant(self):
        vae = _mock_wan_vae()
        est = estimate_vae_working_memory_wan(
            operation="encode", vae=vae, pixel_height=256, pixel_width=256, pixel_frames=1
        )
        area_bytes = 256 * 256 * 2
        # Encode consumes the input clip without duplicating it — one copy only.
        assert est == area_bytes * 1450 + 3 * 1 * area_bytes

    def test_additional_frames_add_only_clip_bytes(self):
        """The Wan VAE is causal (frame-at-a-time with cached features), so extra frames
        grow the resident clip (2 copies at decode peak), not the conv working set."""
        vae = _mock_wan_vae()
        one = estimate_vae_working_memory_wan(
            operation="decode", vae=vae, pixel_height=128, pixel_width=128, pixel_frames=1
        )
        many = estimate_vae_working_memory_wan(
            operation="decode", vae=vae, pixel_height=128, pixel_width=128, pixel_frames=81
        )
        assert many - one == 2 * 3 * 80 * 128 * 128 * 2

    def test_tile_size_bounds_the_per_frame_term(self):
        vae = _mock_wan_vae()
        tiled = estimate_vae_working_memory_wan(
            operation="decode", vae=vae, pixel_height=1920, pixel_width=1080, pixel_frames=17, tile_size=256
        )
        expected = int(256 * 256 * 2 * 2900 * 1.25 + 2 * 3 * 17 * 1920 * 1080 * 2)
        assert tiled == expected
        full = estimate_vae_working_memory_wan(
            operation="decode", vae=vae, pixel_height=1920, pixel_width=1080, pixel_frames=17
        )
        assert tiled < full


class TestWanInvocationsRequestWorkingMemory:
    """Every Wan VAE path must reserve its estimated working memory with the model cache."""

    def test_latents_to_image_requests_decode_memory(self):
        vae = _mock_wan_vae()
        vae_info = _mock_vae_info(vae)
        mock_context = MagicMock()
        mock_context.models.load.return_value = vae_info
        mock_context.tensors.load.return_value = torch.zeros(1, 16, 64, 64)

        with patch("invokeai.app.invocations.wan_latents_to_image.estimate_vae_working_memory_wan") as mock_estimate:
            mock_estimate.return_value = 1234
            invocation = WanLatentsToImageInvocation.model_construct(
                latents=MagicMock(latents_name="l"), vae=MagicMock(vae=MagicMock())
            )
            try:
                invocation.invoke(mock_context)
            except Exception:
                pass  # Downstream image math fails under mocking; the reservation is what matters.

        assert mock_estimate.call_args.kwargs["operation"] == "decode"
        # Latent 64x64 at the standard 8x spatial scale -> 512x512 pixels.
        assert mock_estimate.call_args.kwargs["pixel_height"] == 512
        assert mock_estimate.call_args.kwargs["pixel_width"] == 512
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=1234)

    def test_latents_to_image_uses_config_spatial_scale_for_ti2v(self):
        vae = _mock_wan_vae(z_dim=48, spatial_scale=16)
        vae_info = _mock_vae_info(vae)
        mock_context = MagicMock()
        mock_context.models.load.return_value = vae_info
        mock_context.tensors.load.return_value = torch.zeros(1, 48, 32, 32)

        with patch("invokeai.app.invocations.wan_latents_to_image.estimate_vae_working_memory_wan") as mock_estimate:
            mock_estimate.return_value = 1
            invocation = WanLatentsToImageInvocation.model_construct(
                latents=MagicMock(latents_name="l"), vae=MagicMock(vae=MagicMock())
            )
            try:
                invocation.invoke(mock_context)
            except Exception:
                pass

        # TI2V-5B VAE compresses 16x spatially -> 32 latent px = 512 pixel px.
        assert mock_estimate.call_args.kwargs["pixel_height"] == 512

    def test_image_to_latents_requests_encode_memory(self):
        vae = _mock_wan_vae()
        vae_info = _mock_vae_info(vae)

        with patch("invokeai.app.invocations.wan_image_to_latents.estimate_vae_working_memory_wan") as mock_estimate:
            mock_estimate.return_value = 4321
            try:
                WanImageToLatentsInvocation.vae_encode(vae_info, torch.zeros(1, 3, 512, 512))
            except Exception:
                pass

        assert mock_estimate.call_args.kwargs["operation"] == "encode"
        assert mock_estimate.call_args.kwargs["pixel_frames"] == 1
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=4321)

    def test_ref_image_encoder_requests_encode_memory(self):
        vae = _mock_wan_vae()
        vae_info = _mock_vae_info(vae)
        mock_context = MagicMock()
        mock_context.models.load.return_value = vae_info
        mock_context.images.get_pil.return_value = PIL.Image.new("RGB", (32, 32))
        mock_context.tensors.save.return_value = "t"

        with (
            patch("invokeai.app.invocations.wan_ref_image_encoder.estimate_vae_working_memory_wan") as mock_estimate,
            patch(
                "invokeai.app.invocations.wan_ref_image_encoder.encode_reference_image_to_condition",
                return_value=torch.zeros(1, 20, 1, 4, 4),
            ),
        ):
            mock_estimate.return_value = 999
            invocation = WanRefImageEncoderInvocation.model_construct(
                image=MagicMock(image_name="i"),
                vae=MagicMock(vae=MagicMock()),
                width=32,
                height=32,
                num_frames=1,
                end_image=None,
            )
            invocation.invoke(mock_context)

        assert mock_estimate.call_args.kwargs["operation"] == "encode"
        assert mock_estimate.call_args.kwargs["pixel_frames"] == 1
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=999)

    def _video_context(self, vae_info: MagicMock, t_lat: int = 5) -> MagicMock:
        mock_context = MagicMock()
        mock_context.models.load.return_value = vae_info
        mock_context.tensors.load.return_value = torch.zeros(1, 16, t_lat, 32, 32)
        mock_context.util.is_canceled.return_value = False
        return mock_context

    def test_latents_to_video_requests_decode_memory_for_all_frames(self):
        vae = _mock_wan_vae()
        vae_info = _mock_vae_info(vae)
        mock_context = self._video_context(vae_info)

        with patch("invokeai.app.invocations.wan_latents_to_video.estimate_vae_working_memory_wan") as mock_estimate:
            mock_estimate.return_value = 5678
            invocation = WanLatentsToVideoInvocation.model_construct(
                latents=MagicMock(latents_name="l"), vae=MagicMock(vae=MagicMock()), fps=16
            )
            try:
                invocation.invoke(mock_context)
            except Exception:
                pass

        assert mock_estimate.call_args.kwargs["operation"] == "decode"
        # 5 latent frames -> (5-1)*4+1 = 17 pixel frames at 32*8 = 256px.
        assert mock_estimate.call_args.kwargs["pixel_frames"] == 17
        assert mock_estimate.call_args.kwargs["pixel_height"] == 256
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=5678)

    def test_latents_to_video_falls_back_to_tiling_when_estimate_exceeds_vram(self):
        vae = _mock_wan_vae()
        vae_info = _mock_vae_info(vae)
        mock_context = self._video_context(vae_info)

        with (
            patch(
                "invokeai.app.invocations.wan_latents_to_video.estimate_vae_working_memory_wan",
                side_effect=[100 * 2**30, 4 * 2**30],
            ) as mock_estimate,
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cuda")),
            patch.object(TorchDevice, "empty_cache"),
            patch("torch.cuda.get_device_properties", return_value=MagicMock(total_memory=8 * 2**30)),
        ):
            invocation = WanLatentsToVideoInvocation.model_construct(
                latents=MagicMock(latents_name="l"), vae=MagicMock(vae=MagicMock()), fps=16
            )
            try:
                invocation.invoke(mock_context)
            except Exception:
                pass

        assert mock_estimate.call_count == 2
        assert mock_estimate.call_args_list[1].kwargs["tile_size"] == 256
        vae.enable_tiling.assert_called_once()
        vae.disable_tiling.assert_called_once()
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=4 * 2**30)

    def test_latents_to_video_skips_tiling_for_cpu_only_vae(self):
        """A cpu_only VAE runs in system RAM; VRAM-based tiling must not kick in."""
        vae = _mock_wan_vae()
        vae_info = _mock_vae_info(vae, cpu_only=True)
        mock_context = self._video_context(vae_info)

        with (
            patch(
                "invokeai.app.invocations.wan_latents_to_video.estimate_vae_working_memory_wan",
                return_value=100 * 2**30,
            ) as mock_estimate,
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cuda")),
            patch.object(TorchDevice, "empty_cache"),
        ):
            invocation = WanLatentsToVideoInvocation.model_construct(
                latents=MagicMock(latents_name="l"), vae=MagicMock(vae=MagicMock()), fps=16
            )
            try:
                invocation.invoke(mock_context)
            except Exception:
                pass

        assert mock_estimate.call_count == 1
        vae.enable_tiling.assert_not_called()


class TestWanCpuOnlyDeviceHandling:
    """With a cpu_only VAE, inputs must go to the VAE's device — not the globally
    selected accelerator (which would crash every Wan VAE invocation on GPU hosts)."""

    def test_latents_to_image_moves_latents_to_vae_device(self):
        vae = _mock_wan_vae(dtype=torch.float32)
        seen_devices: list[torch.device] = []

        def _decode(latents, return_dict):
            seen_devices.append(latents.device)
            return (torch.zeros(1, 3, 1, 8, 8),)

        vae.decode.side_effect = _decode
        vae_info = _mock_vae_info(vae, cpu_only=True)
        mock_context = MagicMock()
        mock_context.models.load.return_value = vae_info
        mock_context.tensors.load.return_value = torch.zeros(1, 16, 1, 1)

        # Poison the global device: if the invocation still routed through
        # choose_torch_device, the .to() below would produce a meta tensor and fail
        # the device assertion.
        with (
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("meta")),
            patch.object(TorchDevice, "empty_cache"),
        ):
            invocation = WanLatentsToImageInvocation.model_construct(
                latents=MagicMock(latents_name="l"), vae=MagicMock(vae=MagicMock())
            )
            try:
                invocation.invoke(mock_context)
            except Exception:
                pass

        assert seen_devices == [torch.device("cpu")]

    def test_image_to_latents_moves_image_to_vae_device(self):
        vae = _mock_wan_vae(dtype=torch.float32)
        seen_devices: list[torch.device] = []

        def _encode(pixel, return_dict):
            seen_devices.append(pixel.device)
            raise RuntimeError("stop here")

        vae.encode.side_effect = _encode
        vae_info = _mock_vae_info(vae, cpu_only=True)

        with patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("meta")):
            try:
                WanImageToLatentsInvocation.vae_encode(vae_info, torch.zeros(1, 3, 64, 64))
            except Exception:
                pass

        assert seen_devices == [torch.device("cpu")]
