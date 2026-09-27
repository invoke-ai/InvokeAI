"""Tests for the Anima VAE invocations: which VAEs they accept, working-memory estimation, the
tiled-decode decision, and the tiled retry on out-of-memory."""

import math
from unittest.mock import MagicMock, patch

import accelerate
import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.vae.anima_image_to_latents import AnimaImageToLatentsInvocation
from invokeai.app.invocations.vae.anima_latents_to_image import (
    ANIMA_VAE_TILE_SIZE,
    ANIMA_VAE_TILE_STRIDE,
    AnimaLatentsToImageInvocation,
)
from invokeai.backend.model_manager.load.model_loaders.vae import _WAN_TI2V_5B_VAE_CONFIG
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_anima

# The two classes the Wan 2.1 VAE loads as: the original-layout file as AutoencoderKLWan, the
# diffusers-layout Qwen-Image export as AutoencoderKLQwenImage.
WAN21_VAE_LAYOUTS = [AutoencoderKLWan, AutoencoderKLQwenImage]


def _mock_vae(vae_class: type = AutoencoderKLWan, dtype: torch.dtype = torch.float16) -> MagicMock:
    vae = MagicMock(spec=vae_class)
    param = torch.zeros(1, dtype=dtype)
    # Return a fresh iterator on every call so the estimator can be called repeatedly.
    vae.parameters.side_effect = lambda: iter([param])
    # `patch_qwen_image_vae_tiling` records and restores these; they are set in `__init__` rather
    # than on the class, so a spec'd mock does not have them. The stock values keep any assertion
    # against them reading as real numbers.
    vae.use_tiling = False
    vae.tile_sample_min_height = 256
    vae.tile_sample_min_width = 256
    vae.tile_sample_stride_height = 192
    vae.tile_sample_stride_width = 192
    # The Wan 2.1 geometry, which the nodes check before using a Wan VAE.
    vae.config.z_dim = 16
    vae.config.patch_size = None
    vae.config.scale_factor_spatial = 8
    vae.config.latents_mean = [0.0] * 16
    vae.config.latents_std = [1.0] * 16
    return vae


def _mock_vae_info(vae) -> MagicMock:
    vae_info = MagicMock()
    vae_info.model = vae
    # The invocation places latents on the VAE's intended compute device (see #9373), so this
    # must be a real torch.device rather than a MagicMock for `latents.to(device=...)` to work.
    vae_info.compute_device = torch.device("cpu")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=(None, vae))
    cm.__exit__ = MagicMock(return_value=None)
    vae_info.model_on_device.return_value = cm
    return vae_info


class TestEstimateVaeWorkingMemoryAnima:
    @pytest.mark.parametrize("vae_class", WAN21_VAE_LAYOUTS)
    def test_untiled_decode_uses_decode_constant_and_scales_latent_dims(self, vae_class):
        latents = torch.zeros(1, 16, 1, 128, 128)
        result = estimate_vae_working_memory_anima(
            operation="decode", image_tensor=latents, vae=_mock_vae(vae_class, torch.float16), tile_size=None
        )
        out_h = out_w = 128 * LATENT_SCALE_FACTOR
        assert result == int(out_h * out_w * 2 * 2900)

    def test_untiled_encode_uses_encode_constant_and_pixel_dims(self):
        image = torch.zeros(1, 3, 1, 1024, 1024)
        result = estimate_vae_working_memory_anima(
            operation="encode", image_tensor=image, vae=_mock_vae(dtype=torch.float16), tile_size=None
        )
        assert result == int(1024 * 1024 * 2 * 1450)

    @pytest.mark.parametrize("latent_hw", [(64, 64), (160, 160)])
    def test_tiled_decode_estimate_is_independent_of_image_size(self, latent_hw):
        latents = torch.zeros(1, 16, 1, *latent_hw)
        result = estimate_vae_working_memory_anima(
            operation="decode", image_tensor=latents, vae=_mock_vae(dtype=torch.float16), tile_size=512
        )
        assert result == int(512 * 512 * 2 * 2900 * 1.25)

    def test_estimate_scales_with_element_size(self):
        latents = torch.zeros(1, 16, 1, 128, 128)
        fp16 = estimate_vae_working_memory_anima(
            operation="decode", image_tensor=latents, vae=_mock_vae(dtype=torch.float16), tile_size=None
        )
        fp32 = estimate_vae_working_memory_anima(
            operation="decode", image_tensor=latents, vae=_mock_vae(dtype=torch.float32), tile_size=None
        )
        assert fp32 == 2 * fp16


class TestTheTiledEncodeResidual:
    """A tiled encode does not bound the frame it slices from, nor the moments it assembles.

    Without those terms the estimate is flat in the image size while the measurement grows about
    4 bytes per pixel: measured on a 4090 it fell from 1.07x headroom at 1024px to 0.69x at 4096px,
    i.e. the cache promised less than the encode used -- and this node has no tiled retry, so that
    is a failed generation rather than a slower one.
    """

    def test_the_tiled_encode_estimate_grows_with_the_frame(self):
        vae = _mock_vae()
        estimates = [
            estimate_vae_working_memory_anima("encode", torch.zeros(1, 3, edge, edge), vae, tile_size=256)
            for edge in (1024, 4096)
        ]
        assert estimates[1] > estimates[0], "the un-bounded part of a tiled encode is not priced"

    def test_the_tiled_decode_estimate_is_left_flat(self):
        """Stated rather than assumed: the decode has the same kind of un-bounded assembly and does
        not price it either. That is pre-existing -- this node has tiled since it was written -- and
        is deliberately not changed in a diff about the encode."""
        vae = _mock_vae()
        estimates = [
            estimate_vae_working_memory_anima("decode", torch.zeros(1, 16, edge // 8, edge // 8), vae, tile_size=256)
            for edge in (1024, 4096)
        ]
        assert estimates[0] == estimates[1]


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


def _build_decode_mocks(latents: torch.Tensor, decoded: torch.Tensor, vae_class: type = AutoencoderKLWan):
    """Mock the Anima VAE decode path: a spec'd VAE, its LoadedModel wrapper, and the invocation
    context, wired so `AnimaLatentsToImageInvocation.invoke` runs end-to-end on CPU."""
    vae = _mock_vae(vae_class, torch.float32)
    vae.decode.return_value = (decoded,)
    vae_info = _mock_vae_info(vae)

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


class TestAnimaAcceptsBothWan21VaeLayouts:
    """The picker offers a `qwen-image` VAE for Anima. Its diffusers-layout file loads as
    AutoencoderKLQwenImage, which the nodes used to refuse with a TypeError; it is the same network
    on the same weights with the same latent statistics, so it must encode and decode like the
    original-layout AutoencoderKLWan."""

    @pytest.mark.parametrize("vae_class", WAN21_VAE_LAYOUTS)
    def test_decode_denormalises_with_the_vae_statistics(self, vae_class):
        decoded = torch.zeros(1, 3, 1, 32, 32)
        vae, _, context = _build_decode_mocks(
            latents=torch.full((1, 16, 4, 4), 0.5), decoded=decoded, vae_class=vae_class
        )
        vae.config.latents_mean = [1.0] * 16
        vae.config.latents_std = [2.0] * 16

        with patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")):
            result = _build_l2i_invocation().invoke(context)

        vae.decode.assert_called_once()
        vae_input = vae.decode.call_args.args[0]
        assert vae_input.shape == (1, 16, 1, 4, 4)
        # latents * std + mean = 0.5 * 2 + 1
        assert torch.all(vae_input == 2.0)
        assert result.width == 32

    @pytest.mark.parametrize("vae_class", WAN21_VAE_LAYOUTS)
    def test_encode_normalises_with_the_vae_statistics(self, vae_class):
        vae = _mock_vae(vae_class, torch.float32)
        vae.config.latents_mean = [1.0] * 16
        vae.config.latents_std = [2.0] * 16
        posterior = MagicMock()
        posterior.sample.return_value = torch.full((1, 16, 1, 4, 4), 3.0)
        vae.encode.return_value = (posterior,)

        with patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")):
            latents = AnimaImageToLatentsInvocation.vae_encode(
                vae_info=_mock_vae_info(vae), image_tensor=torch.zeros(1, 3, 32, 32)
            )

        assert vae.encode.call_args.args[0].shape == (1, 3, 1, 32, 32)
        # (latents - mean) / std = (3 - 1) / 2
        assert latents.shape == (1, 16, 4, 4)
        assert torch.all(latents == 1.0)


def _flux_vae_info() -> MagicMock:
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

    vae_info = MagicMock()
    # The FLUX.1 autoencoder is loaded as a diffusers `AutoencoderKL`, which is *not* one of the two
    # classes `as_qwen_image_vae` accepts (`AutoencoderKLQwenImage`, `AutoencoderKLWan`). So the
    # refusal is still a class check here -- what changed is which class the FLUX VAE presents as.
    vae_info.model = MagicMock(spec=AutoencoderKL)
    return vae_info


def _flux_decode_context(vae_info: MagicMock) -> MagicMock:
    context = MagicMock()
    context.models.load.return_value = vae_info
    context.tensors.load.return_value = torch.zeros(1, 16, 64, 64)
    return context


class TestAnimaRefusesAForeignDecoder:
    """A FLUX VAE has Anima's channel count and compression but a different basis.

    It used to be accepted -- the node had a branch for that class, the picker offered it, and
    the decode returned a magenta moire with the subject barely visible while the run reported
    success. Measured against the correct decode of the same latent: 8.67 dB PSNR, mean absolute
    error 84 of 255. Nothing downstream can tell such an image from an intended one, so the node
    refuses instead of guessing."""

    def test_a_flux_vae_is_refused_before_any_decode(self):
        vae_info = _flux_vae_info()

        with pytest.raises(TypeError, match="16-channel Wan 2.1 latent space"):
            _build_l2i_invocation().invoke(_flux_decode_context(vae_info))

        # Refused before the model is placed on a device: a wrong decode costs VRAM and time
        # before it produces the wrong image.
        vae_info.model_on_device.assert_not_called()

    def test_the_message_names_what_to_choose_instead(self):
        with pytest.raises(TypeError) as excinfo:
            _build_l2i_invocation().invoke(_flux_decode_context(_flux_vae_info()))

        message = str(excinfo.value)
        for base in ("'anima'", "'qwen-image'", "'wan'"):
            assert base in message, message

    def test_a_flux_vae_is_refused_before_any_encode_with_the_same_advice(self):
        vae_info = _flux_vae_info()

        with pytest.raises(TypeError, match="16-channel Wan 2.1 latent space") as excinfo:
            AnimaImageToLatentsInvocation.vae_encode(vae_info=vae_info, image_tensor=torch.zeros(1, 3, 32, 32))

        message = str(excinfo.value)
        for base in ("'anima'", "'qwen-image'", "'wan'"):
            assert base in message, message
        vae_info.model_on_device.assert_not_called()


class TestAnimaRefusesTheWan22Vae:
    """Wan 2.2's TI2V VAE is also an AutoencoderKLWan, but a 48-channel, patchified latent space.
    The class check alone let it through to a device, where the 16-entry normalisation broke on it."""

    @pytest.fixture
    def wan22_vae_info(self) -> MagicMock:
        with accelerate.init_empty_weights():
            vae = AutoencoderKLWan(**_WAN_TI2V_5B_VAE_CONFIG)
        vae_info = MagicMock()
        vae_info.model = vae
        return vae_info

    def test_decode_refuses_it_before_any_decode(self, wan22_vae_info):
        context = MagicMock()
        context.models.load.return_value = wan22_vae_info
        context.tensors.load.return_value = torch.zeros(1, 48, 16, 16)

        with pytest.raises(ValueError, match="z_dim=48"):
            _build_l2i_invocation().invoke(context)

        wan22_vae_info.model_on_device.assert_not_called()

    def test_encode_refuses_it_before_any_encode(self, wan22_vae_info):
        with pytest.raises(ValueError, match="z_dim=48"):
            AnimaImageToLatentsInvocation.vae_encode(vae_info=wan22_vae_info, image_tensor=torch.zeros(1, 3, 32, 32))

        wan22_vae_info.model_on_device.assert_not_called()


class TestAnimaLatentsToImageOomFallback:
    @pytest.mark.parametrize(
        "oom_error",
        [
            torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 5.9 GiB"),
            RuntimeError("CUDA error: out of memory"),
            RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"),
            # XPU reports exhaustion through the Level Zero/UR runtime as a plain RuntimeError.
            # Note the underscores: these do not contain the words "out of memory".
            RuntimeError("Native API failed. Native API returns: UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY"),
            RuntimeError("UR error: UR_RESULT_ERROR_OUT_OF_HOST_MEMORY"),
            RuntimeError("ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY"),
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
        # The retry takes noticeably longer than the failed attempt; the user is told why.
        context.util.signal_progress.assert_any_call("VAE decode ran out of memory, retrying tiled")
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

        estimation_path = "invokeai.app.invocations.vae.anima_latents_to_image.estimate_vae_working_memory_anima"
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
        vae = _mock_vae(dtype=torch.float32)
        mock_dist = MagicMock()
        mock_dist.sample.return_value = torch.zeros(1, 16, 1, 4, 4)
        vae.encode.return_value = (mock_dist,)
        vae_info = _mock_vae_info(vae)

        estimation_path = "invokeai.app.invocations.vae.anima_image_to_latents.estimate_vae_working_memory_anima"
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
