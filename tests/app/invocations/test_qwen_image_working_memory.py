"""Test that Qwen Image VAE invocations properly estimate and request working memory."""

import math
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

from invokeai.app.invocations.qwen_image_image_to_latents import QwenImageImageToLatentsInvocation
from invokeai.app.invocations.qwen_image_latents_to_image import QwenImageLatentsToImageInvocation
from invokeai.backend.krea2.vae_compat import (
    QWEN_IMAGE_VAE_DEFAULT_TILE_SIZE,
    patch_qwen_image_vae_tiling,
)
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

    def _mock_fp16_vae(self):
        mock_vae = MagicMock(spec=AutoencoderKLQwenImage)
        mock_vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.float16)])  # element_size == 2
        return mock_vae

    @pytest.mark.parametrize("operation, image_copies", [("decode", 5), ("encode", 1)])
    def test_tiled_estimate_is_one_tile_plus_the_pixel_buffers(self, operation, image_copies):
        """The tiled branch must budget the per-tile conv working set *and* the pixel-space buffers.

        The tile term is constant in the output size, so the pixel term is the one that decides whether
        the estimate still holds at the resolutions tiling exists for. `tiled_decode` keeps several full
        frames alive at once (every decoded tile in `rows`, the cropped `result_rows`, and the final
        `torch.cat`), so a single-frame budget under-counts by ~5x and does so linearly with area.
        """
        # 2560x1440, the resolution the PR was measured at.
        if operation == "decode":
            image_tensor = torch.zeros(1, 16, 1, 180, 320)
        else:
            image_tensor = torch.zeros(1, 3, 1, 1440, 2560)
        h, w = 1440, 2560
        tile_size = 256

        with patch("torch.version.hip", None):
            tiled = estimate_vae_working_memory_qwen_image(
                operation=operation, image_tensor=image_tensor, vae=self._mock_fp16_vae(), tile_size=tile_size
            )
            untiled = estimate_vae_working_memory_qwen_image(
                operation=operation, image_tensor=image_tensor, vae=self._mock_fp16_vae(), tile_size=None
            )

        scaling_constant = 2900 if operation == "decode" else 1600
        assert tiled == int(tile_size * tile_size * 2 * scaling_constant * 1.25 + image_copies * 3 * h * w * 2)
        # The whole point: tiling has to shrink the reservation, not just bound the VAE.
        assert tiled < untiled / 4

    def test_tiled_estimate_grows_with_the_tile_not_the_image(self):
        """Reserved memory scales with tile_size^2 and is otherwise flat in resolution.

        Only the pixel-buffer term varies with the image, so doubling the output area must not come
        close to doubling the reservation -- that is what lets a 24 GB card survive a 2560x1440 decode.
        """
        with patch("torch.version.hip", None):
            small = estimate_vae_working_memory_qwen_image(
                operation="decode",
                image_tensor=torch.zeros(1, 16, 1, 128, 128),
                vae=self._mock_fp16_vae(),
                tile_size=256,
            )
            large = estimate_vae_working_memory_qwen_image(
                operation="decode",
                image_tensor=torch.zeros(1, 16, 1, 180, 320),
                vae=self._mock_fp16_vae(),
                tile_size=256,
            )
            big_tile = estimate_vae_working_memory_qwen_image(
                operation="decode",
                image_tensor=torch.zeros(1, 16, 1, 128, 128),
                vae=self._mock_fp16_vae(),
                tile_size=512,
            )

        assert large < small * 2
        # 512px tiles cover 4x the area of 256px tiles; the tile term must follow.
        assert big_tile - small == pytest.approx((512**2 - 256**2) * 2 * 2900 * 1.25, rel=1e-6)


class TestQwenImageWorkingMemory:
    """Test that Qwen Image VAE invocations request working memory before decode/encode."""

    def _mock_vae_info(self):
        """Build a mocked AutoencoderKLQwenImage and its LoadedModel wrapper."""
        mock_vae = MagicMock(spec=AutoencoderKLQwenImage)

        # Create mock parameter for dtype detection
        mock_param = torch.zeros(1)
        mock_vae.parameters.return_value = iter([mock_param])
        mock_vae.dtype = torch.float32
        # patch_qwen_image_vae_tiling records and restores these; give them the stock values so
        # assertions read against real numbers rather than auto-created MagicMocks.
        mock_vae.use_tiling = False
        mock_vae.tile_sample_min_height = 256
        mock_vae.tile_sample_min_width = 256
        mock_vae.tile_sample_stride_height = 192
        mock_vae.tile_sample_stride_width = 192

        # Create mock vae_info with a model_on_device context manager yielding (None, vae)
        mock_vae_info = MagicMock()
        mock_vae_info.model = mock_vae
        # Decode places latents on the VAE's intended compute device (see #9373); this must be a
        # real torch.device so `latents.to(device=...)` works instead of raising TypeError.
        mock_vae_info.compute_device = torch.device("cpu")

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
        # Tiling state lives on the instance, so `spec=` does not provide it.
        converted_vae.use_tiling = False
        converted_vae.tile_sample_min_height = 256
        converted_vae.tile_sample_min_width = 256
        converted_vae.tile_sample_stride_height = 192
        converted_vae.tile_sample_stride_width = 192
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

        # Stop at the encode itself: everything before it is what this test is about, and a bare
        # `except Exception` here would also swallow a failure in the code under test.
        mock_vae.encode.side_effect = RuntimeError("stop at encode")

        with patch(estimation_path) as mock_estimate:
            expected_memory = 1024 * 1024 * 5000  # 5GB
            mock_estimate.return_value = expected_memory

            with pytest.raises(RuntimeError, match="stop at encode"):
                QwenImageImageToLatentsInvocation.vae_encode(mock_vae_info, mock_image_tensor)

            mock_estimate.assert_called_once()
            assert mock_estimate.call_args.kwargs["operation"] == "encode"
            mock_vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected_memory)

    def test_qwen_image_to_latents_passes_resolved_tile_size_to_the_estimate(self):
        """Tiling only helps if the *estimate* shrinks with it.

        The cache reserves whatever the estimator returns, so enabling tiling on the VAE without
        telling the estimator would leave it reserving the full-frame figure (~11 GB at 2560x1440)
        and evicting models to honour it: the encode would be bounded, but nothing else would fit.
        tile_size=0 means "use the model default", which must be resolved before estimating.
        """
        mock_vae, mock_vae_info = self._mock_vae_info()
        mock_vae.encode.side_effect = RuntimeError("stop at encode")
        # A previous invocation left a different tile size on the shared module. The resolved default
        # must come from the constant, not from whatever is currently on the model.
        mock_vae_info.model.tile_sample_min_height = 512
        mock_image_tensor = torch.zeros(1, 3, 512, 512)

        estimation_path = "invokeai.app.invocations.qwen_image_image_to_latents.estimate_vae_working_memory_qwen_image"

        cases = (
            (False, 0, None),
            (True, 0, QWEN_IMAGE_VAE_DEFAULT_TILE_SIZE),
            (True, 512, 512),
            # Below the minimum the tile size is clamped, and the estimate must follow the clamp.
            (True, 8, 64),
        )
        for tiled, tile_size, expected in cases:
            with patch(estimation_path) as mock_estimate:
                mock_estimate.return_value = 1024
                with pytest.raises(RuntimeError, match="stop at encode"):
                    QwenImageImageToLatentsInvocation.vae_encode(
                        mock_vae_info, mock_image_tensor, tiled=tiled, tile_size=tile_size
                    )
                assert mock_estimate.call_args.kwargs["tile_size"] == expected, f"tiled={tiled}, tile_size={tile_size}"

    @pytest.mark.parametrize(
        "tile_size, expected_min, expected_stride",
        [
            (0, 256, 192),
            (512, 512, 384),
            (128, 128, 96),
            (8, 64, 48),
            # 3/4 of these is 54 and 60, neither a multiple of the VAE's 8x spatial compression:
            # the stride must be rounded down to 48 / 56 or the pixel and latent tile loops disagree.
            (72, 72, 48),
            (80, 80, 56),
        ],
    )
    def test_image_to_latents_sets_a_matched_tile_stride(self, tile_size, expected_min, expected_stride):
        """`enable_tiling` must always receive all four parameters.

        Left out, the stride keeps whatever value the module already carries (192 by default): a
        smaller `min` then makes the tile loops skip whole bands of the image, and a larger one grows
        every tile without removing any. See TestQwenImageVaeTiling for the end-to-end consequence.
        """
        mock_vae, mock_vae_info = self._mock_vae_info()
        mock_vae.encode.side_effect = RuntimeError("stop at encode")

        with patch(
            "invokeai.app.invocations.qwen_image_image_to_latents.estimate_vae_working_memory_qwen_image",
            return_value=1024,
        ):
            with pytest.raises(RuntimeError, match="stop at encode"):
                QwenImageImageToLatentsInvocation.vae_encode(
                    mock_vae_info, torch.zeros(1, 3, 512, 512), tiled=True, tile_size=tile_size
                )

        mock_vae.enable_tiling.assert_called_once_with(
            tile_sample_min_height=expected_min,
            tile_sample_min_width=expected_min,
            tile_sample_stride_height=expected_stride,
            tile_sample_stride_width=expected_stride,
        )

    def test_latents_to_image_sets_a_matched_tile_stride(self):
        """Same contract on the decode node, which is where the inert-tiling bug actually was."""
        mock_vae, mock_vae_info = self._mock_vae_info()
        mock_vae.decode.side_effect = RuntimeError("stop at decode")
        mock_vae.config.z_dim = 16
        mock_vae.config.latents_mean = [0.0] * 16
        mock_vae.config.latents_std = [1.0] * 16

        mock_context = MagicMock()
        mock_context.models.load.return_value = mock_vae_info
        mock_context.tensors.load.return_value = torch.zeros(1, 16, 1, 64, 64)
        mock_context.config.get.return_value.force_tiled_decode = False

        with (
            patch(
                "invokeai.app.invocations.qwen_image_latents_to_image.estimate_vae_working_memory_qwen_image",
                return_value=1024,
            ) as mock_estimate,
            patch(
                "invokeai.app.invocations.qwen_image_latents_to_image.SeamlessExt.static_patch_model",
                return_value=nullcontext(),
            ),
        ):
            invocation = QwenImageLatentsToImageInvocation.model_construct(
                latents=MagicMock(latents_name="test_latents"),
                vae=MagicMock(vae=MagicMock(), seamless_axes=[]),
                tiled=True,
                tile_size=512,
            )
            with pytest.raises(RuntimeError, match="stop at decode"):
                invocation.invoke(mock_context)

        assert mock_estimate.call_args.kwargs["tile_size"] == 512
        mock_vae.enable_tiling.assert_called_once_with(
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=384,
            tile_sample_stride_width=384,
        )

    def test_latents_to_image_honours_the_global_force_tiled_decode(self):
        """`force_tiled_decode` must reach both the VAE and the estimate, or tiling stays inert."""
        mock_vae, mock_vae_info = self._mock_vae_info()
        mock_vae.decode.side_effect = RuntimeError("stop at decode")
        mock_vae.config.z_dim = 16
        mock_vae.config.latents_mean = [0.0] * 16
        mock_vae.config.latents_std = [1.0] * 16

        mock_context = MagicMock()
        mock_context.models.load.return_value = mock_vae_info
        mock_context.tensors.load.return_value = torch.zeros(1, 16, 1, 64, 64)
        mock_context.config.get.return_value.force_tiled_decode = True

        with (
            patch(
                "invokeai.app.invocations.qwen_image_latents_to_image.estimate_vae_working_memory_qwen_image",
                return_value=1024,
            ) as mock_estimate,
            patch(
                "invokeai.app.invocations.qwen_image_latents_to_image.SeamlessExt.static_patch_model",
                return_value=nullcontext(),
            ),
        ):
            invocation = QwenImageLatentsToImageInvocation.model_construct(
                latents=MagicMock(latents_name="test_latents"),
                vae=MagicMock(vae=MagicMock(), seamless_axes=[]),
                tiled=False,
                tile_size=0,
            )
            with pytest.raises(RuntimeError, match="stop at decode"):
                invocation.invoke(mock_context)

        assert mock_estimate.call_args.kwargs["tile_size"] == QWEN_IMAGE_VAE_DEFAULT_TILE_SIZE
        mock_vae.enable_tiling.assert_called_once()


class TestQwenImageVaeTiling:
    """Exercise the tiling parameters against a real (tiny, randomly initialised) Qwen-Image VAE.

    The mocked tests above can only assert which arguments are passed. These assert what those
    arguments actually do, which is where both tiling bugs lived: `AutoencoderKLQwenImage` steps its
    tile loops by *stride* while slicing each accumulated tile to *min*, so an inherited stride
    silently truncates the output instead of raising.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def vae(cls):
        # Smallest configuration that keeps the real 8x spatial compression and 16 latent channels.
        return AutoencoderKLQwenImage(
            base_dim=4,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True],
        ).eval()

    # 72 and 80 are the interesting ones: their raw 3/4 stride (54, 60) is not a multiple of 8.
    @pytest.mark.parametrize("tile_size", [64, 72, 80, 128, 192, 256, 512])
    def test_tiled_decode_keeps_the_full_output_size(self, vae, tile_size):
        latents = torch.zeros(1, 16, 1, 32, 32)  # -> 256x256
        with torch.inference_mode(), patch_qwen_image_vae_tiling(vae, tile_size):
            decoded = vae.decode(latents, return_dict=False)[0]
        assert decoded.shape == (1, 3, 1, 256, 256)

    @pytest.mark.parametrize("tile_size", [64, 72, 80, 128, 192, 256, 512])
    def test_tiled_encode_keeps_the_full_latent_size(self, vae, tile_size):
        image = torch.zeros(1, 3, 1, 256, 256)
        with torch.inference_mode(), patch_qwen_image_vae_tiling(vae, tile_size):
            latents = vae.encode(image).latent_dist.mode()
        assert latents.shape == (1, 16, 1, 32, 32)

    def test_inherited_stride_would_truncate(self, vae):
        """Pin the failure mode the four-parameter call exists to prevent.

        This is the pre-fix call: `min` only, stride left at the module's 192. If diffusers ever
        starts deriving the stride from `min`, this test fails and the explicit stride can be dropped.
        """
        latents = torch.zeros(1, 16, 1, 32, 32)
        with patch_qwen_image_vae_tiling(vae, 256):
            vae.enable_tiling(tile_sample_min_height=128, tile_sample_min_width=128)
            with torch.inference_mode():
                decoded = vae.decode(latents, return_dict=False)[0]
        assert decoded.shape == (1, 3, 1, 192, 192)  # not 256x256 -- silently truncated

    def test_unrounded_stride_would_misalign(self, vae):
        """Pin the failure mode the multiple-of-8 rounding exists to prevent.

        `tiled_encode` steps the tile loop by `tile_sample_stride_height` (pixels) but slices each
        accumulated tile to `tile_sample_stride_height // spatial_compression_ratio` (latents). Unless
        the pixel stride is exactly 8x the latent stride the two disagree, and the encode silently
        returns a latent smaller than the image -- the same class of bug as the inherited stride, but
        reachable from a `multiple_of=8` field value whose 3/4 stride is not itself a multiple of 8.
        """
        image = torch.zeros(1, 3, 1, 512, 512)
        with patch_qwen_image_vae_tiling(vae, 72):
            # The rounded stride the helper actually applies.
            assert vae.tile_sample_stride_height == 48
            with torch.inference_mode():
                assert vae.encode(image).latent_dist.mode().shape == (1, 16, 1, 64, 64)

            # ...and the un-rounded 3/4 value it deliberately avoids.
            vae.enable_tiling(tile_sample_stride_height=54, tile_sample_stride_width=54)
            with torch.inference_mode():
                misaligned = vae.encode(image).latent_dist.mode()
        assert misaligned.shape == (1, 16, 1, 57, 57)  # not 64x64 -- silently truncated

    def test_tiling_state_is_restored(self, vae):
        """The VAE module belongs to the model cache and outlives the invocation.

        `enable_tiling` writes the geometry onto the module and `disable_tiling` only clears
        `use_tiling`, so without the restore a tile size set once would persist for the lifetime of
        the cache entry -- and leak into other nodes sharing the instance.
        """
        before = (
            vae.use_tiling,
            vae.tile_sample_min_height,
            vae.tile_sample_min_width,
            vae.tile_sample_stride_height,
            vae.tile_sample_stride_width,
        )
        with patch_qwen_image_vae_tiling(vae, 512):
            assert vae.use_tiling is True
            assert vae.tile_sample_min_height == 512
            assert vae.tile_sample_stride_height == 384
        assert (
            vae.use_tiling,
            vae.tile_sample_min_height,
            vae.tile_sample_min_width,
            vae.tile_sample_stride_height,
            vae.tile_sample_stride_width,
        ) == before

        with patch_qwen_image_vae_tiling(vae, None):
            assert vae.use_tiling is False
        assert (
            vae.use_tiling,
            vae.tile_sample_min_height,
            vae.tile_sample_min_width,
            vae.tile_sample_stride_height,
            vae.tile_sample_stride_width,
        ) == before

    def test_geometry_left_by_another_node_does_not_affect_the_decode(self, vae):
        """`anima_latents_to_image` sets 512/384 on the same instance and never restores it.

        A native-layout `qwen_image_vae` single file is classified with the Anima base, so one loaded
        VAE can feed both nodes in a single workflow. Because all four parameters are passed
        explicitly and the 0 sentinel resolves against a constant, the leaked geometry cannot change
        this decode -- and is still intact afterwards for whoever set it.
        """
        vae.enable_tiling(
            tile_sample_min_height=512,
            tile_sample_min_width=512,
            tile_sample_stride_height=384,
            tile_sample_stride_width=384,
        )
        try:
            latents = torch.zeros(1, 16, 1, 32, 32)
            with torch.inference_mode(), patch_qwen_image_vae_tiling(vae, QWEN_IMAGE_VAE_DEFAULT_TILE_SIZE):
                decoded = vae.decode(latents, return_dict=False)[0]
            assert decoded.shape == (1, 3, 1, 256, 256)
            assert vae.tile_sample_min_height == 512
            assert vae.tile_sample_stride_height == 384
        finally:
            vae.disable_tiling()
            vae.tile_sample_min_height = vae.tile_sample_min_width = 256
            vae.tile_sample_stride_height = vae.tile_sample_stride_width = 192

    def test_tile_count_is_bounded_by_the_stride_ratio(self, vae):
        """Raising `tile_size` must remove tiles, not just enlarge them.

        With a fixed stride the loops emit the same number of tiles at every `tile_size`, so compute
        grows with `tile_size**2` -- 8x a full frame at 512px on a 2560x1440 decode. The proportional
        stride keeps the processed area flat at ~2x regardless of tile size.
        """
        latent_h, latent_w = 180, 320  # 2560x1440
        for tile_size in (256, 512, 768):
            with patch_qwen_image_vae_tiling(vae, tile_size):
                latent_stride = vae.tile_sample_stride_height // 8
                tiles = math.ceil(latent_h / latent_stride) * math.ceil(latent_w / latent_stride)
            processed = tiles * tile_size * tile_size
            assert processed / (1440 * 2560) < 2.5, f"tile_size={tile_size} processes {processed} px"

    @pytest.mark.parametrize("tile_size", [0, 64, 128, 512])
    def test_image_to_latents_node_produces_the_untiled_latent_shape(self, vae, tile_size):
        """The whole node path, tiled, against a real VAE -- the shape must not depend on tile_size.

        An undersized latent raises nothing here: it flows onward as a perfectly valid-looking tensor,
        which is why this is asserted end-to-end rather than only on the arguments.
        """
        vae_info = MagicMock()
        vae_info.model = vae
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=(None, vae))
        cm.__exit__ = MagicMock(return_value=None)
        vae_info.model_on_device = MagicMock(return_value=cm)

        with patch(
            "invokeai.app.invocations.qwen_image_image_to_latents.TorchDevice.choose_torch_device",
            return_value=torch.device("cpu"),
        ):
            latents = QwenImageImageToLatentsInvocation.vae_encode(
                vae_info, torch.zeros(1, 3, 256, 256), tiled=True, tile_size=tile_size
            )

        assert latents.shape == (1, 16, 1, 32, 32)
        # And the shared module is left exactly as it was found.
        assert (vae.use_tiling, vae.tile_sample_min_height, vae.tile_sample_stride_height) == (False, 256, 192)
