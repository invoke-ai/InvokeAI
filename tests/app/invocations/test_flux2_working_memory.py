"""FLUX.2 working-memory estimates: the transformer denoise and both VAE directions.

The FLUX.2 path originally called `model_on_device()` with no `working_mem_bytes` anywhere, so the
model cache reserved only the small default `device_working_mem_gb` and filled the rest of the card
with the model. Reference images make that fatal rather than merely tight: their latents are
concatenated onto the image stream, so three 1024x1024 references quadruple the attended sequence of
a 1024x1024 generation. See https://github.com/invoke-ai/InvokeAI/issues/9500.

The denoise estimate (`denoise.flux2`) and the VAE's ship their coefficients and measured points in
`invokeai/backend/working_memory/data/`, where `tests/backend/working_memory/test_shipped_coefficients.py`
holds every estimate to at least each point and, where a fitted coefficient decides it, at most 1.3x (or
the point's `max_ratio`). `tests/app/invocations/test_flux2_denoise_working_memory.py` holds the denoise
estimate to the formula it replaced where the legacy constants still price, and to the checks the shipped
points leave out. The tests here pin how the estimates are composed and what the nodes hand them, reading
each line's slope from the shipped data.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from invokeai.app.invocations.flux2.flux2_denoise import FLUX2_MAX_HIDDEN_SIZE, Flux2DenoiseInvocation
from invokeai.backend.flux2.denoise_working_memory import (
    FLUX2_ATTENTION_HEAD_DIM,
    FLUX2_DENOISE_SPEC,
    estimate_flux2_denoise_working_memory,
)
from invokeai.backend.flux2.extensions.regional_prompting_extension import Flux2RegionalPromptingExtension
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.util.attention import (
    SDPA_MATH_BYTES_PER_SCORE_ELEMENT,
    _diffusers_attention_dispatch,
    _torch_sdpa_materializes_score_matrix,
    rocm_sdpa_uses_math_kernel,
    sdpa_score_matrix_bytes,
)
from invokeai.backend.util.vae_working_memory import VAE_FLUX2_SPEC, estimate_vae_working_memory_flux2
from invokeai.backend.working_memory import spec as spec_module
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.platform import Platform

MB = 1024**2
GB = 1024**3


def _mock_bf16_vae() -> MagicMock:
    """A bf16 `AutoencoderKLFlux2` stand-in for the estimator: the parameter dtype, and the stock geometry a tiled
    estimate snaps its tile with as `scoped_vae_tiling` does, an 8x downsample and the 0.25 overlap."""
    vae = MagicMock(spec=AutoencoderKLFlux2)
    vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
    vae.config.block_out_channels = (128, 256, 512, 512)
    vae.tile_overlap_factor = 0.25
    return vae


def _vae_line(conv_backend, operation, pixels, allocator="native", os="linux"):
    """What the shipped FLUX.2 VAE line reserves in bf16 over `pixels` output pixels on one platform."""
    values = shipped_coefficients(VAE_FLUX2_SPEC, Platform(conv_backend, allocator, os)).values
    return int(values[f"k_{operation}_bf16"] * pixels + values[f"b_{operation}_bf16"])


CPU = Platform("cpu", "native", "linux")
"""The platform the `FUSED` device prices on: the CPU, priced as cuDNN with the native allocator, floored."""


# The width the denoise tests price at unless they say otherwise, and the head count that follows from it.
KLEIN_9B = 4096
KLEIN_9B_HEADS = KLEIN_9B // FLUX2_ATTENTION_HEAD_DIM


DENOISE_PLATFORM = Platform("cudnn", "native", "linux")
"""The platform every denoise estimate here is priced on, the RTX 4090's measured lines, whatever device answers the
attention questions: the CPU stands in for a fused kernel and an unprobed CUDA device for a materializing one, and either
would otherwise select coefficients of its own."""


def _priced_on_denoise_platform():
    return patch.object(spec_module, "platform_for", return_value=DENOISE_PLATFORM)


def _per_token(hidden=KLEIN_9B):
    """What the shipped denoise line of a width charges per attended token and sample."""
    return shipped_coefficients(FLUX2_DENOISE_SPEC, DENOISE_PLATFORM).values[f"k_denoise_bf16_h{hidden}"]


# The lines were measured where SDPA runs a fused kernel and no score matrix is materialized.
# torch reports its CPU flash kernel as eligible for every shape used
# here, so passing a CPU device reproduces that regime without needing a GPU on the test runner. The
# materializing regime gets its own class below.
FUSED = torch.device("cpu")


def _estimate(
    image_seq_len,
    ref_image_seq_len=0,
    text_seq_len=512,
    num_loras=0,
    batch_size=1,
    hidden_size=KLEIN_9B,
    regional=False,
    device=FUSED,
):
    """The denoise reservation of a dense transformer. `regional` passes a regional mask over the image and caption
    tokens, as the node builds it."""
    side = image_seq_len + text_seq_len
    with _priced_on_denoise_platform():
        return estimate_flux2_denoise_working_memory(
            torch.nn.Module(),
            hidden_size=hidden_size,
            image_seq_len=image_seq_len,
            ref_image_seq_len=ref_image_seq_len,
            text_seq_len=text_seq_len,
            batch_size=batch_size,
            lora_count=num_loras,
            regional_attn_mask=torch.empty((side, side), device="meta") if regional else None,
            device=device,
            dtype=torch.bfloat16,
        ).reservation_bytes


class TestTransformerWidthIsBudgeted:
    """Per-token activation cost follows the transformer's hidden width, not only the token count: each width has a
    line of its own in the shipped data, fitted on its own points.

    Calibrating on Klein 9B alone silently under-reserved FLUX.2 [dev] by a third, and [dev] reaches
    the denoise node as a first-class path with its own loader and starter models. So each width has a
    line of its own, and the head count follows the same width, so the score-matrix term gets the
    variant's real count rather than the widest.
    """

    def _slope_per_token(self, hidden):
        short = _estimate(image_seq_len=4096, text_seq_len=512, hidden_size=hidden)
        long = _estimate(image_seq_len=8704, text_seq_len=512, hidden_size=hidden)
        return (long - short) / (9216 - 4608)

    def test_each_width_is_priced_on_its_own_line(self):
        widths = (3072, 4096, 6144)
        slopes = [self._slope_per_token(hidden) for hidden in widths]

        assert slopes == [_per_token(hidden) for hidden in widths]
        assert slopes[0] < slopes[1] < slopes[2]

    @pytest.mark.parametrize("hidden, heads", [(3072, 24), (4096, 32), (6144, 48)])
    def test_the_score_matrix_uses_the_variants_real_head_count(self, hidden, heads):
        """It used to charge 48 heads for every variant. On a materializing backend that is a third
        too much on Klein 9B and double on Klein 4B -- and on MPS, where the probe always takes the
        materializing branch, that term lands on every single estimate."""
        seq_len = 4096 + 512
        fused = _estimate(image_seq_len=4096, regional=True, hidden_size=hidden)
        with _materializing():
            materializing = _estimate(image_seq_len=4096, regional=True, hidden_size=hidden, device=MATERIALIZING)
        assert materializing - fused == heads * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT


class TestFlux2VaeWorkingMemoryEstimate:
    def _tensor_for(self, operation, px):
        # decode receives 32-channel latents at pixels/8; encode receives a pixel image.
        return torch.zeros(1, 32, px // 8, px // 8) if operation == "decode" else torch.zeros(1, 3, px, px)

    @pytest.mark.parametrize("operation", ["decode", "encode"])
    def test_the_estimate_is_the_bf16_line_over_the_pixel_area(self, operation):
        estimate = estimate_vae_working_memory_flux2(
            operation=operation, image_tensor=self._tensor_for(operation, 1024), vae=_mock_bf16_vae(), device=FUSED
        )
        assert estimate.reservation_bytes == _vae_line("cudnn", operation, 1024 * 1024)

    def test_a_tiled_reference_image_encode_reserves_a_fraction_of_the_untiled_one(self):
        """Reference-image encoding forces 512px tiling so the peak stops following the reference resolution. The whole
        point of tiling: it must shrink the reservation, not just bound the VAE."""
        tiled, untiled = (
            estimate_vae_working_memory_flux2(
                operation="encode",
                image_tensor=torch.zeros(1, 3, 2024, 2024),
                vae=_mock_bf16_vae(),
                tile_size=tile_size,
                device=FUSED,
            ).reservation_bytes
            for tile_size in (512, None)
        )
        assert tiled < untiled / 4

    def _decode_estimate(self, height, width, tile_size):
        return estimate_vae_working_memory_flux2(
            operation="decode",
            image_tensor=torch.zeros(1, 32, height // 8, width // 8),
            vae=_mock_bf16_vae(),
            tile_size=tile_size,
            device=FUSED,
        )

    @pytest.mark.parametrize("tile_size", [1024, 2048, 4096])
    def test_a_tile_covering_the_image_is_priced_as_the_single_pass_it_runs(self, tile_size):
        """The VAE tiles only when a side is strictly larger than the tile. Pricing the tile instead
        reserved 85.9 GiB for a 1024px decode that peaks at 4.3 GiB, evicting the transformer for nothing."""
        estimate = self._decode_estimate(1024, 1024, tile_size)
        assert estimate.reservation_bytes == _vae_line("cudnn", "decode", 1024 * 1024)
        assert estimate.features["tile"] is None

    @pytest.mark.parametrize("tile_size, resolved", [(0, 512), (64, 128)], ids=["default-sentinel", "below-minimum"])
    def test_the_node_tile_field_is_resolved_before_pricing(self, tile_size, resolved):
        """The nodes pass their `tile_size` field through: 0 means the default tile, and anything under
        the minimum is raised to it -- pricing the raw value would reserve nothing or too little."""
        estimate = self._decode_estimate(1024, 1024, tile_size)
        assert estimate.features["tile"] == resolved
        assert estimate.reservation_bytes == self._decode_estimate(1024, 1024, resolved).reservation_bytes


class TestVaeConstantsFollowTheConvBackend:
    """The pixel-area coefficient is not one number. MIOpen's convolution workspaces cost far more
    than cuDNN's for the same decode, and the gap is not the attention term -- it is identical on the
    fused path. Both columns, the cards they were fitted on and the points that bound them are in
    `invokeai/backend/working_memory/data/vae.json`; what is pinned here is that a device selects the
    column of the backend it really runs on.
    """

    def test_a_rocm_build_selects_the_miopen_column(self):
        """A HIP build reports `device.type == "cuda"`, so the torch build is the discriminator."""
        vae = MagicMock(spec=AutoencoderKLFlux2)
        vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
        latents = torch.zeros(1, 32, 128, 128)

        def estimate():
            v = MagicMock(spec=AutoencoderKLFlux2)
            v.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
            return estimate_vae_working_memory_flux2(
                operation="decode", image_tensor=latents, vae=v, device=torch.device("cuda")
            ).reservation_bytes

        # `_IS_ROCM` is pinned because it is bound at import: on a real HIP build the head-dim
        # guard short-circuits ahead of the probe patched here and charges a score matrix to both
        # legs, leaving the ratio correct only while the cuDNN linear term still exceeds it -- 1%
        # at this size. The column is this test's subject; the guard is the next test's.
        with (
            patch("invokeai.backend.util.attention._IS_ROCM", False),
            patch("torch.version.hip", "7.1.25424"),
            patch("invokeai.backend.util.attention._torch_sdpa_materializes_score_matrix", return_value=False),
        ):
            rocm = estimate()
        with (
            patch("invokeai.backend.util.attention._IS_ROCM", False),
            patch("torch.version.hip", None),
            patch("invokeai.backend.util.attention._torch_sdpa_materializes_score_matrix", return_value=False),
        ):
            cuda = estimate()

        assert rocm > cuda
        assert rocm == _vae_line("miopen", "decode", 1024 * 1024)
        assert cuda == _vae_line("cudnn", "decode", 1024 * 1024)

    def test_a_rocm_build_composes_the_miopen_column_with_the_guard_score_matrix(self):
        """The state every ROCm rig is actually in, driven through the estimator.

        A HIP build takes the MIOpen column *and* the head-dim guard's score matrix -- the guard is
        unconditional for this VAE's 512-wide head -- so `max` has to pick between two terms that
        are both larger than the cuDNN pair. The shipped points check the same two terms against the
        measurements, which cannot see an estimator that stops combining them: a
        plausible "the score matrix covers it anyway" refactor that drops back to the cuDNN column
        whenever SDPA materializes under-reserves a 1024px decode by 1.6x and passes every other
        test in this module.
        """

        def estimate(px):
            vae = MagicMock(spec=AutoencoderKLFlux2)
            vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
            return estimate_vae_working_memory_flux2(
                operation="decode",
                image_tensor=torch.zeros(1, 32, px // 8, px // 8),
                vae=vae,
                device=torch.device("cuda"),
            ).reservation_bytes

        with (
            patch("torch.version.hip", "7.1.25424"),
            # Expandable segments on Windows, the ROCm default there, where the MIOpen line is the smaller one.
            patch("invokeai.backend.working_memory.platform.expandable_segments_enabled", return_value=True),
            patch("invokeai.backend.working_memory.platform._host_os", return_value="windows"),
            patch("invokeai.backend.util.attention._IS_ROCM", True),
            # Pinned against `INVOKE_ROCM_FUSED_SDPA_MAX_HEAD_DIM`, which a user can set high
            # enough to disarm the guard; any threshold between the denoise's 128-wide head and
            # this VAE's 512-wide one gives the shipped behavior.
            patch("invokeai.backend.util.attention.ROCM_FUSED_SDPA_MAX_HEAD_DIM", 256),
        ):
            at_1024 = estimate(1024)
            at_1536 = estimate(1536)

        assert at_1024 == _vae_line("miopen", "decode", 1024 * 1024, "expandable", "windows"), "the line is larger"
        assert at_1536 == 36864 * 36864 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT, "at 1536px the score matrix is"
        assert at_1536 > _vae_line("miopen", "decode", 1536 * 1536, "expandable", "windows")


class TestFlux2VaeBatchIsBudgeted:
    """`vae.decode` is handed whatever batch the latents carry, and a `LatentsField` is not pinned to
    one. An estimate built from H and W alone gives a two-sample decode the same reservation as a
    single one, so the cache admits it to a card that cannot run it -- the reservation is there, and
    the OOM happens anyway.

    Measured at 1024px on CUDA/bf16, peak reserved, each point in a fresh process: 4.23GB at batch 1,
    7.96GB at batch 2, 11.89GB at batch 3 (shipped as `vae.flux2` points). Linear, and slightly
    sub-linear per sample, so scaling the single-sample estimate is an upper bound rather than a fit.
    """

    def _decode_estimate(self, batch, px=1024, tile_size=None, device=FUSED):
        return estimate_vae_working_memory_flux2(
            operation="decode",
            image_tensor=torch.zeros(batch, 32, px // 8, px // 8),
            vae=_mock_bf16_vae(),
            tile_size=tile_size,
            device=device,
        ).reservation_bytes

    def test_estimate_scales_with_the_batch(self):
        """The regression in one assertion: before this, all three of these were equal."""
        single = self._decode_estimate(1)
        assert self._decode_estimate(2) == 2 * single
        assert self._decode_estimate(3) == 3 * single

    def test_a_three_dimensional_tensor_is_one_sample(self):
        """A bare `(C, H, W)` latent has no batch axis; `shape[0]` would read the channel count."""
        vae = MagicMock(spec=AutoencoderKLFlux2)
        vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
        unbatched = estimate_vae_working_memory_flux2(
            operation="decode", image_tensor=torch.zeros(32, 128, 128), vae=vae, device=FUSED
        ).reservation_bytes
        assert unbatched == self._decode_estimate(1)

    def test_tiling_bounds_the_tile_not_the_batch(self):
        """Tiling caps the spatial term at one tile, but every sample still runs through it."""
        single = self._decode_estimate(1, px=1024, tile_size=512)
        assert self._decode_estimate(3, px=1024, tile_size=512) == 3 * single

    def test_the_score_matrix_scales_with_the_batch(self):
        """It is shaped (batch, heads, S, S), so where it is materialized at all it scales with the
        batch just as the linear term does -- and since both scale together, the larger of the two
        stays the larger at every batch size.

        Sized at 1536px deliberately. At 1024px the cuDNN linear term beats the score matrix by 1%,
        so `max` returns the linear term at every batch size and the assertion holds even if the
        score matrix stops scaling with the batch entirely -- which is a 35GB under-reservation for
        a batched 1536px decode, the exact OOM this term exists to prevent.
        """
        tokens = 192 * 192
        score = tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        linear = self._decode_estimate(1, px=1536)  # fused: the spatial term on its own
        assert score > linear, "the size is load-bearing: `max` must be returning the score matrix"
        with _materializing():
            assert self._decode_estimate(1, px=1536, device=MATERIALIZING) == score
            assert self._decode_estimate(3, px=1536, device=MATERIALIZING) == 3 * score


class _StopBeforeLoad(Exception):
    """Raised in place of entering the transformer's device context, to end _run_diffusion early."""


class TestFlux2DenoiseRequestsWorkingMemory:
    """The denoise node must hand its estimate to the cache, and that estimate must grow with the
    attached reference images -- the combination that #9500 was missing."""

    def _run(
        self,
        num_ref_tokens: int,
        batch: int = 1,
        init_batch: int | None = None,
        variant="klein_9b",
        transformer: torch.nn.Module | None = None,
        on_device=None,
        negative_tokens: int | None = None,
        regional_mask: torch.Tensor | None = None,
    ):
        """Drive `_run_diffusion` up to the transformer load and return the reservation of the estimate it hands
        `model_on_device()`.

        `transformer` replaces the mocked model the node reads before locking. It has to be a real
        module for anything that walks the module tree: a `MagicMock` iterates empty, so an int8
        build looks exactly like a dense one. `negative_tokens` adds a negative caption of that length;
        `regional_mask` stands in for the mask regional prompting builds from masked positive captions.
        """
        from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
        from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
            ConditioningFieldData,
            FLUXConditioningInfo,
        )

        transformer_info = MagicMock()
        transformer_info.model_on_device = MagicMock(side_effect=on_device or _StopBeforeLoad)
        if transformer is not None:
            transformer_info.model = transformer

        context = MagicMock()
        context.models.load.return_value = transformer_info
        from invokeai.backend.model_manager.taxonomy import Flux2VariantType

        context.models.get_config.return_value = MagicMock(
            base=BaseModelType.Flux2,
            type=ModelType.Main,
            format=ModelFormat.Checkpoint,
            variant=Flux2VariantType(variant) if variant is not None else None,
        )

        def conditioning(name: str) -> ConditioningFieldData:
            tokens = negative_tokens if name == "neg" else 512
            return ConditioningFieldData(
                conditionings=[
                    FLUXConditioningInfo(clip_embeds=torch.zeros(1, 768), t5_embeds=torch.zeros(1, tokens, 12288))
                ]
            )

        context.conditioning.load.side_effect = conditioning

        ref_extension = MagicMock()
        ref_extension.ref_image_latents = torch.zeros(1, num_ref_tokens, 128)

        if init_batch is not None:
            # img2img: the node loads these, then preblends them with its own batch-1 noise.
            context.tensors.load.return_value = torch.zeros(init_batch, 32, 128, 128)

        invocation = Flux2DenoiseInvocation.model_construct(
            latents=MagicMock(latents_name="init") if init_batch is not None else None,
            noise=None,
            denoise_mask=None,
            denoising_start=0.0,
            denoising_end=1.0,
            add_noise=True,
            transformer=MagicMock(transformer=MagicMock(), loras=[]),
            positive_text_conditioning=MagicMock(conditioning_name="pos", mask=None),
            negative_text_conditioning=MagicMock(conditioning_name="neg") if negative_tokens is not None else None,
            guidance=4.0,
            cfg_scale=1.0,
            width=1024,
            height=1024,
            num_steps=4,
            scheduler="euler",
            seed=0,
            vae=MagicMock(vae=MagicMock()),
            kontext_conditioning=MagicMock() if num_ref_tokens else None,
        )

        with (
            patch.object(Flux2DenoiseInvocation, "_get_bn_stats", return_value=None),
            patch("invokeai.backend.util.devices.TorchDevice.choose_torch_device", return_value=torch.device("cpu")),
            patch("invokeai.app.invocations.flux2.flux2_denoise.Flux2RefImageExtension", return_value=ref_extension),
            patch.object(
                Flux2DenoiseInvocation, "_prepare_noise_tensor", return_value=torch.zeros(batch, 32, 128, 128)
            ),
            _priced_on_denoise_platform(),
            pytest.raises(_StopBeforeLoad),
            patch.object(Flux2RegionalPromptingExtension, "_prepare_restricted_attn_mask", return_value=regional_mask),
        ):
            invocation._run_diffusion(context)

        transformer_info.model_on_device.assert_called_once()
        requested = transformer_info.model_on_device.call_args.kwargs
        assert "working_mem_bytes" not in requested
        return requested["working_memory"].reservation_bytes

    # `on_device` lets a caller get past the load instead of stopping at it, for the wires that sit
    # further down the node.

    def test_estimate_reaches_the_model_cache(self):
        """Without this the cache reserves only the default `device_working_mem_gb`."""
        assert self._run(num_ref_tokens=0) == _estimate(image_seq_len=64 * 64, text_seq_len=512)

    def test_reference_images_raise_the_reservation(self):
        """Three 1024x1024 references add 12288 tokens to a 1024x1024 generation's 4096."""
        without_refs = self._run(num_ref_tokens=0)
        with_refs = self._run(num_ref_tokens=12288)
        assert with_refs - without_refs == 12288 * _per_token()

    def test_the_real_batch_reaches_the_reservation(self):
        """A batched latent tensor is reachable through the API and custom graphs. The node has `b`
        in hand at the estimate; before this it simply did not pass it, so a two-sample run reserved
        one sample's worth and the cache admitted it to a card that could not run it."""
        assert self._run(num_ref_tokens=0, batch=2) == _estimate(image_seq_len=64 * 64, text_seq_len=512, batch_size=2)

    def test_a_batched_run_reserves_more_than_a_single_one(self):
        single = self._run(num_ref_tokens=0, batch=1)
        assert self._run(num_ref_tokens=0, batch=2) - single == (64 * 64 + 512) * _per_token()

    @pytest.mark.parametrize(
        "variant, hidden",
        [("klein_4b", 3072), ("klein_4b_base", 3072), ("klein_9b", 4096), ("klein_9b_base", 4096), ("dev", 6144)],
    )
    def test_the_variants_width_reaches_the_reservation(self, variant, hidden):
        """Per-token cost is linear in the transformer's width, so the reservation has to know which
        variant it is loading. Calibrating on Klein 9B alone under-reserved [dev] by a third. Base
        variants share their distilled twin's geometry."""
        assert self._run(num_ref_tokens=0, variant=variant) == _estimate(
            image_seq_len=64 * 64, text_seq_len=512, hidden_size=hidden
        )

    def test_dev_reserves_half_again_what_klein_9b_does(self):
        """The blocker in one assertion: 1024x1024 with three 1024x1024 references is 16896 tokens,
        where the difference is ~3GB."""
        tokens = 64 * 64 + 12288 + 512
        klein = self._run(num_ref_tokens=12288, variant="klein_9b")
        dev = self._run(num_ref_tokens=12288, variant="dev")
        assert dev - klein == tokens * (_per_token(6144) - _per_token(4096))
        assert dev - klein > 2 * GB

    def test_an_unreadable_variant_falls_back_to_the_widest(self):
        """Over-reserving on a model we cannot identify beats under-reserving on the largest one."""
        assert self._run(num_ref_tokens=0, variant=None) == _estimate(
            image_seq_len=64 * 64, text_seq_len=512, hidden_size=FLUX2_MAX_HIDDEN_SIZE
        )

    def test_a_batched_init_latent_beats_the_batch_1_noise_it_is_blended_with(self):
        """img2img takes `x = t_0 * noise + (1 - t_0) * init_latents`, and this node builds its noise
        at batch 1 from width/height/seed. Two batched init latents therefore broadcast up to a
        two-sample `x` while the noise tensor -- the thing the batch used to be read from -- still
        says 1. The reservation has to follow `x`."""
        assert self._run(num_ref_tokens=0, init_batch=2) == _estimate(
            image_seq_len=64 * 64, text_seq_len=512, batch_size=2
        )

    def test_the_blended_batch_is_read_after_the_broadcast(self):
        single = self._run(num_ref_tokens=0, init_batch=1)
        assert self._run(num_ref_tokens=0, init_batch=3) - single == 2 * (64 * 64 + 512) * _per_token()

    def test_a_regional_mask_reaches_the_reservation_without_reference_images(self):
        """The mask becomes an additive bias the forward attends with, so its bytes are reserved."""
        side = 64 * 64 + 512
        mask = torch.ones(side, side, dtype=torch.bool)

        assert self._run(num_ref_tokens=0, regional_mask=mask) == _estimate(image_seq_len=64 * 64, regional=True)
        assert self._run(num_ref_tokens=0, regional_mask=mask) > self._run(num_ref_tokens=0)

    def test_a_negative_caption_longer_than_the_positive_one_sets_the_text_tokens(self):
        """The CFG passes run one after the other, so the longer caption of the pair is what a forward attends."""
        assert self._run(num_ref_tokens=0, negative_tokens=700) == _estimate(image_seq_len=64 * 64, text_seq_len=700)
        assert self._run(num_ref_tokens=0, negative_tokens=77) == _estimate(image_seq_len=64 * 64, text_seq_len=512)

    def test_repeated_reference_latents_are_counted_per_sample(self):
        """`ensure_batch_size` repeats the reference latents across the batch, so their tokens scale
        with it as well -- the worst case in #9500, doubled."""
        single = self._run(num_ref_tokens=12288, batch=1)
        double = self._run(num_ref_tokens=12288, batch=2)
        assert double - single == (64 * 64 + 12288 + 512) * _per_token()


def _materializing_probe(device_type, device_index, dtype, head_dim, has_attn_mask):
    """A synthetic stand-in for the torch probe, so both regimes can be exercised without hardware.

    It reports `math` for a head dim above 128 -- which is what ROCm really does, gfx1100 answers
    MATH for the VAE's 512-wide head -- and also for an additive mask, which is *not* a hardware
    claim: gfx1100 reports the memory-efficient kernel for a masked 128-wide head, same as CUDA.
    The mask leg is here only to drive the masked branch of the estimator; which builds take it is
    the probe's business, not this module's.
    """
    return head_dim > 128 or has_attn_mask


@contextmanager
def _pinned_build():
    """Hold the two build-dependent inputs the estimators read at their non-HIP values.

    `MATERIALIZING` below is a `cuda` device object used purely as a token for "SDPA builds the
    score matrix here", but on a HIP build that same token carries two more decisions with it, and
    both would make this module's verdict depend on the runner:

    * `platform_for` selects the MIOpen convolution coefficients, 1.6-2.5x the cuDNN ones every
      expectation guarded by this helper is written against -- so the VAE assertions would pass on
      a CUDA runner and fail on a ROCm one.
    * `rocm_sdpa_uses_math_kernel` short-circuits ahead of the probe below, on a threshold a user
      can move with `INVOKE_ROCM_FUSED_SDPA_MAX_HEAD_DIM`. At the default it happens to agree with
      `_materializing_probe`; at a lower one it charges a score matrix for the denoise's 128-wide
      head that a CUDA runner never sees. `attention._IS_ROCM` is bound at import, so pinning
      `torch.version.hip` alone does not reach it.

    Both regimes below pin these, so they differ in the score-matrix term alone -- which is what
    they are here to isolate. Which column a device really selects, and what the guard really does
    on top of it, is `TestVaeConstantsFollowTheConvBackend`'s subject.
    """
    with (
        patch("torch.version.hip", None),
        patch("invokeai.backend.util.attention._IS_ROCM", False),
    ):
        yield


@contextmanager
def _materializing():
    """`_pinned_build()` plus a synthetic probe, so SDPA reports that it materializes."""
    with (
        _pinned_build(),
        patch(
            "invokeai.backend.util.attention._torch_sdpa_materializes_score_matrix",
            side_effect=_materializing_probe,
        ),
    ):
        yield


# Any CUDA device object works here: the probe is patched out, so nothing is allocated on it.
MATERIALIZING = torch.device("cuda")


class TestMaterializedScoreMatrixIsBudgeted:
    """The linear estimates above assume SDPA never builds the O(S^2) score matrix. That is a
    property of the *build*, not of FLUX.2: some builds have no fused kernel for the VAE's 512-wide
    head, or for the dense additive mask regional prompting attaches, and fall through to `math`.
    Where that happens the score matrix eventually dominates, so the estimate has to account for it
    -- otherwise the fix works on CUDA and still OOMs elsewhere.

    How it accounts for it differs between the two estimators, and the difference is measured rather
    than assumed. In the transformer the score matrix is live alongside the block activations, so it
    adds. In the VAE the mid-block sits alone at the 8x-downsampled bottleneck, so the two peak in
    different phases and the estimate takes the larger -- see `TestVaeTermsDoNotAdd`.
    """

    def _vae_estimate(self, operation, px, device, tile_size=None):
        tensor = torch.zeros(1, 32, px // 8, px // 8) if operation == "decode" else torch.zeros(1, 3, px, px)
        return estimate_vae_working_memory_flux2(
            operation=operation,
            image_tensor=tensor,
            vae=_mock_bf16_vae(),
            tile_size=tile_size,
            device=device,
        ).reservation_bytes

    # (operation, pixel size, mid-block tokens). The VAE attends on the 8x-downsampled grid.
    @pytest.mark.parametrize(
        "operation, px, tokens",
        [
            ("decode", 1536, 192 * 192),
            ("encode", 1024, 128 * 128),
            ("encode", 1328, 166 * 166),
        ],
    )
    def test_the_score_matrix_takes_over_where_it_is_the_larger_term(self, operation, px, tokens):
        """Quadratic beats linear eventually. At these sizes it already has, so the estimate is the
        score matrix exactly -- not the linear term, and not the two added together."""
        with _materializing():
            materializing = self._vae_estimate(operation, px, device=MATERIALIZING)
        assert materializing == tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert materializing > self._vae_estimate(operation, px, device=FUSED)

    def test_mps_style_dispatch_failure_reserves_the_vae_score_matrix(self):
        """The MPS case end to end, through the real probe rather than a stand-in: on a device torch
        cannot answer a dispatch query for, the score matrix has to be priced. Reporting those
        devices as fused -- as this PR first did -- is what let the decode be admitted to a card that
        could not run it."""
        with patch("torch.ops.aten._fused_sdp_choice", side_effect=NotImplementedError("no MPS kernel")):
            materializing = self._vae_estimate("decode", 1536, device=FUSED)
        fused = self._vae_estimate("decode", 1536, device=FUSED)

        tokens = 192 * 192
        assert materializing == tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert materializing > fused + 10 * GB

    def test_vae_score_matrix_dominates_at_high_resolution(self):
        """A 1536px decode is ~9.7GB of convolution activations on cuDNN, and more than twice that
        again in scores where SDPA has to materialize them. An estimate that omits the term is not
        merely tight, it is wrong by a factor of two."""
        with _materializing():
            materializing = self._vae_estimate("decode", 1536, device=MATERIALIZING)
        assert materializing > 20 * GB
        assert materializing > 2 * self._vae_estimate("decode", 1536, device=FUSED)

    def test_tiling_makes_the_score_matrix_irrelevant(self):
        """Tiling caps the mid-block sequence at (tile/8)^2, which drops the quadratic term far below
        the tile's own linear cost -- so a tiled estimate is the convolutions' whether the build
        materializes or not."""
        with _materializing():
            estimates = [
                estimate_vae_working_memory_flux2(
                    operation="encode",
                    image_tensor=torch.zeros(1, 3, px, px),
                    vae=_mock_bf16_vae(),
                    tile_size=512,
                    device=MATERIALIZING,
                )
                for px in (1024, 1328, 2024)
            ]
            untiled = self._vae_estimate("encode", 2024, device=MATERIALIZING)
        for estimate in estimates:
            [scores] = [term for term in estimate.terms if term.name == "mid_block_scores"]
            assert scores.bytes > 0 and not scores.included
        # Tiling turns a 100GB+ reservation at the 2024px reference cap into under 2GB.
        assert estimates[-1].reservation_bytes < untiled / 50

    def test_the_score_matrix_of_an_elongated_image_follows_the_clamped_tile(self):
        """A 2048px tile on a 1024x4096 image slices at most 1024x2048, so the mid-block attends over
        128x256 tokens, not the 256x256 an unclamped tile implies -- a quarter of the scores."""
        with _materializing():
            estimate = estimate_vae_working_memory_flux2(
                operation="encode",
                image_tensor=torch.zeros(1, 3, 1024, 4096),
                vae=_mock_bf16_vae(),
                tile_size=2048,
                device=MATERIALIZING,
            ).reservation_bytes
        tokens = 128 * 256
        assert estimate == tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT, "the scores are the larger phase here"


class TestVaeTermsDoNotAdd:
    """The two VAE terms peak in different phases of the same forward, so the estimate takes the
    larger rather than the sum. The mid-block sits at the 8x-downsampled bottleneck -- first in the
    decoder, last in the encoder -- so the full-resolution convolution feature maps that drive the
    linear term are not live while the score matrix is, and peak *reserved* is a high-water mark.

    Measured, decode, peak reserved, fresh process per point (the cuDNN fused and forced-math points
    in `invokeai/backend/working_memory/data/vae.json`): forcing math measures *below* the fused path
    until 1536px, and even there it exceeds it by 2.6GB against the 21.5GB the score term prices
    standalone -- the attention phase reuses blocks the allocator is already holding. On gfx1100 and
    gfx1201, forcing math moves the measured peak by nothing at all up to 1024px, and the totals stay
    flat-linear in area either way.

    Summing the terms reserved 11.1GB for a 1024px gfx1100 decode that measures 6.7GB. Taking the
    max reserves 6.8GB. Where a max model is not an upper bound on its own -- the crossover, a 768px
    encode with cuDNN's coefficient and a materializing kernel -- the `device_working_mem_gb` floor
    is, and the shipped point there is checked against the floor.
    """

    def _estimate(self, px, device, materializing):
        vae = MagicMock(spec=AutoencoderKLFlux2)
        vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
        with _materializing() if materializing else _pinned_build():
            return estimate_vae_working_memory_flux2(
                operation="decode", image_tensor=torch.zeros(1, 32, px // 8, px // 8), vae=vae, device=device
            ).reservation_bytes

    def test_the_estimate_is_the_larger_term_not_the_sum(self):
        """At 1024px the two are within 2% of each other on the cuDNN column, which makes this the
        sharpest place to tell the models apart."""
        linear = _vae_line("cudnn", "decode", 1024 * 1024)
        score = 16384 * 16384 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert self._estimate(1024, MATERIALIZING, materializing=True) == max(linear, score)
        assert self._estimate(1024, MATERIALIZING, materializing=True) < linear + score

    def test_a_fused_build_is_unaffected(self):
        """The max only ever removes reservation, never adds it: with no score matrix the estimate is
        the linear term, exactly as before."""
        linear = _vae_line("cudnn", "decode", 1024 * 1024)
        assert self._estimate(1024, FUSED, materializing=False) == linear

    def test_regional_prompting_adds_the_score_matrix(self):
        """The dense `S x S` additive bias is what pushes SDPA off its fused kernel. Budgeting only
        the bias tensor -- as this PR first did -- under-reserves by the score matrix, which is two
        orders of magnitude larger."""
        seq_len = 4096 + 512
        bias_bytes = seq_len * seq_len * 2
        fused = _estimate(image_seq_len=4096, text_seq_len=512, regional=True, device=FUSED)
        with _materializing():
            materializing = _estimate(image_seq_len=4096, text_seq_len=512, regional=True, device=MATERIALIZING)
        assert materializing - fused == (KLEIN_9B_HEADS * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT)
        assert materializing - fused > 50 * bias_bytes

    def test_denoise_without_a_regional_mask_is_unaffected(self):
        """FLUX.2's 128-wide attention head is inside every backend's fused limit, so an ordinary
        generation -- reference images included -- keeps the plain linear estimate. This term exists
        for the masked case; it must not tax the common one."""
        with _materializing():
            assert _estimate(image_seq_len=4096, ref_image_seq_len=12288, device=MATERIALIZING) == _estimate(
                image_seq_len=4096, ref_image_seq_len=12288, device=FUSED
            )


class TestSdpaBackendProbe:
    """`sdpa_score_matrix_bytes` decides the term above, so its defaults are load-bearing."""

    def test_cpu_reports_its_fused_flash_kernel(self):
        """torch ships a fused flash-attention CPU kernel that takes the VAE's 512-wide head and an
        additive mask, so the CPU estimate stays linear -- and the rest of this module can use a CPU
        device to stand in for the CUDA regime the constants were measured on."""
        assert (
            sdpa_score_matrix_bytes(
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
                num_heads=1,
                head_dim=512,
                seq_len=16384,
                has_attn_mask=True,
            )
            == 0
        )

    def test_a_device_torch_cannot_answer_for_is_budgeted_as_math(self):
        """MPS is the case that matters: torch registers `_fused_sdp_choice` for CPU, CUDA/ROCm and
        XPU only, and it is exactly the devices it cannot answer for that have no fused SDPA kernel
        either. A 1024px FLUX.2 VAE decode there materializes 16384^2 scores, ~3.5GB the estimate
        used to omit entirely."""
        with patch("torch.ops.aten._fused_sdp_choice", side_effect=NotImplementedError("no MPS kernel")):
            estimated = sdpa_score_matrix_bytes(
                device=torch.device("cpu"), dtype=torch.bfloat16, num_heads=1, head_dim=512, seq_len=16384
            )
        assert estimated == 16384 * 16384 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert estimated > 3 * GB

    def test_a_failed_probe_is_budgeted_as_math(self):
        """A probe that cannot allocate, or a torch without the op, leaves us knowing nothing. The
        old code read that as "fused" and reserved zero; the shortfall it hides is an OOM, so the
        unknown answer has to be the expensive one."""
        with patch("torch.empty", side_effect=torch.cuda.OutOfMemoryError("probe could not allocate")):
            estimated = sdpa_score_matrix_bytes(
                device=torch.device("cpu"), dtype=torch.bfloat16, num_heads=1, head_dim=512, seq_len=16384
            )
        assert estimated == 16384 * 16384 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT

    def _cpu_estimate(self):
        return sdpa_score_matrix_bytes(
            device=torch.device("cpu"), dtype=torch.bfloat16, num_heads=1, head_dim=128, seq_len=4096
        )

    def test_disabling_the_fused_kernels_at_runtime_changes_the_answer(self):
        """The probe is not cached, so an estimate priced after a runtime switch does not inherit the
        answer from before it. Nothing is cleared between these calls on purpose."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        assert self._cpu_estimate() == 0
        with sdpa_kernel([SDPBackend.MATH]):
            assert self._cpu_estimate() == 4096 * 4096 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert self._cpu_estimate() == 0

    def test_a_priority_reorder_changes_the_answer_with_every_flag_unchanged(self):
        """`sdpa_kernel(..., set_priority=True)` puts `MATH` first while leaving all four enable
        flags True, and torch takes the first eligible backend in that order. A cache keyed on the
        flags -- which is what this probe used to have -- could not see the switch and would keep
        reserving zero. Not caching at all is what makes that unrepresentable."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        math_first = [
            SDPBackend.MATH,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        ]
        flags = ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled")

        assert self._cpu_estimate() == 0
        with sdpa_kernel(math_first, set_priority=True):
            # The finding in one line: every flag a key could hold is still True in here.
            assert all(getattr(torch.backends.cuda, name)() for name in flags if hasattr(torch.backends.cuda, name))
            # CPU's chooser ignores the priority order, so stand in for the answer CUDA gives.
            with patch("torch.ops.aten._fused_sdp_choice", return_value=int(SDPBackend.MATH)):
                assert self._cpu_estimate() == 4096 * 4096 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert self._cpu_estimate() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="only CUDA's chooser honours the priority order")
    def test_this_build_reports_a_real_priority_reorder(self):
        """The same case against the real dispatcher rather than a stand-in. Verified on torch
        2.7.1+cu128: `_fused_sdp_choice` answers EFFICIENT outside and MATH inside."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def estimate():
            # `_IS_ROCM` pinned off: torch's dispatch order is the subject here, and a user-lowered
            # `INVOKE_ROCM_FUSED_SDPA_MAX_HEAD_DIM` would otherwise decide this before torch does.
            with patch("invokeai.backend.util.attention._IS_ROCM", False):
                return sdpa_score_matrix_bytes(
                    device=torch.device("cuda"), dtype=torch.bfloat16, num_heads=1, head_dim=128, seq_len=4096
                )

        assert estimate() == 0
        with sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ],
            set_priority=True,
        ):
            assert estimate() == 4096 * 4096 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert estimate() == 0

    def test_the_probe_asks_torch_the_same_question_sdpa_does(self):
        """`_fused_sdp_choice` is the dispatch query `F.scaled_dot_product_attention` itself runs, so
        a `MATH` answer means the real forward materializes. Reimplementing the eligibility rules
        instead would go stale with every torch release."""
        from torch.nn.attention import SDPBackend

        with patch("torch.ops.aten._fused_sdp_choice", return_value=int(SDPBackend.MATH)):
            assert _torch_sdpa_materializes_score_matrix("cpu", None, torch.bfloat16, 128, False)
        with patch("torch.ops.aten._fused_sdp_choice", return_value=int(SDPBackend.EFFICIENT_ATTENTION)):
            assert not _torch_sdpa_materializes_score_matrix("cpu", None, torch.bfloat16, 128, False)

    def test_empty_sequences_cost_nothing(self):
        with _materializing():
            assert (
                sdpa_score_matrix_bytes(
                    device=MATERIALIZING, dtype=torch.bfloat16, num_heads=48, head_dim=128, seq_len=0
                )
                == 0
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="asks the real CUDA/ROCm dispatcher")
    def test_this_build_reports_its_own_dispatch(self):
        """The head dim is the discriminator that actually holds: CUDA's memory-efficient kernel
        takes the VAE's 512-wide head, and on ROCm the head-dim guard sends it to `math` by policy
        (its fused kernels return wrong output there). Masks are not a discriminator -- gfx1100
        reports the memory-efficient kernel for a masked 128-wide head just as CUDA does -- so the
        masked case only has to agree with whatever torch says, which is the whole point of asking
        it."""
        vae_bytes = sdpa_score_matrix_bytes(
            device=torch.device("cuda"), dtype=torch.bfloat16, num_heads=1, head_dim=512, seq_len=16384
        )
        # The masked case is torch's own answer, so the guard is pinned out of the way rather than
        # folded into the expectation below -- at the default threshold it does not reach a
        # 128-wide head anyway, but the threshold is user-settable.
        with patch("invokeai.backend.util.attention._IS_ROCM", False):
            masked_bytes = sdpa_score_matrix_bytes(
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
                num_heads=48,
                head_dim=128,
                seq_len=4608,
                has_attn_mask=True,
            )
        # Asked of the guard's own rule rather than of `torch.version.hip`: the threshold is
        # user-settable, and a HIP build with the guard turned off is back to whatever its fused
        # kernel does. `rocm_sdpa_uses_math_kernel` is the single rule the estimators share.
        guard_forces_math = rocm_sdpa_uses_math_kernel("cuda", 512)
        assert vae_bytes == (16384 * 16384 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT if guard_forces_math else 0)
        materializes = _torch_sdpa_materializes_score_matrix(
            "cuda", torch.device("cuda").index, torch.bfloat16, 128, True
        )
        assert masked_bytes == (48 * 4608 * 4608 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT if materializes else 0)

    # (platform, heads, seq, head_dim, bytes per score element), SDPBackend.MATH forced, each point
    # in a fresh process. Reproduce with `scripts/calibrate_flux2_working_memory.py --only sdpa`.
    MEASURED_BYTES_PER_ELEMENT = [
        ("cuda-4090", 1, 4096, 512, 12.88),
        ("cuda-4090", 1, 8192, 512, 10.28),
        ("cuda-4090", 1, 16384, 512, 9.71),
        ("cuda-4090", 4, 4096, 128, 9.97),
        ("cuda-4090", 48, 4608, 128, 9.58),
        ("rocm-gfx1100", 1, 4096, 512, 13.62),
        ("rocm-gfx1100", 1, 8192, 512, 10.78),
        ("rocm-gfx1100", 1, 16384, 512, 9.76),
        ("rocm-gfx1100", 4, 4096, 128, 10.16),
        ("rocm-gfx1100", 48, 4608, 128, 9.59),
        ("rocm-gfx1201", 1, 4096, 512, 16.38),
        ("rocm-gfx1201", 1, 8192, 512, 13.47),
        ("rocm-gfx1201", 1, 16384, 512, 11.99),
        ("rocm-gfx1201", 4, 4096, 128, 12.84),
        ("rocm-gfx1201", 48, 4608, 128, 11.69),
    ]

    @pytest.mark.parametrize("platform, num_heads, seq_len, head_dim, measured", MEASURED_BYTES_PER_ELEMENT)
    def test_the_constant_upper_bounds_every_measured_platform(self, platform, num_heads, seq_len, head_dim, measured):
        """The constant has to bound the worst build anyone has measured, not the one it was written
        on. It was 13 (CUDA's worst point rounded up), then 14 when gfx1100 came in at 13.62, and is
        17 because gfx1201 costs 16.38 for the same shape. The spread between the two AMD cards is
        wider than the gap between CUDA and either of them, so there is no "the ROCm number"."""
        assert SDPA_MATH_BYTES_PER_SCORE_ELEMENT >= measured

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="measures real peak reserved memory")
    @pytest.mark.parametrize("num_heads, seq_len, head_dim", [(1, 4096, 512), (4, 4096, 128)])
    def test_constant_upper_bounds_a_forced_math_forward(self, num_heads, seq_len, head_dim):
        """Pin the bytes-per-score-element calibration against a real `math` forward, so a future
        edit to the constant cannot silently reintroduce the shortfall it exists to cover."""
        from torch.nn.attention import SDPBackend, sdpa_kernel

        device = torch.device("cuda")
        q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_reserved()
        with sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        measured = torch.cuda.max_memory_reserved() - before

        estimate = num_heads * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert estimate >= measured, (
            f"{estimate / num_heads / seq_len / seq_len:.2f} bytes/element measured here; "
            f"SDPA_MATH_BYTES_PER_SCORE_ELEMENT={SDPA_MATH_BYTES_PER_SCORE_ELEMENT} must be an upper bound"
        )
        # Loose on purpose. The constant is the worst of three measured platforms (CUDA ~10
        # bytes/element, gfx1100 ~10-14, gfx1201 ~12-16), so on any one of them it over-shoots by
        # design; and this runs in-process, where the allocator may satisfy the forward from blocks
        # it already holds and report a smaller delta than a fresh process would. The guard is here
        # to catch an order-of-magnitude blunder, not to pin the calibration -- that is what
        # `test_the_constant_upper_bounds_every_measured_platform` and the calibration script do.
        assert estimate <= 3 * measured


def _diffusers_backend(name):
    """Force the process-wide diffusers attention backend, as `DIFFUSERS_ATTN_BACKEND` would."""
    from diffusers.models.attention_dispatch import AttentionBackendName

    return patch(
        "diffusers.models.attention_dispatch._AttentionBackendRegistry.get_active_backend",
        return_value=(AttentionBackendName(name), None),
    )


class TestDiffusersAttentionDispatchIsConsulted:
    """The FLUX.2 transformer does not call `F.scaled_dot_product_attention` -- it calls diffusers'
    `dispatch_attention_fn`, which honours `DIFFUSERS_ATTN_BACKEND` and the `attention_backend()`
    context manager. A user on `_native_math` materializes the score matrix on hardware where the
    torch probe reports a fused kernel, so asking torch alone is not enough for the transformer.

    The VAE is the other half of the same point: its mid-block attention goes through
    `AttnProcessor2_0`, which calls `F.scaled_dot_product_attention` itself, so the diffusers
    backend must *not* move its estimate.
    """

    def test_forced_math_backend_reaches_the_denoise_estimate(self):
        with _diffusers_backend("_native_math"):
            forced_math = _estimate(image_seq_len=4096, device=FUSED)
        native = _estimate(image_seq_len=4096, device=FUSED)
        seq_len = 4096 + 512
        assert forced_math - native == (KLEIN_9B_HEADS * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT)

    def test_a_fused_backend_leaves_the_denoise_estimate_linear(self):
        """`flash`, `sage`, `xformers` and friends exist precisely to avoid the score matrix; they
        must not be taxed for it, on any device."""
        with _diffusers_backend("flash"), _materializing():
            forced_flash = _estimate(image_seq_len=4096, regional=True, device=MATERIALIZING)
        assert forced_flash == _estimate(image_seq_len=4096, regional=True, device=FUSED)

    def test_the_vae_estimate_ignores_the_diffusers_backend(self):
        """`AttnProcessor2_0` bypasses the dispatcher, so the VAE's answer comes from torch alone."""

        def estimate():
            v = MagicMock(spec=AutoencoderKLFlux2)
            v.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
            return estimate_vae_working_memory_flux2(
                operation="decode", image_tensor=torch.zeros(1, 32, 128, 128), vae=v, device=FUSED
            ).reservation_bytes

        with _diffusers_backend("_native_math"):
            forced_math = estimate()
        assert forced_math == estimate()

    def test_a_backend_switch_is_not_masked_by_an_earlier_estimate(self):
        """The active backend is mutable process state. Caching the first answer would keep
        reserving zero for every later estimate in a long-lived process that has since switched to
        `_native_math` -- the exact case this lookup exists to catch. Deliberately no cache is
        cleared between the two calls here; the production code must not be holding one."""
        native = _estimate(image_seq_len=4096, device=FUSED)
        with _diffusers_backend("_native_math"):
            after_switch = _estimate(image_seq_len=4096, device=FUSED)
        back_to_native = _estimate(image_seq_len=4096, device=FUSED)

        seq_len = 4096 + 512
        assert after_switch - native == (KLEIN_9B_HEADS * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT)
        assert back_to_native == native

    def test_a_model_level_override_reaches_the_registry(self):
        """Why the estimator does not need the model in hand: it is priced before the transformer is
        loaded, and `set_attention_backend()` stamps its choice onto the process-wide registry as
        well as onto the model's attention processors -- deliberately, "so that it propagates
        gracefully throughout". If diffusers ever stops doing that, a per-model override could
        disagree with the estimate, and this test is where that shows up."""
        from diffusers.configuration_utils import ConfigMixin, register_to_config
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry
        from diffusers.models.modeling_utils import ModelMixin

        class _Tiny(ModelMixin, ConfigMixin):
            @register_to_config
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(2, 2)

        previous = _AttentionBackendRegistry._active_backend
        try:
            _Tiny().set_attention_backend("_native_math")
            assert _diffusers_attention_dispatch() == "math"
        finally:
            _AttentionBackendRegistry._active_backend = previous

    def test_an_unreadable_dispatcher_is_budgeted_as_math(self):
        """`_AttentionBackendRegistry` is private; if diffusers moves it we lose the answer. The
        conservative reading is the materializing one, and it is logged rather than silent."""
        with patch(
            "diffusers.models.attention_dispatch._AttentionBackendRegistry.get_active_backend",
            side_effect=AttributeError("moved"),
        ):
            assert _diffusers_attention_dispatch() == "math"


class TestTheInt8DequantTransientReachesTheReservation:
    """A helper existing is not the same as it being called.

    An `int8_tensorwise` build materializes each linear's dequantized weight inside `forward`, and
    that peak is not part of the model's resident size. Pricing another module than the one about to
    be locked leaves every other test in this file green, because they drive it with a `MagicMock` whose
    module tree is empty -- so this is the only place the wire is pinned. For Klein 9B the term is
    576 MiB, against a card this build exists to fit into.
    """

    @staticmethod
    def _int8_transformer() -> torch.nn.Module:
        from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear

        model = torch.nn.Module()
        model.dense = torch.nn.Linear(64, 64, bias=False).to(torch.bfloat16)
        model.quantized = Int8ConvrotLinear(
            weight=torch.zeros(128, 64, dtype=torch.int8),
            weight_scale=torch.ones(()),
            convrot=False,
        )
        return model

    def test_an_int8_build_adds_its_transient_on_top_of_the_activation_estimate(self):
        harness = TestFlux2DenoiseRequestsWorkingMemory()
        model = self._int8_transformer()

        dense = harness._run(num_ref_tokens=0)
        with_int8 = harness._run(num_ref_tokens=0, transformer=model)

        # Two weight-sized tensors in the compute dtype, over the largest quantized layer, plus that layer's stored
        # int8 weight and float32 scale, which partial loading would stream.
        assert with_int8 - dense == 2 * 128 * 64 * torch.bfloat16.itemsize + 128 * 64 + 4

    def test_a_dense_build_adds_nothing(self):
        # The other half: a bf16 or fp8 checkpoint must not be charged for a dequantization it
        # never performs, only for its largest layer as stored, which partial loading would stream.
        harness = TestFlux2DenoiseRequestsWorkingMemory()
        dense_model = torch.nn.Linear(64, 64).to(torch.bfloat16)

        added = harness._run(num_ref_tokens=0, transformer=dense_model) - harness._run(num_ref_tokens=0)
        assert added == (64 * 64 + 64) * torch.bfloat16.itemsize


class TestTheSidecarDecisionReachesThePatcher:
    """An int8 checkpoint carries `ModelFormat.Checkpoint` like any other single file.

    Asking the format alone therefore answers "not quantized", and `LayerPatcher` picks direct
    patching -- which calls `get_parameter("weight")` on an `Int8ConvrotLinear` that owns only
    buffers. Verified consequence: `RuntimeError: generator raised StopIteration`, i.e. every LoRA
    on an int8 Klein 9B dies with an error that names nothing. So the question has to be put to the
    loaded module tree.
    """

    @staticmethod
    def _sidecar_flag_for(transformer: torch.nn.Module, model_format) -> bool:
        from invokeai.backend.quantization.dequantizing_linear import requires_sidecar_patching

        return requires_sidecar_patching(transformer, model_format)

    def test_an_int8_checkpoint_is_patched_as_a_sidecar(self):
        from invokeai.backend.model_manager.taxonomy import ModelFormat

        model = TestTheInt8DequantTransientReachesTheReservation._int8_transformer()

        assert self._sidecar_flag_for(model, ModelFormat.Checkpoint) is True

    def test_a_dense_checkpoint_is_not(self):
        from invokeai.backend.model_manager.taxonomy import ModelFormat

        assert self._sidecar_flag_for(torch.nn.Linear(4, 4), ModelFormat.Checkpoint) is False

    @staticmethod
    def _flag_at_the_node(transformer: torch.nn.Module) -> bool:
        """What the node actually hands `LayerPatcher`, driven through `_run_diffusion`.

        The wire, not the helper: the node used to key on a hardcoded format list, and a list
        cannot see an int8 build at all.
        """
        from contextlib import contextmanager

        captured: dict = {}

        @contextmanager
        def on_device(**_kwargs):
            yield (None, transformer)

        def apply_patches(**kwargs):
            captured["force_sidecar_patching"] = kwargs["force_sidecar_patching"]
            raise _StopBeforeLoad

        harness = TestFlux2DenoiseRequestsWorkingMemory()
        with patch.object(LayerPatcher, "apply_smart_model_patches", staticmethod(apply_patches)):
            harness._run(num_ref_tokens=0, transformer=transformer, on_device=on_device)
        return captured["force_sidecar_patching"]

    def test_the_node_asks_the_model_and_not_only_the_format(self):
        int8 = TestTheInt8DequantTransientReachesTheReservation._int8_transformer()

        assert self._flag_at_the_node(int8) is True
        assert self._flag_at_the_node(torch.nn.Linear(4, 4).to(torch.bfloat16)) is False
