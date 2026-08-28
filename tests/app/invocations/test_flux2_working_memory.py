"""FLUX.2 working-memory estimates: the transformer denoise and both VAE directions.

The FLUX.2 path originally called `model_on_device()` with no `working_mem_bytes` anywhere, so the
model cache reserved only the small default `device_working_mem_gb` and filled the rest of the card
with the model. Reference images make that fatal rather than merely tight: their latents are
concatenated onto the image stream, so three 1024x1024 references quadruple the attended sequence of
a 1024x1024 generation. See https://github.com/invoke-ai/InvokeAI/issues/9500.

The `MEASURED_*` tables below are peak *reserved* memory measured on CUDA in bf16 (the conservative
quantity, including allocator overhead). Every estimate must stay an upper bound on them.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from invokeai.app.invocations.flux2_denoise import FLUX2_MAX_ATTENTION_HEADS, Flux2DenoiseInvocation
from invokeai.app.invocations.flux2_vae_decode import Flux2VaeDecodeInvocation
from invokeai.app.invocations.flux2_vae_encode import Flux2VaeEncodeInvocation
from invokeai.backend.util.attention import (
    SDPA_MATH_BYTES_PER_SCORE_ELEMENT,
    _diffusers_attention_dispatch,
    _torch_sdpa_materializes_score_matrix,
    sdpa_score_matrix_bytes,
)
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux2

MB = 1024**2
GB = 1024**3

# The measured tables in this module were all taken on CUDA, where SDPA runs a fused kernel and no
# score matrix is materialized. torch reports its CPU flash kernel as eligible for every shape used
# here, so passing a CPU device reproduces that regime without needing a GPU on the test runner. The
# materializing regime gets its own class below.
FUSED = torch.device("cpu")


def _estimate(
    image_seq_len,
    ref_image_seq_len=0,
    text_seq_len=512,
    num_loras=0,
    batch_size=1,
    regional_bias=0,
    has_regional_mask=False,
    device=FUSED,
):
    return Flux2DenoiseInvocation._estimate_working_memory(
        MagicMock(spec=Flux2DenoiseInvocation),
        image_seq_len=image_seq_len,
        ref_image_seq_len=ref_image_seq_len,
        text_seq_len=text_seq_len,
        num_loras=num_loras,
        batch_size=batch_size,
        regional_attention_bias_bytes=regional_bias,
        has_regional_attention_mask=has_regional_mask,
        device=device,
    )


class TestFlux2DenoiseWorkingMemoryEstimate:
    # (image tokens, reference tokens, measured peak reserved MB) on the Klein 9B geometry.
    # Token grids are pixels/16, so a 1024px square is 4096 tokens.
    MEASURED_DENOISE = [
        (1024, 0, 448),  # 512px
        (4096, 0, 1702),  # 1024px
        (4096, 4096, 3324),  # 1024px + one 1024px reference
        (4096, 8192, 4840),  # + two references
        (4096, 12288, 6538),  # + three references (the tiled-refiner case from #9500)
        (6889, 12288, 7528),  # 1328px tile + three 1024px references
        (6889, 20667, 10954),  # 1328px tile + three 1328px references
        (16384, 0, 6538),  # 2048px, no references
    ]

    @pytest.mark.parametrize("image_seq_len, ref_image_seq_len, measured_mb", MEASURED_DENOISE)
    def test_estimate_is_an_upper_bound_on_measured_peak(self, image_seq_len, ref_image_seq_len, measured_mb):
        """The cache treats the estimate as the amount it must keep free, so under-estimating OOMs."""
        assert _estimate(image_seq_len, ref_image_seq_len) >= measured_mb * MB

    @pytest.mark.parametrize("image_seq_len, ref_image_seq_len, measured_mb", MEASURED_DENOISE)
    def test_estimate_does_not_wildly_over_reserve(self, image_seq_len, ref_image_seq_len, measured_mb):
        """Over-estimating is not free: the cache offloads the transformer to RAM to honor the
        reservation, and a model running over PCIe is indistinguishable from a hang."""
        assert _estimate(image_seq_len, ref_image_seq_len) <= measured_mb * MB + 2 * GB

    def test_reference_image_tokens_are_counted(self):
        """The regression this whole module exists for: reference tokens are attended like image
        tokens and cost the same per token, so they must enter the estimate."""
        without_refs = _estimate(image_seq_len=4096)
        with_refs = _estimate(image_seq_len=4096, ref_image_seq_len=12288)
        assert with_refs - without_refs == 12288 * int(0.4 * MB)

    def test_estimate_is_linear_in_total_sequence(self):
        """Attention runs through SDPA, so there is no O(seq^2) term to model -- image, reference and
        text tokens are interchangeable at the same per-token cost."""
        assert _estimate(image_seq_len=8192) == _estimate(image_seq_len=4096, ref_image_seq_len=4096)
        assert _estimate(image_seq_len=4096, text_seq_len=1024) - _estimate(image_seq_len=4096, text_seq_len=512) == (
            512 * int(0.4 * MB)
        )

    def test_lora_margin_is_added_per_lora(self):
        """Sidecar-patched LoRAs add an activation branch per patched layer."""
        base = _estimate(image_seq_len=4096)
        assert _estimate(image_seq_len=4096, num_loras=1) - base == int(0.5 * GB)
        assert _estimate(image_seq_len=4096, num_loras=3) - base == int(1.5 * GB)

    def test_regional_attention_bias_is_added(self):
        base = _estimate(image_seq_len=4096)
        assert _estimate(image_seq_len=4096, regional_bias=123 * MB) - base == 123 * MB


class TestFlux2DenoiseBatchIsBudgeted:
    """A batch of B is B independent sequences, so it enters the linear term exactly as extra
    sequence does. Measured on the Klein geometry (48 heads x 128, mlp 3.0) with a reduced block
    count -- the constant is block-count independent -- peak reserved, each point in a fresh process:

        B=1, 4608 tokens ->  2570MB      B=2, 4608 each (9216 total) ->  5126MB
        B=1, 9728 tokens ->  5584MB      B=2, 9728 each (19456 total) -> 11120MB
        B=1, 14336 tokens -> 8284MB      B=3, 4608 each (13824 total) ->  7656MB

    Per *total* token that is 0.554-0.578MB across every row: batch and sequence are interchangeable.
    (The absolute figure is not comparable to the Klein table elsewhere in this module -- a 3-block
    stand-in amortizes per-forward overhead differently. Only the equivalence is being tested.)

    Batched latents reach this node through the API and custom graphs, not the stock UI.
    """

    def test_batch_and_sequence_are_interchangeable(self):
        """Two samples of 4608 tokens must cost what one sample of 9216 costs -- the measurement
        above says 5126MB against 5584MB, equal to within the allocator's noise."""
        assert _estimate(image_seq_len=4096, text_seq_len=512, batch_size=2) == _estimate(
            image_seq_len=8704, text_seq_len=512, batch_size=1
        )

    @pytest.mark.parametrize("batch", [2, 3, 4])
    def test_each_extra_sample_adds_exactly_its_own_tokens(self, batch):
        single = _estimate(image_seq_len=4096)
        assert _estimate(image_seq_len=4096, batch_size=batch) - single == (batch - 1) * (4096 + 512) * int(0.4 * MB)

    def test_reference_tokens_scale_with_the_batch(self):
        """`ensure_batch_size` repeats the reference latents across the batch."""
        single = _estimate(image_seq_len=4096, ref_image_seq_len=12288)
        assert _estimate(image_seq_len=4096, ref_image_seq_len=12288, batch_size=2) - single == (
            (4096 + 12288 + 512) * int(0.4 * MB)
        )

    def test_the_fixed_base_does_not_scale_with_the_batch(self):
        """It covers transient weight casts and allocator slack -- properties of the weights, not of
        how many samples run through them. Scaling it would add a GB per sample for nothing."""
        deltas = {
            _estimate(image_seq_len=4096, batch_size=b + 1) - _estimate(image_seq_len=4096, batch_size=b)
            for b in (1, 2, 3)
        }
        assert deltas == {(4096 + 512) * int(0.4 * MB)}

    def test_the_regional_bias_does_not_scale_with_the_batch(self):
        """`get_joint_attention_kwargs` builds it as (1, 1, S, S) and lets SDPA broadcast it, so
        there is exactly one of them however many samples are in flight."""
        bias = (4096 + 512) ** 2 * 2
        single = _estimate(image_seq_len=4096, regional_bias=bias, has_regional_mask=True)
        double = _estimate(image_seq_len=4096, regional_bias=bias, has_regional_mask=True, batch_size=2)
        assert double - single == (4096 + 512) * int(0.4 * MB)

    def test_the_score_matrix_scales_with_the_batch(self):
        """Where it is materialized at all it is shaped (batch, heads, S, S)."""
        seq_len = 4096 + 512
        with _materializing():
            single = _estimate(image_seq_len=4096, has_regional_mask=True, device=MATERIALIZING)
            double = _estimate(image_seq_len=4096, has_regional_mask=True, batch_size=2, device=MATERIALIZING)
        assert double - single == (
            seq_len * int(0.4 * MB) + FLUX2_MAX_ATTENTION_HEADS * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        )

    def test_the_lora_margin_scales_with_the_batch(self):
        """A sidecar patch adds an activation branch, and activations are per sample."""
        assert _estimate(image_seq_len=4096, num_loras=2, batch_size=3) - _estimate(
            image_seq_len=4096, num_loras=0, batch_size=3
        ) == 3 * int(1.0 * GB)


class TestFlux2VaeWorkingMemoryEstimate:
    # (operation, pixel size, measured peak reserved MB), bf16, untiled.
    MEASURED_VAE = [
        ("decode", 512, 1086),
        ("decode", 768, 2414),
        ("decode", 1024, 4260),
        ("decode", 1328, 7146),
        ("decode", 1536, 9578),
        ("encode", 512, 536),
        ("encode", 1024, 2122),
        ("encode", 1328, 3022),
    ]

    def _mock_bf16_vae(self):
        vae = MagicMock(spec=AutoencoderKLFlux2)
        vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])  # element_size == 2
        return vae

    def _tensor_for(self, operation, px):
        # decode receives 32-channel latents at pixels/8; encode receives a pixel image.
        return torch.zeros(1, 32, px // 8, px // 8) if operation == "decode" else torch.zeros(1, 3, px, px)

    @pytest.mark.parametrize("operation, px, measured_mb", MEASURED_VAE)
    def test_estimate_is_an_upper_bound_on_measured_peak(self, operation, px, measured_mb):
        estimate = estimate_vae_working_memory_flux2(
            operation=operation, image_tensor=self._tensor_for(operation, px), vae=self._mock_bf16_vae(), device=FUSED
        )
        assert estimate >= measured_mb * MB

    @pytest.mark.parametrize("operation, expected_constant", [("decode", 2200), ("encode", 1100)])
    def test_constant_scales_pixel_area_and_element_size(self, operation, expected_constant):
        estimate = estimate_vae_working_memory_flux2(
            operation=operation, image_tensor=self._tensor_for(operation, 1024), vae=self._mock_bf16_vae(), device=FUSED
        )
        assert estimate == 1024 * 1024 * 2 * expected_constant

    def test_tiled_estimate_is_bounded_by_the_tile_not_the_image(self):
        """Reference-image encoding forces 512px tiling precisely so the peak stops following the
        reference resolution -- measured flat at ~0.55GB from 1024px up to the 2024px reference cap."""
        estimates = [
            estimate_vae_working_memory_flux2(
                operation="encode",
                image_tensor=torch.zeros(1, 3, px, px),
                vae=self._mock_bf16_vae(),
                tile_size=512,
                device=FUSED,
            )
            for px in (1024, 1328, 2024)
        ]
        assert len(set(estimates)) == 1
        assert estimates[0] == int(512 * 512 * 2 * 1100 * 1.25)
        assert estimates[0] >= 558 * MB  # measured tiled peak at 2024px
        # The whole point of tiling: it must shrink the reservation, not just bound the VAE.
        untiled = estimate_vae_working_memory_flux2(
            operation="encode", image_tensor=torch.zeros(1, 3, 2024, 2024), vae=self._mock_bf16_vae(), device=FUSED
        )
        assert estimates[0] < untiled / 4


class TestFlux2VaeBatchIsBudgeted:
    """`vae.decode` is handed whatever batch the latents carry, and a `LatentsField` is not pinned to
    one. An estimate built from H and W alone gives a two-sample decode the same reservation as a
    single one, so the cache admits it to a card that cannot run it -- the reservation is there, and
    the OOM happens anyway.

    Measured at 1024px on CUDA/bf16, peak reserved, each point in a fresh process: 4.23GB at batch 1,
    7.96GB at batch 2, 11.89GB at batch 3. Linear, and slightly sub-linear per sample, so scaling the
    single-sample estimate is an upper bound rather than a fit.
    """

    # (batch, measured peak reserved MB) for a 1024px decode.
    MEASURED_DECODE_BATCH = [(1, 4229), (2, 7955), (3, 11889)]

    def _decode_estimate(self, batch, px=1024, tile_size=None, device=FUSED):
        vae = MagicMock(spec=AutoencoderKLFlux2)
        vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
        return estimate_vae_working_memory_flux2(
            operation="decode",
            image_tensor=torch.zeros(batch, 32, px // 8, px // 8),
            vae=vae,
            tile_size=tile_size,
            device=device,
        )

    @pytest.mark.parametrize("batch, measured_mb", MEASURED_DECODE_BATCH)
    def test_estimate_is_an_upper_bound_on_the_measured_batch_peak(self, batch, measured_mb):
        estimate = self._decode_estimate(batch)
        assert estimate >= measured_mb * MB
        assert estimate <= 2 * measured_mb * MB

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
        )
        assert unbatched == self._decode_estimate(1)

    def test_tiling_bounds_the_tile_not_the_batch(self):
        """Tiling caps the spatial term at one tile, but every sample still runs through it."""
        single = self._decode_estimate(1, px=1024, tile_size=512)
        assert self._decode_estimate(3, px=1024, tile_size=512) == 3 * single

    def test_the_score_matrix_scales_with_the_batch(self):
        """It is shaped (batch, heads, S, S), so where it is materialized at all it scales too."""
        tokens = 128 * 128
        score = tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        linear = self._decode_estimate(1)  # fused: the spatial term on its own
        with _materializing():
            assert self._decode_estimate(1, device=MATERIALIZING) == linear + score
            assert self._decode_estimate(3, device=MATERIALIZING) == 3 * (linear + score)


class TestFlux2VaeInvocationsRequestWorkingMemory:
    """The estimate is worthless unless it reaches `model_on_device()`."""

    def _mock_vae_info(self):
        vae = MagicMock(spec=AutoencoderKLFlux2)
        vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])

        vae_info = MagicMock()
        vae_info.model = vae
        vae_info.compute_device = torch.device("cpu")
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=(None, vae))
        cm.__exit__ = MagicMock(return_value=None)
        vae_info.model_on_device = MagicMock(return_value=cm)
        return vae_info

    def test_decode_requests_working_memory(self):
        vae_info = self._mock_vae_info()
        context = MagicMock()
        context.models.load.return_value = vae_info
        context.tensors.load.return_value = torch.zeros(1, 32, 128, 128)

        expected = 10 * GB
        with patch(
            "invokeai.app.invocations.flux2_vae_decode.estimate_vae_working_memory_flux2", return_value=expected
        ) as estimate:
            invocation = Flux2VaeDecodeInvocation.model_construct(
                latents=MagicMock(latents_name="latents"), vae=MagicMock(vae=MagicMock())
            )
            try:
                invocation.invoke(context)
            except Exception:
                # The mocked decode math fails downstream; we only care that the cache was asked to
                # reserve the estimate before the device context was entered.
                pass

        estimate.assert_called_once()
        assert estimate.call_args.kwargs["operation"] == "decode"
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected)

    def test_encode_requests_working_memory(self):
        vae_info = self._mock_vae_info()
        context = MagicMock()
        context.models.load.return_value = vae_info

        expected = 4 * GB
        with (
            patch(
                "invokeai.app.invocations.flux2_vae_encode.estimate_vae_working_memory_flux2", return_value=expected
            ) as estimate,
            patch(
                "invokeai.app.invocations.flux2_vae_encode.image_resized_to_grid_as_tensor",
                return_value=torch.zeros(3, 1024, 1024),
            ),
        ):
            invocation = Flux2VaeEncodeInvocation.model_construct(
                image=MagicMock(image_name="image"), vae=MagicMock(vae=MagicMock())
            )
            try:
                invocation.invoke(context)
            except Exception:
                pass

        estimate.assert_called_once()
        assert estimate.call_args.kwargs["operation"] == "encode"
        vae_info.model_on_device.assert_called_once_with(working_mem_bytes=expected)


class _StopBeforeLoad(Exception):
    """Raised in place of entering the transformer's device context, to end _run_diffusion early."""


class TestFlux2DenoiseRequestsWorkingMemory:
    """The denoise node must hand its estimate to the cache, and that estimate must grow with the
    attached reference images -- the combination that #9500 was missing."""

    def _run(self, num_ref_tokens: int, batch: int = 1):
        """Drive `_run_diffusion` up to the transformer load and return the requested working memory."""
        from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
        from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
            ConditioningFieldData,
            FLUXConditioningInfo,
        )

        transformer_info = MagicMock()
        transformer_info.model_on_device = MagicMock(side_effect=_StopBeforeLoad)

        context = MagicMock()
        context.models.load.return_value = transformer_info
        context.models.get_config.return_value = MagicMock(
            base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.Checkpoint
        )
        context.conditioning.load.return_value = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=torch.zeros(1, 768), t5_embeds=torch.zeros(1, 512, 12288))]
        )

        ref_extension = MagicMock()
        ref_extension.ref_image_latents = torch.zeros(1, num_ref_tokens, 128)

        invocation = Flux2DenoiseInvocation.model_construct(
            latents=None,
            noise=None,
            denoise_mask=None,
            denoising_start=0.0,
            denoising_end=1.0,
            add_noise=True,
            transformer=MagicMock(transformer=MagicMock(), loras=[]),
            positive_text_conditioning=MagicMock(conditioning_name="pos", mask=None),
            negative_text_conditioning=None,
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
            patch("invokeai.app.invocations.flux2_denoise.Flux2RefImageExtension", return_value=ref_extension),
            patch.object(
                Flux2DenoiseInvocation, "_prepare_noise_tensor", return_value=torch.zeros(batch, 32, 128, 128)
            ),
            pytest.raises(_StopBeforeLoad),
        ):
            invocation._run_diffusion(context)

        transformer_info.model_on_device.assert_called_once()
        return transformer_info.model_on_device.call_args.kwargs["working_mem_bytes"]

    def test_estimate_reaches_the_model_cache(self):
        """Without this the cache reserves only the default `device_working_mem_gb`."""
        assert self._run(num_ref_tokens=0) == _estimate(image_seq_len=64 * 64, text_seq_len=512)

    def test_reference_images_raise_the_reservation(self):
        """Three 1024x1024 references add 12288 tokens to a 1024x1024 generation's 4096."""
        without_refs = self._run(num_ref_tokens=0)
        with_refs = self._run(num_ref_tokens=12288)
        assert with_refs - without_refs == 12288 * int(0.4 * MB)

    def test_the_real_batch_reaches_the_reservation(self):
        """A batched latent tensor is reachable through the API and custom graphs. The node has `b`
        in hand at the estimate; before this it simply did not pass it, so a two-sample run reserved
        one sample's worth and the cache admitted it to a card that could not run it."""
        assert self._run(num_ref_tokens=0, batch=2) == _estimate(image_seq_len=64 * 64, text_seq_len=512, batch_size=2)

    def test_a_batched_run_reserves_more_than_a_single_one(self):
        single = self._run(num_ref_tokens=0, batch=1)
        assert self._run(num_ref_tokens=0, batch=2) - single == (64 * 64 + 512) * int(0.4 * MB)

    def test_repeated_reference_latents_are_counted_per_sample(self):
        """`ensure_batch_size` repeats the reference latents across the batch, so their tokens scale
        with it as well -- the worst case in #9500, doubled."""
        single = self._run(num_ref_tokens=12288, batch=1)
        double = self._run(num_ref_tokens=12288, batch=2)
        assert double - single == (64 * 64 + 12288 + 512) * int(0.4 * MB)


def _rocm_like_probe(device_type, device_index, dtype, head_dim, has_attn_mask):
    """Stand in for the torch probe on a build with ROCm's fused-kernel rules.

    ROCm's fused SDPA kernels cap the head dim at 128 and do not take an arbitrary additive mask;
    anything else falls through to the `math` fallback, which materializes the score matrix. CUDA's
    memory-efficient kernel accepts both -- verified on torch 2.7.1+cu128, where
    `_fused_sdp_choice` reports the efficient kernel for the VAE's 512-wide head and for a masked
    128-wide transformer head, and measured peak stays linear in both cases -- which is why the
    estimates were linear to begin with.
    """
    return head_dim > 128 or has_attn_mask


def _materializing():
    return patch("invokeai.backend.util.attention._torch_sdpa_materializes_score_matrix", side_effect=_rocm_like_probe)


# Any CUDA device object works here: the probe is patched out, so nothing is allocated on it.
MATERIALIZING = torch.device("cuda")


class TestMaterializedScoreMatrixIsBudgeted:
    """The linear estimates above assume SDPA never builds the O(S^2) score matrix. That is a
    property of the *build*, not of FLUX.2: ROCm's fused kernels reject both the VAE's 512-wide
    attention head and the dense additive mask regional prompting attaches, and fall back to
    `math`. Where that happens the score matrix is the dominant term, so the estimate has to
    include it -- otherwise the fix works on CUDA and still OOMs on ROCm.
    """

    def _mock_bf16_vae(self):
        vae = MagicMock(spec=AutoencoderKLFlux2)
        vae.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
        return vae

    def _vae_estimate(self, operation, px, device, tile_size=None):
        tensor = torch.zeros(1, 32, px // 8, px // 8) if operation == "decode" else torch.zeros(1, 3, px, px)
        return estimate_vae_working_memory_flux2(
            operation=operation,
            image_tensor=tensor,
            vae=self._mock_bf16_vae(),
            tile_size=tile_size,
            device=device,
        )

    # (operation, pixel size, mid-block tokens). The VAE attends on the 8x-downsampled grid.
    @pytest.mark.parametrize(
        "operation, px, tokens",
        [
            ("decode", 1024, 128 * 128),
            ("decode", 1536, 192 * 192),
            ("encode", 1024, 128 * 128),
            ("encode", 1328, 166 * 166),
        ],
    )
    def test_vae_estimate_gains_exactly_the_score_matrix(self, operation, px, tokens):
        fused = self._vae_estimate(operation, px, device=FUSED)
        with _materializing():
            materializing = self._vae_estimate(operation, px, device=MATERIALIZING)
        assert materializing - fused == tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT

    def test_mps_style_dispatch_failure_reserves_the_vae_score_matrix(self):
        """The MPS case end to end, through the real probe rather than a stand-in: on a device torch
        cannot answer a dispatch query for, a 1024px decode has to come out ~3.5GB heavier than the
        linear term. Reporting those devices as fused -- as this PR first did -- is what let the
        decode be admitted to a card that could not run it."""
        with patch("torch.ops.aten._fused_sdp_choice", side_effect=NotImplementedError("no MPS kernel")):
            materializing = self._vae_estimate("decode", 1024, device=FUSED)
        fused = self._vae_estimate("decode", 1024, device=FUSED)

        tokens = 128 * 128
        assert materializing - fused == tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert materializing - fused > 3 * GB

    def test_vae_score_matrix_dominates_at_high_resolution(self):
        """The reviewer's case: a 1536px decode is ~9.6GB of linear activations on CUDA, and more
        than that again in scores where SDPA has to materialize them. An estimate that omits the
        term is not merely tight, it is wrong by a factor of three."""
        with _materializing():
            materializing = self._vae_estimate("decode", 1536, device=MATERIALIZING)
        assert materializing > 25 * GB
        assert materializing > 2 * self._vae_estimate("decode", 1536, device=FUSED)

    def test_tiled_vae_score_matrix_is_bounded_by_the_tile(self):
        """Tiling already bounds the linear term; it must bound the quadratic one too, or the
        reference-image encode would reserve as if it ran untiled."""
        with _materializing():
            estimates = [
                self._vae_estimate("encode", px, device=MATERIALIZING, tile_size=512) for px in (1024, 1328, 2024)
            ]
            untiled = self._vae_estimate("encode", 2024, device=MATERIALIZING)
        assert len(set(estimates)) == 1
        tile_tokens = (512 // 8) ** 2
        assert estimates[0] - self._vae_estimate("encode", 1024, device=FUSED, tile_size=512) == (
            tile_tokens * tile_tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        )
        # Tiling turns a 62GB reservation at the 2024px reference cap into under 1GB.
        assert estimates[0] < untiled / 50

    def test_regional_prompting_adds_the_score_matrix(self):
        """The dense `S x S` additive bias is what pushes SDPA off its fused kernel. Budgeting only
        the bias tensor -- as this PR first did -- under-reserves by the score matrix, which is two
        orders of magnitude larger."""
        seq_len = 4096 + 512
        bias_bytes = seq_len * seq_len * 2
        fused = _estimate(
            image_seq_len=4096, text_seq_len=512, regional_bias=bias_bytes, has_regional_mask=True, device=FUSED
        )
        with _materializing():
            materializing = _estimate(
                image_seq_len=4096,
                text_seq_len=512,
                regional_bias=bias_bytes,
                has_regional_mask=True,
                device=MATERIALIZING,
            )
        assert materializing - fused == (
            FLUX2_MAX_ATTENTION_HEADS * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        )
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
        """On CUDA both shapes are fused and this whole term is zero -- the fix is a no-op for the
        hardware the constants were measured on. On ROCm the same call reports the fallback and the
        term appears. Either answer is correct; the point is that it comes from torch."""
        vae_bytes = sdpa_score_matrix_bytes(
            device=torch.device("cuda"), dtype=torch.bfloat16, num_heads=1, head_dim=512, seq_len=16384
        )
        masked_bytes = sdpa_score_matrix_bytes(
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            num_heads=48,
            head_dim=128,
            seq_len=4608,
            has_attn_mask=True,
        )
        if torch.version.hip is None:
            assert vae_bytes == 0
            assert masked_bytes == 0
        else:
            assert vae_bytes > 0
            assert masked_bytes > 0

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
        assert estimate >= measured
        assert estimate <= 2 * measured


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
        assert forced_math - native == (
            FLUX2_MAX_ATTENTION_HEADS * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        )

    def test_a_fused_backend_leaves_the_denoise_estimate_linear(self):
        """`flash`, `sage`, `xformers` and friends exist precisely to avoid the score matrix; they
        must not be taxed for it, on any device."""
        with _diffusers_backend("flash"), _materializing():
            forced_flash = _estimate(image_seq_len=4096, has_regional_mask=True, device=MATERIALIZING)
        assert forced_flash == _estimate(image_seq_len=4096, has_regional_mask=True, device=FUSED)

    def test_the_vae_estimate_ignores_the_diffusers_backend(self):
        """`AttnProcessor2_0` bypasses the dispatcher, so the VAE's answer comes from torch alone."""

        def estimate():
            v = MagicMock(spec=AutoencoderKLFlux2)
            v.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
            return estimate_vae_working_memory_flux2(
                operation="decode", image_tensor=torch.zeros(1, 32, 128, 128), vae=v, device=FUSED
            )

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
        assert after_switch - native == (
            FLUX2_MAX_ATTENTION_HEADS * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        )
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
