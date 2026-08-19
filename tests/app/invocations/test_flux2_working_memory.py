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
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from invokeai.app.invocations.flux2_denoise import Flux2DenoiseInvocation
from invokeai.app.invocations.flux2_vae_decode import Flux2VaeDecodeInvocation
from invokeai.app.invocations.flux2_vae_encode import Flux2VaeEncodeInvocation
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux2

MB = 1024**2
GB = 1024**3


def _estimate(image_seq_len, ref_image_seq_len=0, text_seq_len=512, num_loras=0, regional_bias=0):
    return Flux2DenoiseInvocation._estimate_working_memory(
        MagicMock(spec=Flux2DenoiseInvocation),
        image_seq_len=image_seq_len,
        ref_image_seq_len=ref_image_seq_len,
        text_seq_len=text_seq_len,
        num_loras=num_loras,
        regional_attention_bias_bytes=regional_bias,
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
            operation=operation, image_tensor=self._tensor_for(operation, px), vae=self._mock_bf16_vae()
        )
        assert estimate >= measured_mb * MB

    @pytest.mark.parametrize("operation, expected_constant", [("decode", 2200), ("encode", 1100)])
    def test_constant_scales_pixel_area_and_element_size(self, operation, expected_constant):
        estimate = estimate_vae_working_memory_flux2(
            operation=operation, image_tensor=self._tensor_for(operation, 1024), vae=self._mock_bf16_vae()
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
            )
            for px in (1024, 1328, 2024)
        ]
        assert len(set(estimates)) == 1
        assert estimates[0] == int(512 * 512 * 2 * 1100 * 1.25)
        assert estimates[0] >= 558 * MB  # measured tiled peak at 2024px
        # The whole point of tiling: it must shrink the reservation, not just bound the VAE.
        untiled = estimate_vae_working_memory_flux2(
            operation="encode", image_tensor=torch.zeros(1, 3, 2024, 2024), vae=self._mock_bf16_vae()
        )
        assert estimates[0] < untiled / 4


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

    def _run(self, num_ref_tokens: int):
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
