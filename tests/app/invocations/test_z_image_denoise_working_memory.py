"""Z-Image denoise working memory: the estimate, and what the node hands `model_on_device()`.

Until this estimate existed the node reserved only the per-forward dequant transient, so the model cache kept just
`device_working_mem_gb` free. Where SDPA materializes the score matrices -- ROCm on Windows, MPS -- a 1024px
generation needs several times that, and on Windows the driver pages the shortfall into shared system memory instead
of failing: 7.7 GB of it, and a 185 s denoise, on an RX 9060 XT.

The `MEASURED_*` tables are peak *reserved* memory on CUDA (RTX 4090) in bf16 with a 512-token caption, from
`scripts/calibrate_z_image_working_memory.py`. Every estimate must stay an upper bound on them.

MiniMax H3 and Krea-2 add the same dequant transient to their own estimates. Krea-2's wiring is pinned in
`tests/app/invocations/test_krea2_denoise.py` and FLUX.2's in `tests/app/invocations/test_flux2_working_memory.py`;
MiniMax H3's is not, since reaching its `model_on_device()` call needs a packed sequence.
"""

from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.fields import ZImageConditioningField
from invokeai.app.invocations.z_image.z_image_control import ZImageControlField
from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from invokeai.backend.util import attention
from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT
from invokeai.backend.z_image.extensions.regional_prompting_extension import ZImageRegionalPromptingExtension
from invokeai.backend.z_image.text_conditioning import ZImageRegionalTextConditioning
from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter

MB = 1024**2
GB = 1024**3

# Only the device's type is read: every question the estimate asks about it is answered by `_sdpa` below.
DEVICE = torch.device("cuda")


@contextmanager
def _sdpa(materializes):
    """Answer the three dispatch questions `sdpa_score_matrix_bytes` asks, so no test depends on the torch build, the
    diffusers attention backend or the ROCm head-dim guard of the machine it runs on."""
    with (
        patch.object(attention, "_diffusers_attention_dispatch", return_value=attention._DISPATCH_TORCH),
        patch.object(attention, "rocm_sdpa_uses_math_kernel", return_value=False),
        patch.object(
            attention, "_torch_sdpa_materializes_score_matrix", side_effect=lambda *args: materializes(args[-1])
        ),
    ):
        yield


def _estimate(image_seq_len, text_seq_len=512, num_control_blocks=0, has_mask=False, materializes=lambda mask: False):
    with _sdpa(materializes):
        return ZImageDenoiseInvocation._estimate_working_memory(
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
            num_loras=0,
            num_control_blocks=num_control_blocks,
            has_attention_mask=has_mask,
            device=DEVICE,
        )


def _image_tokens(px: int) -> int:
    return (px // 16) ** 2


class TestEstimate:
    # (pixels per side, ControlNet blocks, measured peak reserved MiB)
    MEASURED_FUSED = [
        (1024, 0, 1012),
        (1440, 0, 2004),
        (2048, 0, 3942),
        (1024, 1, 1122),
        (1024, 6, 1212),
        (1024, 15, 1650),
    ]

    @pytest.mark.parametrize("px, control_blocks, measured_mb", MEASURED_FUSED)
    def test_estimate_is_an_upper_bound_on_measured_peak(self, px, control_blocks, measured_mb):
        """The cache treats the estimate as the amount it must keep free, so under-estimating runs out of memory."""
        assert _estimate(_image_tokens(px), num_control_blocks=control_blocks) >= measured_mb * MB

    @pytest.mark.parametrize("px, control_blocks, measured_mb", MEASURED_FUSED)
    def test_estimate_does_not_wildly_over_reserve(self, px, control_blocks, measured_mb):
        """Every byte reserved beyond need is transformer offloaded to RAM and run over PCIe."""
        assert _estimate(_image_tokens(px), num_control_blocks=control_blocks) <= measured_mb * MB + 2 * GB

    def test_image_and_caption_are_counted_as_the_transformer_pads_them(self):
        """Each run is padded to a multiple of 32 before the two are joined, and the padding is attended too."""
        assert _estimate(4096, text_seq_len=97) == _estimate(4096, text_seq_len=128)
        assert _estimate(4001, text_seq_len=128) == _estimate(4032, text_seq_len=128)
        assert _estimate(4096, text_seq_len=129) > _estimate(4096, text_seq_len=128)


class TestMaterializedScoreMatrix:
    def test_the_score_matrix_is_charged_only_where_it_is_built(self):
        """Measured at 1024px: a forced math kernel took 9.4 GiB where the fused one took 1.0. The attention mask
        decides the dispatch, so it has to reach the probe."""
        seq_len = 4096 + 512
        unmasked = _estimate(4096, materializes=lambda has_mask: has_mask)
        masked = _estimate(4096, has_mask=True, materializes=lambda has_mask: has_mask)

        assert unmasked == _estimate(4096)
        assert masked - unmasked == 30 * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT

    def test_estimate_bounds_the_measured_math_kernel(self):
        assert _estimate(_image_tokens(1024), materializes=lambda has_mask: True) >= 9632 * MB


class TestRegionalAttentionBias:
    def test_no_regional_mask_costs_nothing(self):
        assert ZImageDenoiseInvocation._regional_attention_bias_bytes(None, 4096, torch.bfloat16) == 0

    def test_bias_is_sized_on_the_padded_unified_sequence(self):
        """4096 image tokens and a 100-token caption: the boolean copy covers the 4196 real tokens, the additive bias
        the 4096 + 128 padded ones, and `torch.where` fills the image block from a 4096 x 4096 scratch."""
        mask = torch.empty((4196, 4196), dtype=torch.float16, device="meta")

        bias_bytes = ZImageDenoiseInvocation._regional_attention_bias_bytes(mask, 4096, torch.bfloat16)

        assert bias_bytes == 4196**2 + 4224**2 * 2 + 4096**2 * 2


class _StopBeforeLoad(Exception):
    """Raised out of a mocked `model_on_device()` so the node stops at a reservation."""


# Stands in for the activation estimate, which is tested above; any value the transient cannot produce.
ACTIVATION_ESTIMATE = 3_000_000_007
POSITIVE_TOKENS = 8


def _dense_transformer() -> torch.nn.Module:
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 8)
    return model


class TestNodeReservation:
    """The wire from the node to `model_on_device()`: deleting any of these arguments leaves the estimate's own tests
    green."""

    def _run(self, transformer=None, *, control_blocks=None, negative_tokens=None, loras=0, regional_mask=None):
        from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
            ConditioningFieldData,
            ZImageConditioningInfo,
        )

        transformer = transformer if transformer is not None else _dense_transformer()
        transformer_info = MagicMock()
        transformer_info.model = transformer

        control = None
        control_info = MagicMock()
        if control_blocks is None:
            transformer_info.model_on_device = MagicMock(side_effect=_StopBeforeLoad)
        else:
            # Past the transformer's lock, up to the adapter's.
            transformer_info.model_on_device = MagicMock(return_value=nullcontext((None, transformer)))
            control = ZImageControlField.model_construct(control_model=MagicMock(), image_name="control")
            control_info.model = MagicMock(spec=ZImageControlAdapter)
            control_info.model.control_layers = [torch.nn.Identity()] * control_blocks
            control_info.model_on_device = MagicMock(side_effect=_StopBeforeLoad)

        caption_tokens = {"pos": POSITIVE_TOKENS, "neg": negative_tokens}
        context = MagicMock()
        context.models.load.side_effect = lambda identifier: (
            control_info if control is not None and identifier is control.control_model else transformer_info
        )
        context.models.get_config.return_value = SimpleNamespace(
            base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.Checkpoint
        )
        context.conditioning.load.side_effect = lambda name: ConditioningFieldData(
            conditionings=[ZImageConditioningInfo(prompt_embeds=torch.zeros(caption_tokens[name], 2560))]
        )

        invocation = ZImageDenoiseInvocation.model_construct(
            latents=None,
            noise=None,
            denoise_mask=None,
            denoising_start=0.0,
            denoising_end=1.0,
            add_noise=True,
            transformer=MagicMock(transformer=MagicMock(), loras=[MagicMock() for _ in range(loras)]),
            positive_conditioning=ZImageConditioningField(conditioning_name="pos"),
            negative_conditioning=ZImageConditioningField(conditioning_name="neg") if negative_tokens else None,
            control=control,
            vae=None,
            guidance_scale=2.0 if negative_tokens else 1.0,
            width=256,
            height=256,
            steps=2,
            scheduler="euler",
            seed=0,
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "invokeai.app.invocations.z_image.z_image_denoise.TorchDevice.choose_torch_device",
                    return_value=torch.device("cpu"),
                )
            )
            stack.enter_context(
                patch(
                    "invokeai.app.invocations.z_image.z_image_denoise.TorchDevice.choose_bfloat16_safe_dtype",
                    return_value=torch.bfloat16,
                )
            )
            stack.enter_context(
                patch.object(ZImageDenoiseInvocation, "_get_noise", return_value=torch.zeros(1, 16, 32, 32))
            )
            estimate = stack.enter_context(
                patch.object(ZImageDenoiseInvocation, "_estimate_working_memory", return_value=ACTIVATION_ESTIMATE)
            )
            if regional_mask is not None:
                regional = ZImageRegionalPromptingExtension(
                    regional_text_conditioning=ZImageRegionalTextConditioning(
                        prompt_embeds=torch.zeros(POSITIVE_TOKENS, 2560), image_masks=[None], embedding_ranges=[]
                    ),
                    regional_attn_mask=regional_mask,
                )
                stack.enter_context(
                    patch.object(ZImageRegionalPromptingExtension, "from_text_conditionings", return_value=regional)
                )
            with pytest.raises(_StopBeforeLoad):
                invocation._run_diffusion(context)

        return SimpleNamespace(
            reserved=transformer_info.model_on_device.call_args.kwargs["working_mem_bytes"],
            estimate=estimate.call_args.kwargs,
            adapter_reserved=(
                control_info.model_on_device.call_args.kwargs.get("working_mem_bytes") if control is not None else None
            ),
        )

    def test_an_unquantized_transformer_reserves_the_estimate_for_its_sequence(self):
        run = self._run()

        assert run.reserved == ACTIVATION_ESTIMATE
        # 256px is a 16x16 token grid.
        assert (run.estimate["image_seq_len"], run.estimate["text_seq_len"]) == (256, POSITIVE_TOKENS)
        assert (run.estimate["num_loras"], run.estimate["num_control_blocks"]) == (0, 0)
        assert (run.estimate["regional_attention_bias_bytes"], run.estimate["has_attention_mask"]) == (0, False)

    def test_the_dequant_transient_is_added_to_the_estimate(self):
        """An int8 layer materializes two weight-sized bf16 tensors per forward, alive alongside the activations."""
        model = torch.nn.Module()
        model.proj = Int8ConvrotLinear(
            torch.zeros(1024, CONVROT_GROUP_SIZE, dtype=torch.int8), torch.ones(1024, 1), convrot=False
        )

        run = self._run(model)

        assert run.reserved == ACTIVATION_ESTIMATE + 2 * 1024 * CONVROT_GROUP_SIZE * torch.bfloat16.itemsize

    def test_a_longer_negative_caption_sizes_the_sequence(self):
        """With CFG the unconditional pass runs its own forward, so the longer of the two captions is the peak."""
        assert self._run(negative_tokens=40).estimate["text_seq_len"] == 40

    def test_loras_are_counted(self):
        assert self._run(loras=2).estimate["num_loras"] == 2

    def test_a_regional_mask_reaches_the_estimate(self):
        mask = torch.zeros((256 + POSITIVE_TOKENS, 256 + POSITIVE_TOKENS), dtype=torch.float16)

        run = self._run(regional_mask=mask)

        expected_bias = ZImageDenoiseInvocation._regional_attention_bias_bytes(mask, 256, torch.bfloat16)
        assert (run.estimate["regional_attention_bias_bytes"], run.estimate["has_attention_mask"]) == (
            expected_bias,
            True,
        )

    def test_a_control_adapter_reaches_the_estimate_and_keeps_the_reservation(self):
        """The adapter's block count sizes the hints and its forward always passes a padding mask. It is locked after
        the transformer, which can then no longer be offloaded -- so it must ask for the same working memory, or its
        weights fill the room kept for the forward."""
        run = self._run(control_blocks=15)

        assert (run.estimate["num_control_blocks"], run.estimate["has_attention_mask"]) == (15, True)
        assert run.adapter_reserved == run.reserved == ACTIVATION_ESTIMATE
