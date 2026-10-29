"""Z-Image denoise working memory: the estimate, and what the node hands `model_on_device()`.

Until this estimate existed the node reserved only the per-forward dequant transient, so the model cache kept just
`device_working_mem_gb` free. Where SDPA materializes the score matrices -- ROCm on Windows, MPS -- a 1024px
generation needs several times that, and on Windows the driver pages the shortfall into shared system memory instead
of failing: 7.7 GB of it, and a 185 s denoise, on an RX 9060 XT.

The estimate prices with the lines `backend/working_memory/data/denoise.json` ships, the fit of the plain forwards
measured there, which `tests/backend/working_memory/` holds it to. `TestMeasuredChecks` holds it to what those points
leave out: a ControlNet adapter's hints and a forced math kernel's score matrix, measured by
`scripts/calibrate_z_image_working_memory.py` beside the plain forward of the same run.

ROCm with the native allocator off Windows is not measured and still prices with the hand-set `legacy` constants,
which `TestLegacyReservation` holds byte for byte to the formula the node priced with by hand, frozen below, for every
transformer that holds no float8 weight. An fp8-stored one also reserves its cast transient, a term of its own
(`TestEstimate`). Every other platform nothing is measured on prices with a measured one's lines, floored at those
constants, and `TestUnmeasuredPlatforms` holds it to the formula at least.

MiniMax H3 and Krea-2 add the same dequant transient to their own estimates. Krea-2's wiring is pinned in
`tests/app/invocations/test_krea2_denoise.py` and FLUX.2's in `tests/app/invocations/test_flux2_working_memory.py`;
MiniMax H3's is not, since reaching its `model_on_device()` call needs a packed sequence.
"""

import itertools
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, get_args
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from invokeai.app.invocations.fields import ImageField, ZImageConditioningField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.vae.z_image_image_to_latents import ZImageImageToLatentsInvocation
from invokeai.app.invocations.z_image.z_image_control import ZImageControlField, ZImageControlInvocation
from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from invokeai.backend.util import attention
from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT, sdpa_score_matrix_bytes
from invokeai.backend.working_memory import spec as spec_module
from invokeai.backend.working_memory.coefficients import resolve_coefficients, shipped_coefficients, shipped_data
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.platform import Allocator, ConvBackend, OperatingSystem, Platform, is_measured
from invokeai.backend.z_image.denoise_working_memory import (
    Z_IMAGE_DENOISE_SPEC,
    estimate_z_image_denoise_working_memory,
)
from invokeai.backend.z_image.extensions.regional_prompting_extension import ZImageRegionalPromptingExtension
from invokeai.backend.z_image.text_conditioning import ZImageRegionalTextConditioning
from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter

MB = 1024**2

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


def _dense_transformer() -> torch.nn.Module:
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 8)
    return model


def _int8_transformer(out_features: int, device: str = "cpu") -> torch.nn.Module:
    """An int8_tensorwise build: its layer materializes two weight-sized tensors in the compute dtype per forward."""
    model = torch.nn.Module()
    model.proj = Int8ConvrotLinear(
        torch.zeros(out_features, CONVROT_GROUP_SIZE, dtype=torch.int8, device=device),
        torch.ones(out_features, 1, device=device),
        convrot=False,
    )
    return model


def _estimate(
    image_seq_len,
    text_seq_len=512,
    num_control_blocks=0,
    has_mask=False,
    materializes=lambda mask: False,
    transformer=None,
    regional_mask=None,
):
    with _sdpa(materializes):
        return estimate_z_image_denoise_working_memory(
            transformer if transformer is not None else _dense_transformer(),
            image_seq_len=image_seq_len,
            text_seq_len=text_seq_len,
            lora_count=0,
            control_blocks=num_control_blocks,
            regional_attn_mask=regional_mask,
            has_attention_mask=has_mask,
            device=DEVICE,
            dtype=torch.bfloat16,
        ).reservation_bytes


class TestEstimate:
    def test_image_and_caption_are_counted_as_the_transformer_pads_them(self):
        """Each run is padded to a multiple of 32 before the two are joined, and the padding is attended too."""
        assert _estimate(4096, text_seq_len=97) == _estimate(4096, text_seq_len=128)
        assert _estimate(4001, text_seq_len=128) == _estimate(4032, text_seq_len=128)
        assert _estimate(4096, text_seq_len=129) > _estimate(4096, text_seq_len=128)

    def test_a_quantized_transformer_reserves_its_dequant_transient(self):
        """An int8 layer materializes two weight-sized bf16 tensors per forward, alive alongside the activations."""
        quantized = _estimate(4096, transformer=_int8_transformer(1024))

        assert quantized - _estimate(4096) == 2 * 1024 * CONVROT_GROUP_SIZE * torch.bfloat16.itemsize

    def test_an_fp8_stored_transformer_reserves_its_largest_layer_widened(self):
        """FP8 Storage widens a layer's weight and bias to bf16 for its forward, one layer at a time, on top of the
        line the dense forwards were measured on."""
        fp8_stored = _dense_transformer()
        ModelLoader._apply_fp8_to_nn_module(fp8_stored, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

        assert _estimate(4096, transformer=fp8_stored) - _estimate(4096) == (8 * CONVROT_GROUP_SIZE + 8) * 2

    def test_a_dtype_without_a_line_is_refused(self):
        """Pricing it on another dtype's line would be a guess; the estimator says what is missing instead."""
        with pytest.raises(ValueError, match=r"'denoise\.z_image' has no torch\.float64 line on Platform\("):
            estimate_z_image_denoise_working_memory(
                _dense_transformer(),
                image_seq_len=4096,
                text_seq_len=512,
                lora_count=0,
                device=DEVICE,
                dtype=torch.float64,
            )


class TestMaterializedScoreMatrix:
    def test_the_score_matrix_is_charged_only_where_it_is_built(self):
        """Measured at 1024px: a forced math kernel took 9.4 GiB where the fused one took 1.0. The attention mask
        decides the dispatch, so it has to reach the probe."""
        seq_len = 4096 + 512
        unmasked = _estimate(4096, materializes=lambda has_mask: has_mask)
        masked = _estimate(4096, has_mask=True, materializes=lambda has_mask: has_mask)

        assert unmasked == _estimate(4096)
        assert masked - unmasked == 30 * seq_len * seq_len * SDPA_MATH_BYTES_PER_SCORE_ELEMENT


RTX_4090_NATIVE = Platform("cudnn", "native", "windows")
RX_9060_XT_EXPANDABLE = Platform("miopen", "expandable", "windows")

# ControlNet forwards at 1024px with a 512-token caption, measured demand in bytes. 15 blocks: `--mode control`
# (2026-10-18) beside the plain forward of the same run, 19.50 hint-sized tensors more on the RTX 4090 (19.26 in fp32)
# and 18.37 on the RX 9060 XT, against the n + 4 = 19 charged. 1 and 6 blocks: the RTX 4090's peak reserved bytes of the
# script's first runs, 110 and 200 MiB over its plain forward then.
CONTROL_FORWARDS = [
    (RTX_4090_NATIVE, "bf16", 15, 1767899136),
    (RTX_4090_NATIVE, "fp16", 15, 1767899136),
    (RTX_4090_NATIVE, "fp32", 15, 3244294144),
    (RX_9060_XT_EXPANDABLE, "bf16", 15, 1633091584),
    (RX_9060_XT_EXPANDABLE, "fp16", 15, 1668808704),
    (RTX_4090_NATIVE, "bf16", 1, 1122 * MB),
    (RTX_4090_NATIVE, "bf16", 6, 1212 * MB),
]
# The RTX 4090's forwards with SDPA forced onto the math kernel at 1024px (`--mode math`, 2026-10-18): the whole score
# matrix materializes, which CUDA does not chunk.
MATH_KERNEL_FORWARDS = [("bf16", 10116661248), ("fp16", 10116661248), ("fp32", 10081009664)]


def _forward_features(dtype: str, control_blocks: int = 0, score_matrix_bytes: int = 0) -> dict:
    """A 1024px forward with a 512-token caption as the estimator records it; a ControlNet forward passes a mask."""
    return {"image_tokens": 4096, "text_tokens": 512, "tokens": 4608, "dtype": dtype, "control_blocks": control_blocks,
            "has_attention_mask": control_blocks > 0, "regional_bias_bytes": 0, "score_matrix_bytes": score_matrix_bytes,
            "lora_count": 0, "dequant_transient_bytes": 0, "fp8_cast_transient_bytes": 0}  # fmt: skip


def _shipped_reservation(features: dict, platform: Platform) -> int:
    shipped = shipped_coefficients(Z_IMAGE_DENOISE_SPEC, platform)
    return Z_IMAGE_DENOISE_SPEC.evaluate(features, shipped).reservation_bytes


class TestMeasuredChecks:
    """Forwards the shipped points leave out, because a term of their own sits on top of the line."""

    @pytest.mark.parametrize("platform, dtype, control_blocks, measured", CONTROL_FORWARDS)
    def test_a_control_forward_is_covered_by_the_line_and_its_hints(self, platform, dtype, control_blocks, measured):
        """Within the 1.3x a shipped point holds its estimate to: every byte reserved beyond need is transformer
        offloaded to RAM and run over PCIe."""
        estimate = _shipped_reservation(_forward_features(dtype, control_blocks), platform)

        assert measured <= estimate <= 1.3 * measured

    @pytest.mark.parametrize("dtype, measured", MATH_KERNEL_FORWARDS)
    def test_a_math_kernel_forward_is_covered_by_the_line_and_its_score_matrix(self, dtype, measured):
        """A regime no server runs by default, so it bounds the estimate from below only, as the VAE's forced-math
        points do."""
        scores = 30 * 4608 * 4608 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT

        assert _shipped_reservation(_forward_features(dtype, score_matrix_bytes=scores), RTX_4090_NATIVE) >= measured


class TestRegionalAttentionBias:
    def test_bias_is_sized_on_the_padded_unified_sequence(self):
        """4096 image tokens and a 100-token caption: the boolean copy covers the 4196 real tokens, the additive bias
        the 4096 + 128 padded ones, and `torch.where` fills the image block from a 4096 x 4096 scratch."""
        mask = torch.empty((4196, 4196), dtype=torch.float16, device="meta")

        bias_bytes = _estimate(4096, text_seq_len=100, regional_mask=mask) - _estimate(4096, text_seq_len=100)

        assert bias_bytes == 4196**2 + 4224**2 * 2 + 4096**2 * 2


# The reservation the node priced by hand before its estimate declared an `EstimatorSpec`, frozen verbatim with the
# node's constants: `ZImageDenoiseInvocation._estimate_working_memory` and `_regional_attention_bias_bytes`, plus the
# dequant transient the node added. The shipped `legacy` values reproduce it where no measured points replace them.
_LEGACY_SEQ_MULTI_OF = 32
_LEGACY_HIDDEN_SIZE = 3840
_LEGACY_ATTENTION_HEADS = 30
_LEGACY_BYTES_PER_TOKEN = int(0.25 * 1024**2)


def _legacy_padded(tokens: int) -> int:
    return tokens + (-tokens) % _LEGACY_SEQ_MULTI_OF


def _legacy_estimate_working_memory(
    image_seq_len: int,
    text_seq_len: int,
    num_loras: int,
    num_control_blocks: int = 0,
    regional_attention_bias_bytes: int = 0,
    has_attention_mask: bool = False,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    GB = 1024**3
    seq_len = _legacy_padded(image_seq_len) + _legacy_padded(text_seq_len)
    estimated = seq_len * _LEGACY_BYTES_PER_TOKEN + int(1.0 * GB)
    estimated += regional_attention_bias_bytes
    estimated += sdpa_score_matrix_bytes(
        device=device,
        dtype=dtype,
        num_heads=_LEGACY_ATTENTION_HEADS,
        head_dim=_LEGACY_HIDDEN_SIZE // _LEGACY_ATTENTION_HEADS,
        seq_len=seq_len,
        has_attn_mask=has_attention_mask,
        via_diffusers_dispatch=True,
    )
    if num_control_blocks:
        estimated += (num_control_blocks + 4) * seq_len * _LEGACY_HIDDEN_SIZE * dtype.itemsize
    estimated += int(0.5 * num_loras * GB)
    return estimated


def _legacy_regional_attention_bias_bytes(
    regional_attn_mask: torch.Tensor | None, image_seq_len: int, dtype: torch.dtype
) -> int:
    if regional_attn_mask is None:
        return 0
    mask_side = regional_attn_mask.shape[0]
    unified_seq_len = _legacy_padded(image_seq_len) + _legacy_padded(mask_side - image_seq_len)
    return mask_side * mask_side + (unified_seq_len * unified_seq_len + image_seq_len * image_seq_len) * dtype.itemsize


@dataclass(frozen=True)
class _Attention:
    """How a build answers the dispatch questions `sdpa_score_matrix_bytes` asks."""

    device: torch.device
    dispatch: str
    """The diffusers attention backend's routing."""
    chunks: bool
    """Whether the ROCm guard computes math attention in chunks."""
    materializes: Callable[[bool], bool]
    """Whether torch's SDPA builds the score matrix, by whether an attention mask is passed."""

    @contextmanager
    def answers(self):
        with (
            patch.object(attention, "_diffusers_attention_dispatch", new=lambda: self.dispatch),
            patch.object(attention, "rocm_sdpa_uses_math_kernel", new=lambda device_type, head_dim: False),
            patch.object(attention, "rocm_sdpa_chunks_math", new=lambda device_type: self.chunks),
            patch.object(attention, "_torch_sdpa_materializes_score_matrix", new=lambda *a: self.materializes(a[-1])),
        ):
            yield


ATTENTION = {
    # A fused kernel unless a mask is passed.
    "cuda": _Attention(torch.device("cuda"), attention._DISPATCH_TORCH, False, lambda has_mask: has_mask),
    # A fused diffusers backend (flash, sage) never asks torch, which here would materialize.
    "cuda-fused-backend": _Attention(torch.device("cuda"), attention._DISPATCH_FUSED, False, lambda has_mask: True),
    # Windows ROCm without fused kernels, under the guard that chunks the math kernel.
    "hip": _Attention(torch.device("cuda"), attention._DISPATCH_TORCH, True, lambda has_mask: True),
    "mps": _Attention(torch.device("mps"), attention._DISPATCH_TORCH, False, lambda has_mask: True),
}

# Where the shipped entry is still `legacy`: ROCm with the native allocator on every OS but Windows.
LEGACY_PLATFORMS = [Platform("miopen", "native", "linux"), Platform("miopen", "native", "other")]
# Height, width: odd token grids included, whose runs the transformer pads.
SIZES = [(512, 512), (528, 528), (784, 1040), (1024, 1024), (1040, 1328), (1536, 1536), (2048, 1104), (2048, 2048)]
CAPTIONS = [0, 77, 512, 1000]
CONTROL_BLOCKS = [0, 1, 6, 15]
# A transient of 123 MiB in bf16 and fp16, 246 MiB in fp32; on the meta device, so it costs no memory.
TRANSFORMERS = [_dense_transformer(), _int8_transformer(123 * MB // (2 * CONVROT_GROUP_SIZE * 2), device="meta")]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


class TestLegacyReservation:
    @pytest.mark.parametrize("attention_build", list(ATTENTION.values()), ids=list(ATTENTION))
    def test_the_reservation_is_the_legacy_formula_byte_for_byte(self, attention_build: _Attention):
        """On the platforms still `legacy`: estimated on such a machine, and evaluated again there from the features
        alone. None of the transformers holds a float8 weight, the one case the reservation adds a term the formula
        lacked."""
        mismatches = []
        with attention_build.answers(), patch.object(spec_module, "platform_for", return_value=LEGACY_PLATFORMS[0]):
            grid = itertools.product(
                SIZES, CAPTIONS, CONTROL_BLOCKS, (False, True), (False, True), (0, 2), TRANSFORMERS, DTYPES
            )
            for (height, width), caption, blocks, has_mask, regional, loras, transformer, dtype in grid:
                image_seq_len = (height // 16) * (width // 16)
                mask_side = image_seq_len + caption
                regional_mask = torch.empty((mask_side, mask_side), device="meta") if regional else None
                expected = _legacy_estimate_working_memory(
                    image_seq_len=image_seq_len,
                    text_seq_len=caption,
                    num_loras=loras,
                    num_control_blocks=blocks,
                    regional_attention_bias_bytes=_legacy_regional_attention_bias_bytes(
                        regional_mask, image_seq_len, dtype
                    ),
                    has_attention_mask=has_mask,
                    device=attention_build.device,
                    dtype=dtype,
                ) + peak_dequant_transient_bytes(transformer, dtype)

                estimate = estimate_z_image_denoise_working_memory(
                    transformer,
                    image_seq_len=image_seq_len,
                    text_seq_len=caption,
                    lora_count=loras,
                    control_blocks=blocks,
                    regional_attn_mask=regional_mask,
                    has_attention_mask=has_mask,
                    device=attention_build.device,
                    dtype=dtype,
                )
                reserved = [estimate.reservation_bytes] + [
                    Z_IMAGE_DENOISE_SPEC.evaluate(
                        estimate.features, shipped_coefficients(Z_IMAGE_DENOISE_SPEC, platform)
                    ).reservation_bytes
                    for platform in LEGACY_PLATFORMS
                ]
                if any(value != expected for value in reserved):
                    case = (height, width, caption, blocks, has_mask, regional, loras, dtype)
                    mismatches.append((case, expected, reserved))
        assert not mismatches, f"{len(mismatches)} cases differ, e.g. (case, legacy, reserved): {mismatches[:3]}"


EVERY_PLATFORM = [
    Platform(*keys) for keys in itertools.product(get_args(ConvBackend), get_args(Allocator), get_args(OperatingSystem))
]


def _platform_id(platform: Platform) -> str:
    return f"{platform.conv_backend}-{platform.allocator}-{platform.os}"


class TestUnmeasuredPlatforms:
    """`priced_as` prices a platform nothing is measured on -- XPU, MPS, the CPU, cuDNN with expandable segments,
    MIOpen under `hipMallocAsync` -- with a measured platform's lines, which describe the card they were measured on.
    The entries it borrows floor every value at the legacy constants there (`unmeasured_floor`)."""

    @pytest.mark.parametrize("platform", [p for p in EVERY_PLATFORM if not is_measured(p)], ids=_platform_id)
    def test_an_unmeasured_platform_reserves_the_legacy_formula_at_least(self, platform: Platform):
        """The RTX 4090's native-allocator line, which XPU borrows, prices a 1024px bf16 forward at 1172 MiB; the
        formula reserves 2176. The analytic terms are the formula's own (`TestLegacyReservation`), so plain forwards
        compare the lines."""
        short = []
        with ATTENTION["cuda"].answers(), patch.object(spec_module, "platform_for", return_value=platform):
            for (height, width), caption, dtype in itertools.product(SIZES, CAPTIONS, DTYPES):
                image_seq_len = (height // 16) * (width // 16)
                legacy = _legacy_estimate_working_memory(image_seq_len, caption, 0, device=DEVICE, dtype=dtype)
                reserved = estimate_z_image_denoise_working_memory(
                    TRANSFORMERS[0],
                    image_seq_len=image_seq_len,
                    text_seq_len=caption,
                    lora_count=0,
                    device=DEVICE,
                    dtype=dtype,
                ).reservation_bytes
                if reserved < legacy:
                    short.append(((height, width, caption, dtype), legacy, reserved))
        assert not short, f"{len(short)} cases below the formula, e.g. (case, legacy, reserved): {short[:3]}"

    @pytest.mark.parametrize("platform", [p for p in EVERY_PLATFORM if is_measured(p)], ids=_platform_id)
    def test_a_measured_platform_prices_with_its_own_fit_unfloored(self, platform: Platform):
        """A platform the points were measured on is what they describe, below the formula where they measured less."""
        unfloored = [
            file.model_copy(
                update={"coefficients": [e.model_copy(update={"unmeasured_floor": {}}) for e in file.coefficients]}
            )
            for file in shipped_data().values()
        ]

        assert shipped_coefficients(Z_IMAGE_DENOISE_SPEC, platform) == resolve_coefficients(
            unfloored, Z_IMAGE_DENOISE_SPEC, platform, origin="shipped"
        )


class _StopBeforeLoad(Exception):
    """Raised out of a mocked `model_on_device()` so the node stops at a reservation."""


# Stands in for the estimate, which is tested above: a reservation none of its terms can produce.
ESTIMATE = WorkingMemoryEstimate(
    estimator=Z_IMAGE_DENOISE_SPEC.id,
    version=Z_IMAGE_DENOISE_SPEC.version,
    features={},
    terms=(WorkingMemoryTerm("stand_in", 3_000_000_007, True),),
    source="test",
)
POSITIVE_TOKENS = 8


class TestNodeReservation:
    """The wire from the node to `model_on_device()`: deleting any of these arguments leaves the estimate's own tests
    green."""

    def _run(
        self,
        *,
        control_blocks=None,
        control_tiling=None,
        force_tiled_decode=False,
        negative_tokens=None,
        loras=0,
        regional_mask=None,
    ):
        from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
            ConditioningFieldData,
            ZImageConditioningInfo,
        )

        transformer = _dense_transformer()
        transformer_info = MagicMock()
        transformer_info.model = transformer

        control = None
        control_info = MagicMock()
        if control_blocks is None:
            transformer_info.model_on_device = MagicMock(side_effect=_StopBeforeLoad)
        else:
            # Past the transformer's lock, up to the adapter's.
            transformer_info.model_on_device = MagicMock(return_value=nullcontext((None, transformer)))
            control = ZImageControlField.model_construct(
                control_model=MagicMock(), image_name="control", **(control_tiling or {})
            )
            control_info.model = MagicMock(spec=ZImageControlAdapter)
            control_info.model.control_layers = [torch.nn.Identity()] * control_blocks
            if control_tiling is None:
                control_info.model_on_device = MagicMock(side_effect=_StopBeforeLoad)
            else:
                # Past the adapter's lock too, up to the control image's encode.
                control_info.model.config = {"control_in_dim": 33}
                control_info.model_on_device = MagicMock(return_value=nullcontext((None, control_info.model)))

        caption_tokens = {"pos": POSITIVE_TOKENS, "neg": negative_tokens}
        context = MagicMock()
        context.models.load.side_effect = lambda identifier: (
            control_info if control is not None and identifier is control.control_model else transformer_info
        )
        context.images.get_pil.return_value = Image.new("RGB", (256, 256))
        context.config.get.return_value.force_tiled_decode = force_tiled_decode
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
            vae=MagicMock() if control is not None else None,
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
                patch(
                    "invokeai.app.invocations.z_image.z_image_denoise.estimate_z_image_denoise_working_memory",
                    return_value=ESTIMATE,
                )
            )
            control_encode = stack.enter_context(
                patch.object(ZImageImageToLatentsInvocation, "vae_encode", side_effect=_StopBeforeLoad)
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
            transformer=transformer,
            reserved=transformer_info.model_on_device.call_args.kwargs["working_memory"],
            estimated_for=estimate.call_args.args,
            estimate=estimate.call_args.kwargs,
            adapter_reserved=(
                control_info.model_on_device.call_args.kwargs.get("working_mem_bytes") if control is not None else None
            ),
            control_encode=control_encode.call_args.kwargs if control_encode.called else None,
        )

    def test_the_transformer_reserves_the_estimate_for_its_sequence(self):
        run = self._run()

        assert run.reserved is ESTIMATE
        # Priced on the model about to be locked, whose quantized layers dequantize per forward.
        assert run.estimated_for == (run.transformer,)
        # 256px is a 16x16 token grid.
        assert (run.estimate["image_seq_len"], run.estimate["text_seq_len"]) == (256, POSITIVE_TOKENS)
        assert (run.estimate["lora_count"], run.estimate["control_blocks"]) == (0, 0)
        assert (run.estimate["regional_attn_mask"], run.estimate["has_attention_mask"]) == (None, False)
        assert (run.estimate["device"], run.estimate["dtype"]) == (torch.device("cpu"), torch.bfloat16)

    def test_a_longer_negative_caption_sizes_the_sequence(self):
        """With CFG the unconditional pass runs its own forward, so the longer of the two captions is the peak."""
        assert self._run(negative_tokens=40).estimate["text_seq_len"] == 40

    def test_loras_are_counted(self):
        assert self._run(loras=2).estimate["lora_count"] == 2

    def test_a_regional_mask_reaches_the_estimate(self):
        mask = torch.zeros((256 + POSITIVE_TOKENS, 256 + POSITIVE_TOKENS), dtype=torch.float16)

        run = self._run(regional_mask=mask)

        assert run.estimate["regional_attn_mask"] is mask
        assert run.estimate["has_attention_mask"] is True

    def test_a_control_adapter_reaches_the_estimate_and_keeps_the_reservation(self):
        """The adapter's block count sizes the hints and its forward always passes a padding mask. It is locked after
        the transformer, which can then no longer be offloaded -- so it must ask for the same working memory, or its
        weights fill the room kept for the forward."""
        run = self._run(control_blocks=15)

        assert (run.estimate["control_blocks"], run.estimate["has_attention_mask"]) == (15, True)
        assert run.reserved is ESTIMATE
        assert run.adapter_reserved == ESTIMATE.reservation_bytes

    def test_the_control_fields_tiling_reaches_the_control_encode(self):
        """The control image is encoded at generation size. For an upscale that is the enlarged frame, whose untiled
        encode does not fit on a consumer card, so the tiling the control field asks for has to reach the encode."""
        run = self._run(control_blocks=15, control_tiling={"tiled": True, "tile_size": 768})

        assert (run.control_encode["tiled"], run.control_encode["tile_size"]) == (True, 768)

    def test_force_tiled_decode_tiles_a_control_encode_the_field_leaves_whole(self):
        """Low-VRAM setups force every VAE pass to tile, and the control encode is one of them."""
        run = self._run(control_blocks=15, control_tiling={"tiled": False, "tile_size": 0}, force_tiled_decode=True)

        assert run.control_encode["tiled"] is True


def test_the_control_node_hands_its_tiling_to_the_field():
    """The denoise reads the tiling off the field, so a node that stopped copying it would encode whole."""
    node = ZImageControlInvocation(
        id="control",
        image=ImageField(image_name="frame"),
        control_model=ModelIdentifierField(
            key="tile", hash="tile", name="tile", base=BaseModelType.ZImage, type=ModelType.ControlNet
        ),
        tiled=True,
        tile_size=768,
    )

    control = node.invoke(MagicMock()).control

    assert (control.tiled, control.tile_size) == (True, 768)
