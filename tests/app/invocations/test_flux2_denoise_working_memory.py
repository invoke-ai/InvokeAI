"""FLUX.2 denoise working memory: the estimate as `denoise.flux2` declares it, against the formula it replaced.

The estimate prices with the lines `backend/working_memory/data/denoise.json` ships, the fit of the plain forwards
measured there, which `tests/backend/working_memory/` holds it to. `TestMeasuredChecks` holds it to what those points
leave out: a forced math kernel's score matrix, measured by `scripts/calibrate_flux2_denoise_working_memory.py` beside
the plain forward of the same run, and the peaks measured before the lines had points of their own where no shipped
point covers the same case.

ROCm with the native allocator off Windows is not measured and still prices with the hand-set `legacy` constants the
node reserved by hand, which `TestLegacyReservation` holds byte for byte to that formula, frozen below, plus the
terms it lacked: the largest layer partial loading streams, a masked forward's padded bias copy and an fp8-stored
transformer's cast transient (`TestEstimate`). Every other platform nothing is measured on prices as a measured one,
whose entry floors every value at those constants, and `TestUnmeasuredPlatforms` holds it to the formula at least.

What the node hands the estimate and `model_on_device()` is pinned in
`tests/app/invocations/test_flux2_working_memory.py`.
"""

import itertools
from typing import get_args
from unittest.mock import patch

import pytest
import torch
from diffusers import Flux2Transformer2DModel
from gguf import GGMLQuantizationType

from invokeai.backend.flux2.denoise_working_memory import (
    FLUX2_DENOISE_SPEC,
    FLUX2_HIDDEN_SIZES,
    estimate_flux2_denoise_working_memory,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT, sdpa_score_matrix_bytes
from invokeai.backend.working_memory import spec as spec_module
from invokeai.backend.working_memory.coefficients import resolve_coefficients, shipped_coefficients, shipped_data
from invokeai.backend.working_memory.platform import Allocator, ConvBackend, OperatingSystem, Platform, is_measured
from tests.app.invocations.test_z_image_denoise_working_memory import ATTENTION

MB = 1024**2
GB = 1024**3
FLUX2_CONTEXT_DIMS = {3072: 7680, 4096: 12288, 6144: 15360}
"""The caption width the loader reads off each width's checkpoint (`configs/flux2_variant.py`)."""


def _dense_transformer() -> torch.nn.Module:
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 8)
    return model


def _bf16_transformer(hidden_size: int) -> torch.nn.Module:
    """The transformer class the loader builds at a width, one block per stream, in bf16 on the meta device."""
    with torch.device("meta"):
        return Flux2Transformer2DModel(
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=128,
            num_attention_heads=hidden_size // 128,
            joint_attention_dim=FLUX2_CONTEXT_DIMS[hidden_size],
            patch_size=1,
        ).to(torch.bfloat16)


def _int8_transformer(out_features: int) -> torch.nn.Module:
    """An int8_tensorwise build on the meta device: its layer materializes two weight-sized bf16 tensors per forward."""
    model = torch.nn.Module()
    model.proj = Int8ConvrotLinear(
        torch.zeros(out_features, CONVROT_GROUP_SIZE, dtype=torch.int8, device="meta"),
        torch.ones(out_features, 1, device="meta"),
        convrot=False,
    )
    return model


def _estimate(
    transformer=None,
    *,
    hidden_size=4096,
    image=4096,
    refs=0,
    text=512,
    batch=1,
    loras=0,
    control_blocks=0,
    mask=None,
    resident=False,
):
    return estimate_flux2_denoise_working_memory(
        transformer if transformer is not None else _dense_transformer(),
        hidden_size=hidden_size,
        image_seq_len=image,
        ref_image_seq_len=refs,
        text_seq_len=text,
        batch_size=batch,
        lora_count=loras,
        control_blocks=control_blocks,
        regional_attn_mask=mask,
        weights_resident=resident,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )


class TestEstimate:
    def test_an_fp8_stored_transformer_reserves_its_largest_layer_widened(self):
        """FP8 Storage widens a layer's weight and bias to bf16 for its forward, on top of what the formula priced."""
        fp8_stored = _dense_transformer()
        ModelLoader._apply_fp8_to_nn_module(fp8_stored, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

        with ATTENTION["cuda"].answers():
            added = _estimate(fp8_stored, resident=True).reservation_bytes - _estimate(resident=True).reservation_bytes

        assert added == (8 * CONVROT_GROUP_SIZE + 8) * torch.bfloat16.itemsize

    @pytest.mark.parametrize(
        "hidden_size, image, refs, batch", [(6144, 4096, 0, 1), (6144, 4096, 4096, 2), (3072, 1024, 0, 3)]
    )
    def test_a_control_branch_reserves_four_hints_beyond_its_blocks_over_the_image_stream(
        self, hidden_size, image, refs, batch
    ):
        """Each hint is a (batch, image and reference tokens, width) bf16 tensor: `add_hint` touches only the image
        stream, not the branch's text stream, so a longer caption adds nothing to the term."""
        hint = batch * (image + refs) * hidden_size * 2
        shape = {"hidden_size": hidden_size, "image": image, "refs": refs, "batch": batch}

        with ATTENTION["cuda"].answers():
            plain = _estimate(**shape).reservation_bytes
            for blocks in (1, 15):
                for text in (77, 512):
                    controlled = _estimate(**shape, text=text, control_blocks=blocks).reservation_bytes
                    assert controlled - _estimate(**shape, text=text).reservation_bytes == (blocks + 4) * hint
            assert _estimate(**shape, control_blocks=0).reservation_bytes == plain

    @pytest.mark.parametrize("hidden_size", FLUX2_HIDDEN_SIZES)
    def test_a_transformer_that_may_load_partially_reserves_its_largest_layer(self, hidden_size):
        """Partial loading copies a RAM-held layer to the device for its forward, alive beside the activations. On the
        transformer the loader builds at that width, in bf16, that is the single-stream blocks' fused projection of
        9 widths by 1; at [dev]'s it is 679,477,248 bytes."""
        model = _bf16_transformer(hidden_size)

        with ATTENTION["cuda"].answers():
            added = (
                _estimate(model, hidden_size=hidden_size).reservation_bytes
                - _estimate(model, hidden_size=hidden_size, resident=True).reservation_bytes
            )

        assert added == 9 * hidden_size * hidden_size * 2
        if hidden_size == 6144:
            assert added == 679_477_248

    def test_an_fp8_stored_layer_streams_its_float8_bytes(self):
        """What partial loading copies is the stored weight; its bf16 widening is the fp8 cast term's."""
        fp8_stored = _dense_transformer()
        ModelLoader._apply_fp8_to_nn_module(fp8_stored, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

        with ATTENTION["cuda"].answers():
            features = _estimate(fp8_stored).features

        assert features["streamed_layer_bytes"] == (8 * CONVROT_GROUP_SIZE + 8) * 1

    def test_a_gguf_layer_streams_its_quantized_bytes(self):
        """A Q8_0 weight stores 34 bytes per block of 32 elements."""
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(32, 64, bias=False)
        quantized = GGMLTensor(
            torch.zeros(64, 34, dtype=torch.uint8), GGMLQuantizationType.Q8_0, torch.Size([64, 32]), torch.bfloat16
        )
        model.proj.weight = torch.nn.Parameter(quantized, requires_grad=False)

        with ATTENTION["cuda"].answers():
            features = _estimate(model).features

        assert features["streamed_layer_bytes"] == 64 * 34

    @pytest.mark.parametrize("side, batch", [(4608, 1), (9124, 1), (9124, 2)])
    def test_a_regional_mask_reserves_the_bias_and_a_padded_copy_per_sample(self, side, batch):
        """The memory-efficient kernel a masked forward runs on pads an unaligned bias into a new tensor; no point
        measures a masked forward, so the copy is reserved aligned or not. 9124 pads to 9128. A fused diffusers backend
        materializes no score matrix either way, so the bias and its copy are all the mask adds."""
        mask = torch.empty((side, side), device="meta")
        padded = -(-side // 8) * 8

        with ATTENTION["cuda-fused-backend"].answers():
            masked = _estimate(mask=mask, batch=batch, resident=True).reservation_bytes
            added = masked - _estimate(batch=batch, resident=True).reservation_bytes

        assert added == side * side * 2 + batch * side * padded * 2

    @pytest.mark.parametrize("hidden_size, dtype", [(4096, torch.float16), (5120, torch.bfloat16)])
    def test_a_dtype_or_width_without_a_line_is_refused(self, hidden_size, dtype):
        """Pricing it on another width's or dtype's line would be a guess; the estimator says what is missing."""
        with pytest.raises(ValueError, match=r"'denoise\.flux2' has no torch\.\w+ line at width \d+ on Platform\("):
            estimate_flux2_denoise_working_memory(
                _dense_transformer(),
                hidden_size=hidden_size,
                image_seq_len=4096,
                ref_image_seq_len=0,
                text_seq_len=512,
                batch_size=1,
                lora_count=0,
                device=torch.device("cpu"),
                dtype=dtype,
            )


class TestLine:
    """Which forwards a point of may fit the `denoise_bf16_h<width>` line: the plain ones, whose demand is the line's
    alone."""

    def _line(self, *args, resident=True, **kwargs):
        with ATTENTION["cuda"].answers():
            features = _estimate(*args, resident=resident, **kwargs).features
        return FLUX2_DENOISE_SPEC.line(features)

    def test_a_plain_forward_fits_its_widths_line_over_every_attended_token(self):
        line = self._line(hidden_size=3072, image=4096, refs=12288, text=77)

        assert (line.suffix, line.size, line.fits) == ("denoise_bf16_h3072", 4096 + 12288 + 77, True)
        assert line.label == "denoise-bf16-h3072-4096+12288ref+77tok"

    def test_a_batch_fits_the_line_at_its_sequences_together(self):
        """The line prices what a batch adds, its sequences; the base is a forward's, paid once."""
        line = self._line(image=4096, batch=2)

        assert (line.size, line.fits, line.label) == (2 * (4096 + 512), True, "denoise-bf16-h4096-4096+512tok-b2")

    def test_a_forward_with_an_analytic_term_or_a_mask_fits_nothing_and_says_why(self):
        mask = torch.empty((4608, 4608), device="meta")

        regional = self._line(mask=mask)
        lora = self._line(loras=1)
        quantized = self._line(_int8_transformer(64))
        controlled = self._line(hidden_size=6144, control_blocks=4)
        streamed = self._line(_bf16_transformer(6144), hidden_size=6144, resident=False)

        assert not (regional.fits or lora.fits or quantized.fits or controlled.fits or streamed.fits)
        # This build materializes the score matrix for a masked forward, which the label names as well.
        bias = 4608 * 4608 * 2
        assert f"-attention_mask-regional_attention_bias={bias}-padded_attention_bias={bias}-attention_scores=" in (
            regional.label
        )
        assert streamed.label.endswith(f"-streamed_layer={6144 * 55296 * 2}")
        assert lora.label.endswith(f"-lora_patches={2**29}")
        assert quantized.label.endswith(f"-dequant_transient={2 * 64 * CONVROT_GROUP_SIZE * 2}")
        assert controlled.label.endswith(f"-control_hints={8 * 4096 * 6144 * 2}")


# The reservation the node priced by hand before its estimate declared an `EstimatorSpec`, frozen verbatim with the
# node's constants: `Flux2DenoiseInvocation._estimate_working_memory`, the regional bias the node computed for it, and
# the dequant transient the node added. The shipped `legacy` values reproduce it where no measured points replace them.
_LEGACY_ATTENTION_HEAD_DIM = 128
_LEGACY_REFERENCE_HIDDEN_SIZE = 4096
_LEGACY_BYTES_PER_TOKEN_AT_REFERENCE_WIDTH = int(0.42 * 1024**2)


def _legacy_estimate_working_memory(
    image_seq_len: int,
    ref_image_seq_len: int,
    text_seq_len: int,
    num_loras: int,
    batch_size: int = 1,
    hidden_size: int = _LEGACY_REFERENCE_HIDDEN_SIZE,
    regional_attention_bias_bytes: int = 0,
    has_regional_attention_mask: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    GB = 1024**3
    per_token_bytes = int(_LEGACY_BYTES_PER_TOKEN_AT_REFERENCE_WIDTH * hidden_size / _LEGACY_REFERENCE_HIDDEN_SIZE)
    total_seq_len = image_seq_len + ref_image_seq_len + text_seq_len
    estimated = total_seq_len * batch_size * per_token_bytes
    estimated += int(1.0 * GB)
    estimated += regional_attention_bias_bytes
    estimated += sdpa_score_matrix_bytes(
        device=device,
        dtype=dtype,
        num_heads=(hidden_size // _LEGACY_ATTENTION_HEAD_DIM) * batch_size,
        head_dim=_LEGACY_ATTENTION_HEAD_DIM,
        seq_len=total_seq_len,
        has_attn_mask=has_regional_attention_mask,
        via_diffusers_dispatch=True,
    )
    if num_loras > 0:
        estimated += int(0.5 * num_loras * batch_size * GB)
    return estimated


def _legacy_regional_attention_bias_bytes(regional_attn_mask: torch.Tensor | None, dtype: torch.dtype) -> int:
    if regional_attn_mask is None:
        return 0
    return regional_attn_mask.numel() * torch.empty((), dtype=dtype).element_size()


# Where the shipped entry is still `legacy`: ROCm with the native allocator on every OS but Windows.
LEGACY_PLATFORMS = [Platform("miopen", "native", "linux"), Platform("miopen", "native", "other")]
# Height, width: an elongated frame included.
SIZES = [(512, 512), (784, 1040), (1024, 1024), (1440, 1440), (2048, 2048)]
# None, one 1024px reference, and three, which `Flux2RefImageExtension` caps at 1024px each.
REFERENCE_TOKENS = [0, 4096, 12288]
CAPTIONS = [77, 512, 1000]
BATCHES = [1, 2, 3]
# A transient of 123 MiB in bf16; on the meta device, so it costs no memory.
TRANSFORMERS = [_dense_transformer(), _int8_transformer(123 * MB // (2 * CONVROT_GROUP_SIZE * 2))]


class TestLegacyReservation:
    @pytest.mark.parametrize("attention_build", list(ATTENTION.values()), ids=list(ATTENTION))
    def test_the_reservation_is_the_legacy_formula_byte_for_byte(self, attention_build):
        """On the platforms still `legacy`: estimated on such a machine, and evaluated again there from the features
        alone, plus the terms the formula lacked and `TestEstimate` pins: the streamed largest layer and, with a mask,
        its padded bias copy. None of the transformers holds a float8 weight, whose cast is another such term. The node
        prices an unknown variant at the widest width, 6144, which this grid covers."""
        mismatches = []
        with attention_build.answers(), patch.object(spec_module, "platform_for", return_value=LEGACY_PLATFORMS[0]):
            grid = itertools.product(
                FLUX2_HIDDEN_SIZES, SIZES, REFERENCE_TOKENS, CAPTIONS, BATCHES, (False, True), (0, 2), TRANSFORMERS
            )
            for hidden, (height, width), refs, caption, batch, regional, loras, transformer in grid:
                image_seq_len = (height // 16) * (width // 16)
                mask_side = image_seq_len + caption
                regional_mask = torch.empty((mask_side, mask_side), device="meta") if regional else None
                expected = _legacy_estimate_working_memory(
                    image_seq_len=image_seq_len,
                    ref_image_seq_len=refs,
                    text_seq_len=caption,
                    num_loras=loras,
                    batch_size=batch,
                    hidden_size=hidden,
                    regional_attention_bias_bytes=_legacy_regional_attention_bias_bytes(regional_mask, torch.bfloat16),
                    has_regional_attention_mask=regional_mask is not None,
                    device=attention_build.device,
                ) + peak_dequant_transient_bytes(transformer, torch.bfloat16)
                # The terms the formula lacked: the layer partial loading streams, and the mask's padded bias copy.
                expected += max(
                    sum(t.nbytes for t in (*m.parameters(recurse=False), *m.buffers(recurse=False)))
                    for m in transformer.modules()
                )
                if regional_mask is not None:
                    expected += batch * mask_side * (-(-mask_side // 8) * 8) * 2

                estimate = estimate_flux2_denoise_working_memory(
                    transformer,
                    hidden_size=hidden,
                    image_seq_len=image_seq_len,
                    ref_image_seq_len=refs,
                    text_seq_len=caption,
                    batch_size=batch,
                    lora_count=loras,
                    regional_attn_mask=regional_mask,
                    device=attention_build.device,
                    dtype=torch.bfloat16,
                )
                reserved = [estimate.reservation_bytes] + [
                    FLUX2_DENOISE_SPEC.evaluate(
                        estimate.features, shipped_coefficients(FLUX2_DENOISE_SPEC, platform)
                    ).reservation_bytes
                    for platform in LEGACY_PLATFORMS
                ]
                if any(value != expected for value in reserved):
                    case = (hidden, height, width, refs, caption, batch, regional, loras)
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
        """The RTX 4090's native-allocator line, which XPU borrows, prices a 1024px Klein 9B forward at 1788 MiB; the
        formula reserves 2959. The analytic terms are the formula's own (`TestLegacyReservation`), so plain forwards
        compare the lines."""
        short = []
        with ATTENTION["cuda"].answers(), patch.object(spec_module, "platform_for", return_value=platform):
            for hidden, (height, width), refs, caption, batch in itertools.product(
                FLUX2_HIDDEN_SIZES, SIZES, REFERENCE_TOKENS, CAPTIONS, BATCHES
            ):
                image_seq_len = (height // 16) * (width // 16)
                legacy = _legacy_estimate_working_memory(
                    image_seq_len, refs, caption, 0, batch, hidden, device=torch.device("cuda")
                )
                reserved = _estimate(
                    hidden_size=hidden, image=image_seq_len, refs=refs, text=caption, batch=batch
                ).reservation_bytes
                if reserved < legacy:
                    short.append(((hidden, height, width, refs, caption, batch), legacy, reserved))
        assert not short, f"{len(short)} cases below the formula, e.g. (case, legacy, reserved): {short[:3]}"

    @pytest.mark.parametrize("platform", [p for p in EVERY_PLATFORM if is_measured(p)], ids=_platform_id)
    def test_a_measured_platform_prices_with_its_own_values_unfloored(self, platform: Platform):
        """A platform the points are measured on is what they describe, below the formula where they measure less."""
        unfloored = [
            file.model_copy(
                update={"coefficients": [e.model_copy(update={"unmeasured_floor": {}}) for e in file.coefficients]}
            )
            for file in shipped_data().values()
        ]

        assert shipped_coefficients(FLUX2_DENOISE_SPEC, platform) == resolve_coefficients(
            unfloored, FLUX2_DENOISE_SPEC, platform, origin="shipped"
        )


RTX_4090_NATIVE = Platform("cudnn", "native", "windows")


def _forward_features(*, hidden_size=4096, image=4096, refs=0, text=512, batch=1) -> dict:
    with ATTENTION["cuda"].answers():
        return dict(_estimate(hidden_size=hidden_size, image=image, refs=refs, text=text, batch=batch).features)


def _shipped_reservation(platform: Platform, features: dict) -> int:
    return FLUX2_DENOISE_SPEC.evaluate(features, shipped_coefficients(FLUX2_DENOISE_SPEC, platform)).reservation_bytes


# The RTX 4090's forwards at 1024px with a 512-token caption, the default kernel and SDPA forced onto the math kernel,
# measured demand in bytes (`--mode math`, 2026-10-18, native allocator): 8.74 bytes per score element over the default
# kernel at both widths, against the 17 the estimate charges. [dev]'s pair does not fit 24 GB.
MATH_KERNEL_FORWARDS = [(3072, 1317011456, 5769265152), (4096, 1738539008, 7677673472)]


class TestMeasuredChecks:
    """Forwards the shipped points leave out: a term of their own sits on top of the line, or no shipped point measures
    the case."""

    @pytest.mark.parametrize("hidden_size, default_kernel, math_kernel", MATH_KERNEL_FORWARDS)
    def test_a_math_kernel_forward_is_covered_by_the_line_and_its_score_matrix(
        self, hidden_size, default_kernel, math_kernel
    ):
        """A regime no server runs by default, so it bounds the estimate from below only, as the VAE's forced-math
        points do. The score matrix's own term covers what the math kernel adds over the default one."""
        elements = (hidden_size // 128) * 4608 * 4608
        scores = elements * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        features = _forward_features(hidden_size=hidden_size) | {"score_matrix_bytes": scores}

        assert math_kernel - default_kernel <= scores
        assert _shipped_reservation(RTX_4090_NATIVE, features) >= math_kernel

    # (image tokens, reference tokens, measured peak reserved MiB) on the Klein 9B geometry, CUDA (RTX 4090, torch
    # 2.7.1, peak reserved of a warm forward with reduced blocks, the old `scripts/calibrate_flux2_working_memory.py
    # --only denoise`), a 512-token caption, the old build's peaks that still bound the line.
    #
    # A third row, a 1328px tile with three 1328px references (6889 + 20667 tokens, 10954 MiB), is retired. Its shape is
    # unreachable: `Flux2RefImageExtension` resizes each of several references to at most 1024x1024 pixels, so the node
    # builds that forward with 12288 reference tokens, the row above. torch 2.7.1 is a superseded build, which read 2.7
    # to 5.4% above torch 2.13 at every size both measured. And the shipped points now measure the range directly up to
    # 33025 tokens (2048px with one 2024px reference): at width 4096 torch 2.13 takes 9.322 GiB at
    # 25857 tokens and 11.688 GiB at 33025, 10.05 GiB interpolated at that row's 28068, against the old 10.70.
    MEASURED_DENOISE = [
        (4096, 8192, 4840),  # 1024px + two references
        (6889, 12288, 7528),  # 1328px tile + three 1024px references
    ]

    @pytest.mark.parametrize("image, refs, measured_mb", MEASURED_DENOISE)
    def test_the_estimate_covers_the_peaks_measured_before_without_wildly_over_reserving(
        self, image, refs, measured_mb
    ):
        """The cache keeps the estimate free, so under-estimating OOMs; over-estimating offloads the transformer to RAM,
        and a model running over PCIe is indistinguishable from a hang."""
        estimate = _shipped_reservation(RTX_4090_NATIVE, _forward_features(image=image, refs=refs))

        assert measured_mb * MB <= estimate <= measured_mb * MB + 2 * GB
