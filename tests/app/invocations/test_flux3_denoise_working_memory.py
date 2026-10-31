"""FLUX.3 denoise working memory: the estimate as `denoise.flux3` declares it, what the node hands it, and the formula
it replaced.

The estimate prices with the lines `backend/working_memory/data/denoise.json` ships, the fit of the forwards measured
there, which `tests/backend/working_memory/` holds it to. ROCm with the native allocator off Windows is not measured
and still prices with the hand-set `legacy` constants the node reserved by hand, which `TestLegacyReservation` holds
byte for byte to that formula, frozen below, plus the terms it lacked: the largest layer partial loading streams and
an fp8-stored transformer's cast transient (`TestEstimate`). Every other platform nothing is measured on prices as a
measured one, whose entry floors every value at those constants, and `TestUnmeasuredPlatforms` holds it to the formula
at least. `TestMeasuredChecks` holds the estimate to the forced-math forwards the shipped points leave out.
"""

import itertools
from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock, patch

import pytest
import torch
from gguf import GGMLQuantizationType

import invokeai.app.invocations.flux3.flux3_denoise as flux3_denoise
from invokeai.app.invocations.flux3.flux3_denoise import Flux3DenoiseInvocation
from invokeai.backend.flux3.denoise_working_memory import FLUX3_DENOISE_SPEC, estimate_flux3_denoise_working_memory
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loaders.flux3 import _build_transformer
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux3VariantType, ModelType
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, Flux3ConditioningInfo
from invokeai.backend.util.attention import SDPA_MATH_BYTES_PER_SCORE_ELEMENT, sdpa_score_matrix_bytes
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.working_memory import spec as spec_module
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.platform import Allocator, ConvBackend, OperatingSystem, Platform, is_measured
from tests.app.invocations.test_z_image_denoise_working_memory import ATTENTION
from tests.backend.model_manager.load.state_dicts.flux3_klein_keys import state_dict_keys as KLEIN_KEYS

MB = 1024**2
KLEIN_HEADS = 24
HEAD_DIM = 128
WIDTH = KLEIN_HEADS * HEAD_DIM


def _config(*, flow_map: bool = True, heads: int = KLEIN_HEADS) -> SimpleNamespace:
    return SimpleNamespace(flow_map=flow_map, num_attention_heads=heads, attention_head_dim=HEAD_DIM)


def _dense_transformer(**config) -> torch.nn.Module:
    """A one-layer stand-in with the released geometry's config: what the estimate reads off a transformer."""
    model = torch.nn.Module()
    model.config = _config(**config)
    model.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 8)
    return model


def _int8_transformer(out_features: int) -> torch.nn.Module:
    """An int8_tensorwise layer on the meta device: it materializes two weight-sized bf16 tensors per forward."""
    model = torch.nn.Module()
    model.config = _config()
    model.proj = Int8ConvrotLinear(
        torch.zeros(out_features, CONVROT_GROUP_SIZE, dtype=torch.int8, device="meta"),
        torch.ones(out_features, 1, device="meta"),
        convrot=False,
    )
    return model


def _released_transformer(flow_map: bool = True) -> torch.nn.Module:
    """The transformer the loaders build (`_build_transformer`) from the released Klein checkpoint's shapes, one block
    per stack as the captured key fixture holds them, Klein Base without the flow-map embedder, in bf16 on the meta
    device."""
    shapes = {
        key: SimpleNamespace(shape=tuple(shape))
        for key, (shape, _) in KLEIN_KEYS.items()
        if flow_map or not key.startswith("ref_time_in.")
    }
    return torch.nn.Module.to(_build_transformer(shapes), torch.bfloat16)


def _estimate(
    transformer=None, *, image=4096, refs=0, text=560, batch=1, resident=False, dtype=torch.bfloat16, device="cuda"
):
    return estimate_flux3_denoise_working_memory(
        transformer if transformer is not None else _dense_transformer(),
        image_seq_len=image,
        ref_image_seq_len=refs,
        text_seq_len=text,
        batch_size=batch,
        weights_resident=resident,
        device=torch.device(device),
        dtype=dtype,
    )


class TestEstimate:
    @pytest.mark.parametrize("flow_map", [True, False], ids=["klein", "klein-base"])
    def test_a_transformer_that_may_load_partially_reserves_its_largest_layer(self, flow_map):
        """Partial loading copies a RAM-held layer to the device for its forward, alive beside the activations. On the
        released geometry in bf16 the largest is the caption embedder, 20480 x 3072, for Klein and Klein Base alike."""
        model = _released_transformer(flow_map)

        with ATTENTION["cuda"].answers():
            added = _estimate(model).reservation_bytes - _estimate(model, resident=True).reservation_bytes

        assert added == 20480 * 3072 * 2 == 125_829_120

    def test_a_gguf_layer_streams_its_quantized_bytes(self):
        """A Q8_0 weight stores 34 bytes per block of 32 elements; its dequantized copy is the dequant term's."""
        model = _dense_transformer()
        quantized = GGMLTensor(
            torch.zeros(64, 34, dtype=torch.uint8), GGMLQuantizationType.Q8_0, torch.Size([64, 32]), torch.bfloat16
        )
        model.proj = torch.nn.Linear(32, 64, bias=False)
        model.proj.weight = torch.nn.Parameter(quantized, requires_grad=False)

        with ATTENTION["cuda"].answers():
            features = _estimate(model).features

        assert features["streamed_layer_bytes"] == 64 * 34

    def test_an_fp8_stored_transformer_reserves_its_largest_layer_widened(self):
        """FP8 Storage widens a layer's weight and bias to the compute dtype for its forward."""
        fp8_stored = _dense_transformer()
        ModelLoader._apply_fp8_to_nn_module(fp8_stored, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

        with ATTENTION["cuda"].answers():
            added = _estimate(fp8_stored, resident=True).reservation_bytes - _estimate(resident=True).reservation_bytes
            streamed = _estimate(fp8_stored).features["streamed_layer_bytes"]

        assert added == (8 * CONVROT_GROUP_SIZE + 8) * torch.bfloat16.itemsize
        # What partial loading copies is the stored float8 weight; its widening is the cast term's.
        assert streamed == (8 * CONVROT_GROUP_SIZE + 8) * 1

    def test_a_quantized_transformer_reserves_its_dequantization_copy(self):
        """An int8_tensorwise layer materializes its weight and a scaled copy in bf16 per forward."""
        with ATTENTION["cuda"].answers():
            added = (
                _estimate(_int8_transformer(64), resident=True).reservation_bytes
                - _estimate(resident=True).reservation_bytes
            )

        assert added == 2 * 64 * CONVROT_GROUP_SIZE * 2

    @pytest.mark.parametrize("batch", [1, 2])
    def test_a_materialized_score_matrix_counts_the_transformers_heads_per_sample_over_the_joint_sequence(self, batch):
        """The joint blocks attend over the caption, the image and the references at once."""
        tokens = 4096 + 4096 + 560

        with ATTENTION["mps"].answers():
            scores = _estimate(refs=4096, batch=batch, resident=True).features["score_matrix_bytes"]

        assert scores == KLEIN_HEADS * batch * tokens * tokens * SDPA_MATH_BYTES_PER_SCORE_ELEMENT

    @pytest.mark.parametrize("heads, dtype", [(32, torch.bfloat16), (KLEIN_HEADS, torch.float64)])
    def test_a_dtype_or_width_without_a_line_is_refused(self, heads, dtype):
        """Pricing it on another width's or dtype's line would be a guess; the estimator says what is missing."""
        with pytest.raises(ValueError, match=r"'denoise\.flux3' has no torch\.\w+ line at width \d+ on Platform\("):
            _estimate(_dense_transformer(heads=heads), dtype=dtype, device="cpu")


class TestLine:
    """Which forwards a point of may fit the `denoise_<dtype>_h3072` line: the plain ones, whose demand is the line's
    alone."""

    def _line(self, *args, resident=True, **kwargs):
        with ATTENTION["cuda"].answers():
            features = _estimate(*args, resident=resident, **kwargs).features
        return FLUX3_DENOISE_SPEC.line(features)

    @pytest.mark.parametrize(
        "dtype, name", [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]
    )
    def test_a_plain_forward_fits_its_dtypes_line_over_every_attended_token(self, dtype, name):
        line = self._line(image=4096, refs=40960, text=80, dtype=dtype)

        assert (line.suffix, line.size, line.fits) == (f"denoise_{name}_h3072", 4096 + 40960 + 80, True)
        assert line.label == f"denoise-{name}-h3072-4096+40960ref+80tok"

    def test_a_batch_fits_the_line_at_its_sequences_together(self):
        """The line prices what a batch adds, its sequences; the base is a forward's, paid once."""
        line = self._line(batch=2)

        assert (line.size, line.fits, line.label) == (2 * (4096 + 560), True, "denoise-bf16-h3072-4096+560tok-b2")

    def test_klein_base_prices_on_klein_s_line_and_its_points_say_so(self):
        klein = self._line(_dense_transformer(flow_map=True))
        base = self._line(_dense_transformer(flow_map=False))

        assert (base.suffix, base.size, base.fits) == (klein.suffix, klein.size, True)
        assert base.label == f"{klein.label}-base"

    def test_a_forward_with_an_analytic_term_fits_nothing_and_says_why(self):
        quantized = self._line(_int8_transformer(64))
        streamed = self._line(_released_transformer(), resident=False)
        with ATTENTION["mps"].answers():
            scored = FLUX3_DENOISE_SPEC.line(_estimate(resident=True, device="mps").features)

        assert not (quantized.fits or streamed.fits or scored.fits)
        assert quantized.label.endswith(f"-dequant_transient={2 * 64 * CONVROT_GROUP_SIZE * 2}")
        assert streamed.label.endswith(f"-streamed_layer={20480 * 3072 * 2}")
        scores = KLEIN_HEADS * (4096 + 560) ** 2 * SDPA_MATH_BYTES_PER_SCORE_ELEMENT
        assert scored.label.endswith(f"-attention_scores={scores}")


# The reservation the node priced by hand before its estimate declared an `EstimatorSpec`, frozen verbatim with the
# node's constants: `Flux3DenoiseInvocation._estimate_working_memory` and the dequant transient the node added at the
# lock. The shipped `legacy` values reproduce it where no measured points replace them.
_LEGACY_ATTENTION_HEAD_DIM = 128
_LEGACY_ATTENTION_HEADS = 24
_LEGACY_BYTES_PER_TOKEN = int(0.33 * 1024**2)


def _legacy_estimate_working_memory(num_tokens: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> int:
    estimated = num_tokens * batch_size * _LEGACY_BYTES_PER_TOKEN + int(1.0 * 1024**3)
    estimated += sdpa_score_matrix_bytes(
        device=device,
        dtype=dtype,
        num_heads=_LEGACY_ATTENTION_HEADS * batch_size,
        head_dim=_LEGACY_ATTENTION_HEAD_DIM,
        seq_len=num_tokens,
        has_attn_mask=False,
        via_diffusers_dispatch=True,
    )
    return estimated


# Where the shipped entry is still `legacy`: ROCm with the native allocator on every OS but Windows.
LEGACY_PLATFORMS = [Platform("miopen", "native", "linux"), Platform("miopen", "native", "other")]
# Height, width: an elongated frame included.
SIZES = [(512, 512), (784, 1040), (1024, 1024), (2048, 2048)]
# None, one 1024px reference, one 4 MP one, three and ten 1 MP ones.
REFERENCE_TOKENS = [0, 4096, 16384, 12288, 40960]
CAPTIONS = [80, 560, 8192]
BATCHES = [1, 2]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# A transient of 123 MiB in bf16; on the meta device, so it costs no memory.
TRANSFORMERS = [_dense_transformer(), _int8_transformer(123 * MB // (2 * CONVROT_GROUP_SIZE * 2))]


class TestLegacyReservation:
    @pytest.mark.parametrize("attention_build", list(ATTENTION.values()), ids=list(ATTENTION))
    def test_the_reservation_is_the_legacy_formula_byte_for_byte(self, attention_build):
        """On the platforms still `legacy`: estimated on such a machine, and evaluated again there from the features
        alone, plus the term the formula lacked that `TestEstimate` pins, the streamed largest layer. None of the
        transformers holds a float8 weight, whose cast is the other such term."""
        mismatches = []
        with attention_build.answers(), patch.object(spec_module, "platform_for", return_value=LEGACY_PLATFORMS[0]):
            grid = itertools.product(SIZES, REFERENCE_TOKENS, CAPTIONS, BATCHES, DTYPES, TRANSFORMERS)
            for (height, width), refs, caption, batch, dtype, transformer in grid:
                image_seq_len = (height // 16) * (width // 16)
                expected = _legacy_estimate_working_memory(
                    image_seq_len + refs + caption, batch, attention_build.device, dtype
                ) + peak_dequant_transient_bytes(transformer, dtype)
                expected += max(
                    sum(t.nbytes for t in (*m.parameters(recurse=False), *m.buffers(recurse=False)))
                    for m in transformer.modules()
                )

                estimate = estimate_flux3_denoise_working_memory(
                    transformer,
                    image_seq_len=image_seq_len,
                    ref_image_seq_len=refs,
                    text_seq_len=caption,
                    batch_size=batch,
                    device=attention_build.device,
                    dtype=dtype,
                )
                reserved = [estimate.reservation_bytes] + [
                    FLUX3_DENOISE_SPEC.evaluate(
                        estimate.features, shipped_coefficients(FLUX3_DENOISE_SPEC, platform)
                    ).reservation_bytes
                    for platform in LEGACY_PLATFORMS
                ]
                if any(value != expected for value in reserved):
                    mismatches.append(((height, width, refs, caption, batch, dtype), expected, reserved))
        assert not mismatches, f"{len(mismatches)} cases differ, e.g. (case, legacy, reserved): {mismatches[:3]}"


EVERY_PLATFORM = [
    Platform(*keys) for keys in itertools.product(get_args(ConvBackend), get_args(Allocator), get_args(OperatingSystem))
]


class TestUnmeasuredPlatforms:
    @pytest.mark.parametrize(
        "platform",
        [p for p in EVERY_PLATFORM if not is_measured(p)],
        ids=lambda p: f"{p.conv_backend}-{p.allocator}-{p.os}",
    )
    def test_an_unmeasured_platform_reserves_the_legacy_formula_at_least(self, platform: Platform):
        """`priced_as` prices it with a measured platform's lines; the entries it borrows floor every value at the
        legacy constants, so it never reserves less than the node did by hand."""
        short = []
        with ATTENTION["cuda"].answers(), patch.object(spec_module, "platform_for", return_value=platform):
            for (height, width), refs, caption, batch, dtype in itertools.product(
                SIZES, REFERENCE_TOKENS, CAPTIONS, BATCHES, DTYPES
            ):
                image_seq_len = (height // 16) * (width // 16)
                legacy = _legacy_estimate_working_memory(
                    image_seq_len + refs + caption, batch, torch.device("cuda"), dtype
                )
                reserved = _estimate(
                    image=image_seq_len, refs=refs, text=caption, batch=batch, dtype=dtype
                ).reservation_bytes
                if reserved < legacy:
                    short.append(((height, width, refs, caption, batch, dtype), legacy, reserved))
        assert not short, f"{len(short)} cases below the formula, e.g. (case, legacy, reserved): {short[:3]}"


class _StopBeforeLoad(Exception):
    """Raised in place of entering the transformer's device context, to end `_run_diffusion` early."""


SIZE = 256
IMAGE_TOKENS = (SIZE // 16) ** 2


def _requested_features(
    *,
    variant: Flux3VariantType = Flux3VariantType.Klein,
    cfg_scale: float = 1.0,
    positive_tokens: int = 80,
    negative_tokens: int | None = None,
    reference_tokens: int = 0,
    init_batch: int | None = None,
) -> dict:
    """Drive `_run_diffusion` up to the transformer's lock and return the features of the estimate it hands
    `model_on_device()`, after checking that it hands no plain byte count."""
    transformer = _dense_transformer(flow_map=variant is Flux3VariantType.Klein)
    transformer_info = MagicMock()
    transformer_info.model = transformer
    transformer_info.model_on_device = MagicMock(side_effect=_StopBeforeLoad)
    vae_info = MagicMock()
    vae_info.model_on_device.return_value.__enter__.return_value = (None, MagicMock())

    context = MagicMock()
    context.models.get_config.return_value = SimpleNamespace(
        base=BaseModelType.Flux3, type=ModelType.Main, name="flux3", variant=variant
    )
    context.models.load.side_effect = lambda key: vae_info if key == "vae" else transformer_info
    lengths = {"pos": positive_tokens, "neg": negative_tokens}
    context.conditioning.load.side_effect = lambda name: ConditioningFieldData(
        conditionings=[Flux3ConditioningInfo(prompt_embeds=torch.zeros(1, lengths[name], 8))]
    )
    if init_batch is not None:
        context.tensors.load.return_value = torch.zeros(init_batch, 32, SIZE // 8, SIZE // 8)

    references = [SimpleNamespace(image=SimpleNamespace(image_name="ref"))] if reference_tokens else None
    invocation = Flux3DenoiseInvocation.model_construct(
        latents=SimpleNamespace(latents_name="init") if init_batch is not None else None,
        denoise_mask=None,
        denoising_start=0.0,
        denoising_end=1.0,
        add_noise=True,
        transformer=SimpleNamespace(transformer="transformer", loras=[]),
        positive_conditioning=SimpleNamespace(conditioning_name="pos"),
        negative_conditioning=SimpleNamespace(conditioning_name="neg") if negative_tokens is not None else None,
        cfg_scale=cfg_scale,
        width=SIZE,
        height=SIZE,
        steps=4,
        seed=0,
        vae=SimpleNamespace(vae="vae"),
        kontext_conditioning=references,
    )
    encoded = (torch.zeros(1, reference_tokens, 128), torch.zeros(reference_tokens, 4))

    with (
        patch.object(flux3_denoise.TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
        patch.object(flux3_denoise, "read_vae_bn_stats", return_value=(torch.zeros(128), torch.ones(128))),
        patch.object(flux3_denoise, "encode_flux3_references", return_value=encoded),
        pytest.raises(_StopBeforeLoad),
    ):
        invocation._run_diffusion(context)

    transformer_info.model_on_device.assert_called_once()
    requested = transformer_info.model_on_device.call_args.kwargs
    assert set(requested) == {"working_memory"}
    estimate = requested["working_memory"]
    assert estimate.estimator == FLUX3_DENOISE_SPEC.id
    return dict(estimate.features)


class TestFlux3DenoiseRequestsWorkingMemory:
    """What the node hands the estimate: one sample's tokens of each stream, the samples the forward runs, the dtype
    it runs in and the variant, read off the transformer it is about to lock."""

    def test_a_plain_generation_prices_its_image_and_caption_tokens(self):
        features = _requested_features()

        assert (features["image_tokens"], features["reference_tokens"], features["text_tokens"]) == (
            IMAGE_TOKENS,
            0,
            80,
        )
        assert (features["tokens"], features["batch_size"], features["flow_map"]) == (IMAGE_TOKENS + 80, 1, True)
        names = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
        assert features["dtype"] == names[TorchDevice.choose_bfloat16_safe_dtype(torch.device("cpu"))]

    def test_the_layer_partial_loading_streams_is_reserved(self):
        """The node cannot know at the estimate whether the cache loads the transformer whole, so it reserves the
        stand-in's one layer, a float32 `Linear` of 8 x `CONVROT_GROUP_SIZE` with its bias."""
        assert _requested_features()["streamed_layer_bytes"] == (8 * CONVROT_GROUP_SIZE + 8) * 4

    def test_reference_images_join_the_attended_tokens(self):
        """Ten 1 MP references add 40960 tokens to a generation of any size."""
        features = _requested_features(reference_tokens=40960)

        assert (features["reference_tokens"], features["tokens"]) == (40960, IMAGE_TOKENS + 40960 + 80)

    def test_a_batched_init_latent_sets_the_batch(self):
        """The node builds its noise at batch 1; two init latents broadcast the blend to two samples, references
        repeated per sample."""
        features = _requested_features(init_batch=2, reference_tokens=4096)

        assert (features["batch_size"], features["reference_tokens"]) == (2, 4096)

    @pytest.mark.parametrize("negative, text", [(160, 160), (80, 80)], ids=["negative-longer", "positive-longer"])
    def test_klein_base_cfg_prices_the_longer_caption_of_the_pair(self, negative, text):
        """The conditional and unconditional forwards run one after the other, so the longer caption is what one
        forward attends."""
        features = _requested_features(
            variant=Flux3VariantType.KleinBase, cfg_scale=3.5, positive_tokens=80, negative_tokens=negative
        )

        assert (features["text_tokens"], features["flow_map"]) == (text, False)

    def test_a_negative_caption_without_cfg_runs_no_forward_and_is_not_priced(self):
        features = _requested_features(variant=Flux3VariantType.KleinBase, negative_tokens=800)

        assert features["text_tokens"] == 80


# The forwards at 1024px with a 560-token caption, 4656 tokens, the default kernel and SDPA forced onto the math kernel
# (`scripts/calibrate_flux3_denoise_working_memory.py --mode math`, 2026-10-31): platform, the math forward's score term
# as the estimate priced it in that process, and the measured demands in bytes. On the RTX 4090 (native allocator) the
# math kernel took 13.69 bytes per score element over the default one, against the 17 the term charges. On the RX 9060
# XT (expandable segments) the ROCm SDPA guard computes math attention in chunks, so the term is one 1 GiB chunk and the
# forward took 129 MiB more.
MATH_KERNEL_FORWARDS = [
    (Platform("cudnn", "native", "windows"), KLEIN_HEADS * 4656 * 4656 * 17, 1411383296, 8535408640),
    (Platform("miopen", "expandable", "windows"), 2**30, 999686144, 1134821376),
]


class TestMeasuredChecks:
    @pytest.mark.parametrize("platform, scores, default_kernel, math_kernel", MATH_KERNEL_FORWARDS, ids=["cuda", "hip"])
    def test_a_math_kernel_forward_is_covered_by_the_line_and_its_score_matrix(
        self, platform, scores, default_kernel, math_kernel
    ):
        """A regime no server runs by default, so it bounds the estimate from below only. The score matrix's own term
        covers what the math kernel adds over the default one."""
        with ATTENTION["cuda"].answers():
            features = dict(_estimate(image=4096, refs=0, text=560, resident=True).features)
        features["score_matrix_bytes"] = scores
        reserved = FLUX3_DENOISE_SPEC.evaluate(features, shipped_coefficients(FLUX3_DENOISE_SPEC, platform))

        assert math_kernel - default_kernel <= scores
        assert reserved.reservation_bytes >= math_kernel
