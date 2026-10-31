"""Krea-2 denoise working memory: the estimate as `denoise.krea2` declares it, against the formula it replaced.

The terms are evaluated with coefficients of the tests' own where they test the estimator's structure, so those hold
whatever `backend/working_memory/data/denoise.json` ships; `tests/backend/working_memory/` holds the shipped values to
their points. ROCm with the native allocator off Windows is not measured and prices with the hand-set formula the node
reserved before the lines were measured, which `TestLegacyReservation` holds byte for byte to that formula, frozen
below; every other platform nothing measures reserves that formula at least (`TestUnmeasuredPlatforms`). What the
node hands the estimate and `model_on_device()` is pinned in `tests/app/invocations/test_krea2_denoise.py`.
"""

import itertools
import math
from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

import pytest
import torch

from invokeai.backend.krea2 import denoise_working_memory
from invokeai.backend.krea2.denoise_working_memory import KREA2_DENOISE_SPEC, estimate_krea2_denoise_working_memory
from invokeai.backend.krea2.reference_edit import reference_kv_cache_bytes
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.fp8 import peak_fp8_cast_transient_bytes
from invokeai.backend.working_memory import spec as spec_module
from invokeai.backend.working_memory.coefficients import ResolvedCoefficients
from invokeai.backend.working_memory.platform import Allocator, ConvBackend, OperatingSystem, Platform, is_measured
from tests.app.invocations.test_z_image_denoise_working_memory import ATTENTION

MB = 1024**2
GB = 1024**3
K = 100_000
"""The tests' bytes per attended token."""
B = 64 * MB
"""The tests' base."""
CONFIG = SimpleNamespace(num_attention_heads=48, num_key_value_heads=12, attention_head_dim=128)
"""The Krea-2 transformer's attention geometry."""


def _coefficients(k: float = K, b: float = B) -> ResolvedCoefficients:
    return ResolvedCoefficients(
        values={
            f"{part}_denoise_{dtype}": k if part == "k" else b for dtype in ("bf16", "fp16", "fp32") for part in "kb"
        },
        source="test",
    )


def _transformer(*blocks: torch.nn.Module, **outer: torch.nn.Module) -> torch.nn.Module:
    """A stand-in with the transformer's layout: its config, the `transformer_blocks` and the layers outside them."""
    model = torch.nn.Module()
    model.config = CONFIG
    model.transformer_blocks = torch.nn.ModuleList(blocks)
    for name, layer in outer.items():
        setattr(model, name, layer)
    return model


def _int8(out_features: int) -> Int8ConvrotLinear:
    """An int8_tensorwise linear on the meta device: it materializes two weight-sized compute-dtype copies per forward."""
    return Int8ConvrotLinear(
        torch.zeros(out_features, CONVROT_GROUP_SIZE, dtype=torch.int8, device="meta"),
        torch.ones(out_features, 1, device="meta"),
        convrot=False,
    )


def _fp8(out_features: int) -> torch.nn.Module:
    """A linear under FP8 Storage: float8 weights widened to bf16 for its forward."""
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, out_features)
    ModelLoader._apply_fp8_to_nn_module(model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    return model.proj


def _estimate(
    transformer=None, *, image=4096, text=512, refs=0, cached=False, resident=True, dtype=torch.bfloat16, **kw
):
    with ATTENTION["cuda"].answers():
        return estimate_krea2_denoise_working_memory(
            transformer if transformer is not None else _transformer(),
            image_seq_len=image,
            text_seq_len=text,
            reference_seq_len=refs,
            cache_references=cached,
            weights_resident=resident,
            device=ATTENTION["cuda"].device,
            dtype=dtype,
            **kw,
        )


def _features(transformer=None, **kw):
    return _estimate(transformer, **kw).features


def _terms(features, k: float = K, b: float = B) -> dict[str, float]:
    return {term.name: term.bytes for term in KREA2_DENOISE_SPEC.evaluate(features, _coefficients(k, b)).terms}


def _reservation(features, k: float = K, b: float = B) -> int:
    return KREA2_DENOISE_SPEC.evaluate(features, _coefficients(k, b)).reservation_bytes


def _patch(*layers: tuple[str, object]) -> tuple[ModelPatchRaw, int]:
    """A loaded LoRA of `layers` by module path, with a nominal file size."""
    return ModelPatchRaw(layers={f"lora_transformer-{path}": layer for path, layer in layers}), MB


class TestEstimate:
    def test_the_plain_forward_prices_image_caption_and_joint_reference_tokens_on_the_line(self) -> None:
        """Joint references are image tokens to every block, so their forward fits the line as the plain one does."""
        for refs in (0, 4096, 3 * 4096):
            features = _features(refs=refs)
            line = KREA2_DENOISE_SPEC.line(features)

            assert (line.suffix, line.size, line.fits) == ("denoise_bf16", 4096 + 512 + refs, True)
            assert _reservation(features) == K * (4096 + 512 + refs) + B

    @pytest.mark.parametrize(
        ("image", "text", "tokens"),
        [(1024, 77, 3 * 4096), (16384, 512, 16384 + 512)],
        ids=["precompute-longer", "step-longer"],
    )
    def test_cached_references_price_the_longer_pass_on_the_line_plus_their_cache(self, image, text, tokens) -> None:
        """In K/V-cache mode the references run once, alone, and every step attends to their cached keys: the line
        prices the larger of the two passes, and the cache is an exact size on top, for the transformer's own block
        count and heads."""
        transformer = _transformer(*(torch.nn.Identity() for _ in range(5)))
        features = _features(transformer, image=image, text=text, refs=3 * 4096, cached=True)
        cache = reference_kv_cache_bytes(
            num_layers=5, num_heads=48, num_kv_heads=12, head_dim=128, reference_seq_len=3 * 4096, element_size=2
        )

        assert features["tokens"] == tokens
        assert _reservation(features) == K * tokens + B + cache
        assert not KREA2_DENOISE_SPEC.line(features).fits

    def test_a_style_pass_with_a_longer_caption_is_the_longest_pass(self) -> None:
        """A style reference's pass runs alone over the image and its own caption."""
        features = _features(image=4096, text=77, style_text_seq_len=300, style_kv_bytes=MB)

        assert features["tokens"] == 4096 + 300

    def test_a_materialized_score_matrix_is_priced_for_the_cached_precompute_square(self, monkeypatch) -> None:
        """Five cached 1024px references outnumber a 1024px step's 4608 tokens even with their keys: the precompute's
        R x R is the larger area. The node tests pin the joint and the step-area cases."""
        asked: list[dict] = []

        def score_matrix_bytes(**kwargs) -> int:
            asked.append(kwargs)
            return 7 * MB

        monkeypatch.setattr(denoise_working_memory, "sdpa_score_matrix_bytes", score_matrix_bytes)
        features = _features(refs=5 * 4096, cached=True)

        assert [call["seq_len"] for call in asked] == [5 * 4096]
        assert _terms(features)["attention_scores"] == 7 * MB

    @pytest.mark.parametrize("lora_bytes", [30 * MB, 210 * MB, 2 * GB])
    def test_plain_loras_add_their_file_size(self, lora_bytes) -> None:
        plain = _features()
        patched = _features(loras=[(ModelPatchRaw(layers={}), lora_bytes)])

        assert _reservation(patched) - _reservation(plain) == lora_bytes
        assert not KREA2_DENOISE_SPEC.line(patched).fits

    @pytest.mark.parametrize(
        ("quantized", "transient"),
        [
            (_int8, lambda layer: peak_dequant_transient_bytes(layer, torch.bfloat16)),
            (_fp8, lambda layer: peak_fp8_cast_transient_bytes(layer, torch.bfloat16)),
        ],
        ids=["int8-dequant", "fp8-cast"],
    )
    def test_the_block_phase_prices_the_block_layers_transient_and_the_outer_layer_only_its_excess(
        self, quantized, transient
    ) -> None:
        """The timestep modulation never meets the block activations: its transient (here 3x a block layer's) adds
        nothing where the block phase is larger; where it is smaller, the reservation is that one layer's need."""
        block_layer, outer_layer = quantized(256), quantized(768)
        block_transient, outer_transient = transient(block_layer), transient(outer_layer)
        assert outer_transient == 3 * block_transient > 0
        features = _features(_transformer(block_layer, time_mod_proj=outer_layer))
        term = "dequant_transient" if quantized is _int8 else "fp8_cast_transient"

        large_line = _terms(features)
        assert (large_line[term], large_line["outer_layer_peak"]) == (block_transient, 0)
        assert _reservation(features) == K * (4096 + 512) + B + block_transient

        assert _reservation(features, k=0, b=0) == outer_transient
        assert not KREA2_DENOISE_SPEC.line(features).fits

    def test_a_transformer_that_may_load_partially_reserves_its_largest_block_layer(self) -> None:
        """A streamed layer is copied to the device for its forward; a resident model streams nothing. A larger layer
        outside the blocks counts only where it exceeds the block phase."""
        block_layer = torch.nn.Linear(64, 32, dtype=torch.bfloat16, device="meta")
        outer_layer = torch.nn.Linear(64, 256, dtype=torch.bfloat16, device="meta")
        transformer = _transformer(torch.nn.Sequential(block_layer), time_mod_proj=outer_layer)

        streaming = _features(transformer, resident=False)
        resident = _features(transformer, resident=True)

        assert _terms(streaming)["streamed_layer"] == (64 * 32 + 32) * 2
        assert _terms(resident)["streamed_layer"] == 0
        assert _reservation(resident) == K * (4096 + 512) + B
        assert _reservation(streaming, k=0, b=0) == (64 * 256 + 256) * 2

    def test_regional_masks_add_their_bytes_and_the_scores_a_masked_call_materializes(self) -> None:
        """A mask narrows the kernels SDPA can pick: on CUDA a masked call materializes the score matrix."""
        features = _features(regional_mask_bytes=5 * MB)
        terms = _terms(features)
        line = KREA2_DENOISE_SPEC.line(features)

        assert features["has_attention_mask"] is True and terms["attention_scores"] > 0
        assert _reservation(features) - _reservation(_features()) == 5 * MB + terms["attention_scores"]
        assert not line.fits and "attention_mask" in line.label

    def test_a_style_reference_covers_its_measured_overhead_at_1024px_and_not_much_more(self) -> None:
        """Peak VRAM rose ~1.6 GiB over the unstyled run at 1024x1024 (8 steps, cfg 1), 0.5 GiB of it the captured
        K/V of the default 21 styled blocks. The style terms are analytic, so this holds whatever the line; they stay
        within 1.25x of the measurement."""
        style_kv_bytes = 21 * 2 * 12 * 4096 * 128 * 2
        plain = _features(image=4096, text=512)
        styled = _features(image=4096, text=512, style_kv_bytes=style_kv_bytes)

        for k, b in ((0, 0), (K, B)):
            assert 1.6 * GB < _reservation(styled, k, b) - _reservation(plain, k, b) < 1.25 * 1.6 * GB
        assert _terms(plain)["style_reference_activations"] == 0
        assert not KREA2_DENOISE_SPEC.line(styled).fits

    @pytest.mark.parametrize(
        ("dtype", "per_token"), [(torch.bfloat16, 0.34), (torch.float16, 0.34), (torch.float32, 0.74)]
    )
    def test_caption_tokens_beyond_the_fit_grids_512_reserve_their_measured_excess(self, dtype, per_token) -> None:
        """Up to 512 caption tokens the line covers a caption's excess over an image token; beyond, each adds the
        caption checks' largest excess with the fitted terms' 1.3 margin, whichever pass carries the caption."""
        excess = int(per_token * MB)
        within = _features(text=512, dtype=dtype)
        regional = _features(text=1500, dtype=dtype)
        styled = _features(text=77, style_text_seq_len=1000, style_kv_bytes=MB, dtype=dtype)

        assert _terms(within)["caption_excess"] == 0
        assert _terms(regional)["caption_excess"] == (1500 - 512) * excess
        assert _terms(styled)["caption_excess"] == (1000 - 512) * excess
        assert not KREA2_DENOISE_SPEC.line(regional).fits

    def test_a_dtype_without_a_line_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'denoise.krea2' has no torch.float64 line"):
            estimate_krea2_denoise_working_memory(
                _transformer(), image_seq_len=4096, text_seq_len=512, device=torch.device("cpu"), dtype=torch.float64
            )


class TestSidecarResidual:
    """A patch that is no `LoRALayer` costs a full residual weight per forward on a sidecar-patched layer, beside the
    re-dequantized base weight; a plain LoRA layer runs as two thin matmuls and adds only its file."""

    def test_a_quantized_layer_patched_by_a_full_layer_reserves_its_transient_and_residual_once(self) -> None:
        small, large = _int8(256), _int8(768)
        features = _features(
            _transformer(small, large),
            loras=[
                _patch(("transformer_blocks.0", FullLayer(torch.zeros(1, 1), None))),
                _patch(("transformer_blocks.1", FullLayer(torch.zeros(1, 1), None))),
            ],
        )
        residual = peak_dequant_transient_bytes(large, torch.bfloat16) + 768 * CONVROT_GROUP_SIZE * (4 + 2)

        assert _terms(features)["sidecar_patch_residual"] == residual
        assert not KREA2_DENOISE_SPEC.line(features).fits

    def test_a_plain_lora_layer_adds_no_residual(self) -> None:
        lora = LoRALayer(
            up=torch.zeros(768, 1), mid=None, down=torch.zeros(1, CONVROT_GROUP_SIZE), alpha=None, bias=None
        )
        features = _features(_transformer(_int8(768)), loras=[_patch(("transformer_blocks.0", lora))])

        assert _terms(features)["sidecar_patch_residual"] == 0

    def test_a_dense_layer_is_a_sidecar_only_where_it_may_be_streamed(self) -> None:
        """A resident dense layer is patched directly; one partial loading may leave in RAM is a sidecar, copied to
        the device again for the residual."""
        layer = torch.nn.Linear(64, 32, dtype=torch.bfloat16, device="meta")
        loras = [_patch(("transformer_blocks.0", FullLayer(torch.zeros(1, 1), None)))]

        resident = _features(_transformer(layer), loras=loras, resident=True)
        streaming = _features(_transformer(layer), loras=loras, resident=False)

        assert _terms(resident)["sidecar_patch_residual"] == 0
        assert _terms(streaming)["sidecar_patch_residual"] == (64 * 32 + 32) * 2 + 64 * 32 * (4 + 2)

    def test_a_key_the_transformer_does_not_have_is_skipped_as_the_patcher_skips_it(self) -> None:
        loras = [_patch(("no_such.layer", FullLayer(torch.zeros(1), None)))]
        features = _features(_transformer(_int8(768)), loras=loras)

        assert _terms(features)["sidecar_patch_residual"] == 0


class TestMeasuredPlatforms:
    """Fitted values on a measured platform price the measured terms alone, never the formula they replaced."""

    def test_cfg_and_the_lora_count_add_nothing_beyond_the_measured_terms(self) -> None:
        plain = _features()
        guided = _features(cfg=True, loras=[(ModelPatchRaw(layers={}), MB)] * 2)

        assert _reservation(guided) - _reservation(plain) == 2 * MB


def _legacy_estimate_working_memory(
    image_seq_len: int,
    positive_text_seq_len: int,
    negative_text_seq_len: int | None,
    do_cfg: bool,
    num_loras: int,
    regional_attention_mask_bytes: int = 0,
    style_reference_kv_bytes: int = 0,
    reference_precompute_seq_len: int = 0,
    reference_kv_bytes: int = 0,
) -> int:
    """`Krea2DenoiseInvocation._estimate_working_memory` as the node priced with it by hand, frozen."""
    GB = 1024**3
    MB = 1024**2
    per_token_bytes = int(0.5 * MB)
    text_seq_len = max(positive_text_seq_len, negative_text_seq_len or 0)
    estimated = max(image_seq_len + text_seq_len, reference_precompute_seq_len) * per_token_bytes
    estimated += int(1.5 * GB)
    if do_cfg:
        estimated = int(estimated * 1.1)
    estimated += regional_attention_mask_bytes
    if style_reference_kv_bytes > 0:
        estimated = int(estimated * 1.35)
        estimated += style_reference_kv_bytes
    estimated += reference_kv_bytes
    if num_loras > 0:
        estimated += int(0.5 * num_loras * GB)
    return estimated


LEGACY_PLATFORMS = [Platform("miopen", "native", os) for os in get_args(OperatingSystem) if os != "windows"]
TRANSFORMERS = [
    _transformer(torch.nn.Identity(), torch.nn.Identity()),
    _transformer(_int8(256), _int8(256), time_mod_proj=_int8(768)),
]
# Image tokens, caption, references, cached: no reference beside a regional mask or a style reference, as the node
# refuses those.
SHAPES = [
    (4096, 77, 0, False),
    (6912, 512, 0, False),
    (16384, 512, 0, False),
    (6912, 84, 3888, False),
    (4096, 512, 12288, True),
]


def _cases():
    grid = itertools.product(
        SHAPES, (False, True), (0, 2), (0, 123 * MB), (0, 500 * MB), TRANSFORMERS, (torch.bfloat16, torch.float32)
    )
    for shape, cfg, loras, regional, style, transformer, dtype in grid:
        if shape[2] and (regional or style):
            continue
        yield shape, cfg, loras, regional, style, transformer, dtype


def _reserved(platform, attention, shape, cfg, loras, regional, style, transformer, dtype) -> int:
    image, text, refs, cached = shape
    with attention.answers(), patch.object(spec_module, "platform_for", return_value=platform):
        return estimate_krea2_denoise_working_memory(
            transformer,
            image_seq_len=image,
            text_seq_len=text,
            reference_seq_len=refs,
            cache_references=cached,
            cfg=cfg,
            loras=[(ModelPatchRaw(layers={}), 200 * MB)] * loras,
            regional_mask_bytes=regional,
            style_kv_bytes=style,
            weights_resident=True,
            device=attention.device,
            dtype=dtype,
        ).reservation_bytes


def _legacy(attention, shape, cfg, loras, regional, style, transformer, dtype) -> int:
    """The frozen formula plus what the node added to it: the score matrix, for the configured heads and the attended
    sequence (for cached references the side of the larger area), and the whole model's largest dequantization."""
    image, text, refs, cached = shape
    step = image + text
    side = math.isqrt(max(step * (step + refs), refs**2) - 1) + 1 if cached else step + refs
    kv = 0
    if cached:
        kv = reference_kv_cache_bytes(
            num_layers=len(transformer.transformer_blocks),
            num_heads=48,
            num_kv_heads=12,
            head_dim=128,
            reference_seq_len=refs,
            element_size=dtype.itemsize,
        )
    formula = _legacy_estimate_working_memory(
        image_seq_len=image + (0 if cached else refs),
        positive_text_seq_len=text,
        negative_text_seq_len=None,
        do_cfg=cfg,
        num_loras=loras,
        regional_attention_mask_bytes=regional,
        style_reference_kv_bytes=style,
        reference_precompute_seq_len=refs if cached else 0,
        reference_kv_bytes=kv,
    )
    with attention.answers():
        score = sdpa_score_matrix_bytes(
            device=attention.device, dtype=dtype, num_heads=48, head_dim=128, seq_len=side, has_attn_mask=regional > 0
        )
    return formula + score + peak_dequant_transient_bytes(transformer, dtype)


class TestLegacyReservation:
    @pytest.mark.parametrize("attention", [ATTENTION["cuda"], ATTENTION["hip"]], ids=["fused", "materializing"])
    def test_the_reservation_is_the_legacy_formula_byte_for_byte(self, attention) -> None:
        """On the platforms still `legacy`, for resident transformers that hold no float8 weight and LoRAs below half a
        GiB each: the cases where the reservation adds none of the terms the formula lacked."""
        mismatches = []
        for case in _cases():
            expected = _legacy(attention, *case)
            reserved = [_reserved(platform, attention, *case) for platform in LEGACY_PLATFORMS]
            if any(value != expected for value in reserved):
                mismatches.append((case[:5], expected, reserved))
        assert not mismatches, f"{len(mismatches)} cases differ, e.g. (case, legacy, reserved): {mismatches[:3]}"

    def test_the_terms_the_formula_lacked_go_on_top(self) -> None:
        """A transformer the cache may load partially reserves its largest block layer for streaming as well."""
        transformer = _transformer(torch.nn.Linear(64, 32, dtype=torch.bfloat16, device="meta"))
        with patch.object(spec_module, "platform_for", return_value=LEGACY_PLATFORMS[0]):
            resident = _estimate(transformer, resident=True).reservation_bytes
            streaming = _estimate(transformer, resident=False).reservation_bytes

        assert streaming - resident == (64 * 32 + 32) * 2

    @pytest.mark.parametrize(
        ("text", "lora_bytes", "terms", "formula"),
        [
            # A regional prompt's concatenated 1500-token caption: the line at 0.5 MiB per token and the 1.5 GiB base,
            # plus 0.34 MiB per caption token beyond 512, against the formula's line alone (4.56 against 4.23 GiB).
            (
                1500,
                0,
                (4096 + 1500) * 2**19 + 3 * 2**29 + (1500 - 512) * int(0.34 * MB),
                (4096 + 1500) * 2**19 + 3 * 2**29,
            ),
            # One 1.2 GiB plain LoRA at its file's size, against the formula's half a GiB (4.95 against 4.25 GiB).
            (
                512,
                int(1.2 * GB),
                (4096 + 512) * 2**19 + 3 * 2**29 + int(1.2 * GB),
                (4096 + 512) * 2**19 + 3 * 2**29 + GB // 2,
            ),
        ],
        ids=["long-caption", "large-lora"],
    )
    def test_terms_beyond_the_legacy_formula_are_reserved_in_full(self, text, lora_bytes, terms, formula) -> None:
        """The formula is a floor, not a cap: where the terms price more than it, at 4096 image tokens without CFG, the
        reservation is theirs."""
        loras = [(ModelPatchRaw(layers={}), lora_bytes)] if lora_bytes else []
        with patch.object(spec_module, "platform_for", return_value=Platform("miopen", "native", "linux")):
            estimate = _estimate(image=4096, text=text, loras=loras)

        assert formula == _legacy_estimate_working_memory(
            image_seq_len=4096,
            positive_text_seq_len=text,
            negative_text_seq_len=None,
            do_cfg=False,
            num_loras=len(loras),
        )
        assert estimate.reservation_bytes == terms > formula


EVERY_PLATFORM = [
    Platform(*keys) for keys in itertools.product(get_args(ConvBackend), get_args(Allocator), get_args(OperatingSystem))
]


class TestUnmeasuredPlatforms:
    """`priced_as` prices a platform nothing is measured on with a measured platform's lines, whose entries floor
    their values at the legacy constants there, and with them the formula they replaced."""

    @pytest.mark.parametrize(
        "platform",
        [p for p in EVERY_PLATFORM if not is_measured(p)],
        ids=lambda p: f"{p.conv_backend}-{p.allocator}-{p.os}",
    )
    def test_an_unmeasured_platform_reserves_the_legacy_formula_at_least(self, platform) -> None:
        attention = ATTENTION["cuda"]
        short = []
        for case in itertools.islice(_cases(), 0, None, 7):
            expected, reserved = _legacy(attention, *case), _reserved(platform, attention, *case)
            if reserved < expected:
                short.append((case[:5], expected, reserved))
        assert not short, f"{len(short)} cases short, e.g. (case, legacy, reserved): {short[:3]}"
