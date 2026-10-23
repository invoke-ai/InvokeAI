"""The calibration plan: which cases calibrate a model on this GPU, which models share its lines, and when it cannot."""

import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers import AutoencoderKL, AutoencoderKLFlux2

from invokeai.app.services.config import get_config
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.working_memory_calibration.working_memory_calibration_default import (
    DeviceMemory,
    _vae_dtypes,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_plan import (
    DECODE_LINES,
    FLUX2_PIPELINE_VAE_DTYPE,
    FORCE_TILED_BASES,
    SDNQ_PIPELINE_VAE_DTYPE,
    VaeDtypes,
    build_plan,
    case_sides,
)
from invokeai.backend.model_manager.load.model_loaders.flux import (
    Flux2DiffusersModel,
    Flux2SDNQCheckpointModel,
    FluxSDNQDiffusersModel,
)
from invokeai.backend.model_manager.load.model_loaders.z_image import ZImageSDNQCheckpointModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    FluxVariantType,
    ModelFormat,
    ModelType,
    ModelVariantType,
    SubModelType,
)
from invokeai.backend.util import vae_working_memory
from invokeai.backend.util.vae_working_memory import dtype_name
from invokeai.backend.working_memory.fingerprint import DeviceClass
from invokeai.backend.working_memory.platform import Platform
from invokeai.backend.working_memory.spec import import_estimator_modules
from tests.fixtures.working_memory_calibration import GPU, GPU_VAE_DTYPES, calibration_service

GIB = 2**30
CARD = DeviceClass(vendor="nvidia", name="Test GPU 24GB", total_mib=24 * 1024, backend="cuda", arch="sm_89")
CUDNN = Platform("cudnn", "native", "linux")
BYTES_PER_PIXEL = {"fp16": 1000, "bf16": 1000, "fp32": 2000}


def _model(key: str, base: BaseModelType, type: ModelType = ModelType.Main, **fields) -> SimpleNamespace:
    return SimpleNamespace(key=key, name=key.upper(), base=base, type=type, **fields)


def _reservation(estimator: str, height: int, width: int, dtype: str) -> int:
    return BYTES_PER_PIXEL[dtype] * height * width


def _plan(
    target,
    models=(),
    *,
    device_class=CARD,
    unavailable=None,
    precision="auto",
    vae_dtypes=GPU_VAE_DTYPES,
    reservation=_reservation,
    capacity_bytes=None,
):
    return build_plan(
        target,
        [target, *models],
        device_class=device_class if unavailable is None else None,
        unavailable=unavailable,
        platform=CUDNN if unavailable is None else None,
        precision=precision,
        vae_dtypes=vae_dtypes,
        reservation=reservation,
        specs=import_estimator_modules(),
        capacity_bytes=capacity_bytes,
    )


def test_an_sdxl_model_runs_three_sizes_per_vae_precision() -> None:
    """24 GiB: a decode may claim 90% less 1 GiB for the VAE, 22119081574 bytes, and the calibration lock reserves
    1.5x the estimate. At 1000 B/px (fp16) the largest square is 3840px (3904px would need 22.9e9 bytes); a quarter and
    a half of its area are 1920px and 2715px, rounded down to 2688px. At 2000 B/px (fp32) the largest is 2688px."""
    sdxl = _model("sdxl", BaseModelType.StableDiffusionXL)

    plan = _plan(sdxl)

    assert plan.supported and plan.reason is None
    assert plan.device_class is not None and plan.device_class.id == CARD.id
    assert plan.estimators == ["vae.sd"]
    assert [(case.line, case.width, case.height, case.settings) for case in plan.cases] == [
        ("decode_fp16", 1920, 1920, {"vaePrecision": "fp16"}),
        ("decode_fp16", 2688, 2688, {"vaePrecision": "fp16"}),
        ("decode_fp16", 3840, 3840, {"vaePrecision": "fp16"}),
        ("decode_fp32", 1344, 1344, {"vaePrecision": "fp32"}),
        ("decode_fp32", 1856, 1856, {"vaePrecision": "fp32"}),
        ("decode_fp32", 2688, 2688, {"vaePrecision": "fp32"}),
    ]
    assert {(case.mode, case.model_key, case.steps, case.estimator) for case in plan.cases} == {
        ("txt2img", "sdxl", 2, "vae.sd")
    }
    assert len({case.id for case in plan.cases}) == 6
    assert (plan.repetitions.warm_up, plan.repetitions.measured) == (1, 2)
    assert plan.estimated_seconds is not None and plan.estimated_seconds > 0
    assert any("encode" in note for note in plan.not_covered)


@pytest.mark.parametrize(
    "base, precision, estimator, lines",
    [
        (BaseModelType.StableDiffusion1, "auto", "vae.sd", ["decode_fp16", "decode_fp32"]),
        (BaseModelType.StableDiffusion3, "auto", "vae.sd3", ["decode_fp16"]),
        (BaseModelType.StableDiffusion3, "bfloat16", "vae.sd3", ["decode_bf16"]),
        (BaseModelType.Flux, "auto", "vae.flux", ["decode_bf16"]),
        (BaseModelType.ZImage, "auto", "vae.flux", ["decode_bf16"]),
        (BaseModelType.Flux2, "auto", "vae.flux2", ["decode_bf16"]),
        (BaseModelType.CogView4, "auto", "vae.cogview4", ["decode_bf16"]),
    ],
)
def test_each_covered_base_plans_the_lines_its_decode_runs(base, precision, estimator, lines) -> None:
    vae_base = BaseModelType.Flux if base == BaseModelType.ZImage else base
    plan = _plan(_model("main", base), [_model("vae", vae_base, ModelType.VAE)], precision=precision)

    assert plan.supported and plan.estimators == [estimator]
    assert sorted({case.line for case in plan.cases}) == lines
    # Only the SD nodes cast the VAE; elsewhere the loader decides the dtype.
    assert all(bool(case.settings) == (base == BaseModelType.StableDiffusion1) for case in plan.cases)


FP32_VAES = VaeDtypes(standalone_flux="fp32", z_image_pipeline="bf16", sdnq_quantized_vae=lambda config: False)
"""`precision: float32`: the standalone FLUX VAE loaders keep it; a Z-Image pipeline's own VAE still loads in bf16."""


def _shipped_untiled_decode(estimator: str, height: int, width: int, dtype: str) -> int:
    return vae_working_memory.estimate_untiled_vae_decode(estimator, height, width, dtype, GPU).reservation_bytes


@pytest.mark.parametrize(
    "target, models",
    [
        (_model("flux", BaseModelType.Flux), [_model("ae", BaseModelType.Flux, ModelType.VAE)]),
        (
            _model("flux2-vae", BaseModelType.Flux2, ModelType.VAE),
            [_model("klein", BaseModelType.Flux2, format="diffusers")],
        ),
    ],
    ids=["flux-main", "flux2-vae"],
)
def test_a_flux_vae_kept_in_fp32_plans_the_fp32_line_at_the_sizes_fp32_decodes_untiled(target, models) -> None:
    """With the shipped cuDNN/native values (no materialized attention scores) on a 24 GiB card, the largest untiled
    bf16 decode is 1792px and the fp32 one 1344px. A configured float32 loads a standalone FLUX VAE in fp32, so the plan
    must size for that: planned at bf16, its 1792px case would cross the pre-tiling threshold and measure nothing."""
    with patch.object(vae_working_memory, "sdpa_score_matrix_bytes", return_value=0):
        at_bf16 = _plan(target, models, reservation=_shipped_untiled_decode)
        at_fp32 = _plan(target, models, vae_dtypes=FP32_VAES, reservation=_shipped_untiled_decode)

    assert [(case.line, case.width) for case in at_bf16.cases] == [
        ("decode_bf16", 896),
        ("decode_bf16", 1216),
        ("decode_bf16", 1792),
    ]
    assert [(case.line, case.width) for case in at_fp32.cases] == [
        ("decode_fp32", 640),
        ("decode_fp32", 896),
        ("decode_fp32", 1344),
    ]


def test_a_configured_fp32_leaves_a_z_image_pipeline_decoding_in_its_own_vaes_dtype() -> None:
    z_image = _model("z-image", BaseModelType.ZImage, format="diffusers")
    flux_vae = _model("ae", BaseModelType.Flux, ModelType.VAE)

    assert {case.line for case in _plan(z_image, [flux_vae], vae_dtypes=FP32_VAES).cases} == {"decode_bf16"}
    # Its standalone VAE, calibrated through it, keeps the configured fp32.
    assert {case.line for case in _plan(flux_vae, [z_image], vae_dtypes=FP32_VAES).cases} == {"decode_fp32"}


_PIPELINE_SUBMODELS = {
    SubModelType.Transformer: None,
    SubModelType.VAE: None,
    SubModelType.TextEncoder: None,
    SubModelType.Tokenizer: None,
}
_FLUX1_PIPELINE_SUBMODELS = _PIPELINE_SUBMODELS | {SubModelType.TextEncoder2: None, SubModelType.Tokenizer2: None}


def _sdnq_pipeline(key: str, base: BaseModelType, **fields) -> SimpleNamespace:
    submodels = _FLUX1_PIPELINE_SUBMODELS if base == BaseModelType.Flux else _PIPELINE_SUBMODELS
    return _model(key, base, format=ModelFormat.SDNQQuantized, submodels=submodels, **fields)


def _lines_of(plan) -> set[tuple[str, tuple[tuple[str, str], ...]]]:
    return {(case.line, tuple(case.components.items())) for case in plan.cases}


@pytest.mark.parametrize(
    "pipeline, line",
    [
        (_model("klein", BaseModelType.Flux2, format="diffusers"), "decode_bf16"),
        (_model("z-image", BaseModelType.ZImage, format="diffusers"), "decode_fp16"),
        (_sdnq_pipeline("klein-sdnq", BaseModelType.Flux2), "decode_fp32"),
        (_sdnq_pipeline("z-image-sdnq", BaseModelType.ZImage), "decode_fp32"),
        (_sdnq_pipeline("flux-sdnq", BaseModelType.Flux), "decode_fp32"),
        (_sdnq_pipeline("flux-sdnq-quantized-vae", BaseModelType.Flux), "decode_bf16"),
    ],
    ids=["flux2-diffusers", "z-image-diffusers", "flux2-sdnq", "z-image-sdnq", "flux-sdnq", "flux-sdnq-quantized-vae"],
)
def test_a_pipeline_decodes_in_the_dtype_its_own_vae_loads_in(pipeline, line) -> None:
    """With an empty `vae` slot a FLUX.2 or Z-Image Diffusers pipeline and a self-contained SDNQ pipeline decode with
    their own VAE: bf16 for FLUX.2 Diffusers, the device's bf16-safe dtype for Z-Image (here fp16), fp32 for the SDNQ
    pipelines, whose loaders give no dtype, and bf16 for an SDNQ-quantized FLUX.1 VAE. A CPU VAE installed beside them
    is no reason to pick one: Generate fills no `vae` slot for a pipeline."""
    vae_base = BaseModelType.Flux2 if pipeline.base == BaseModelType.Flux2 else BaseModelType.Flux
    standalone = [
        _model("a-cpu-vae", vae_base, ModelType.VAE, cpu_only=True),
        _model("b-gpu-vae", vae_base, ModelType.VAE, cpu_only=False),
    ]
    vaes = VaeDtypes(
        standalone_flux="fp32",
        z_image_pipeline="fp16",
        sdnq_quantized_vae=lambda config: config.key == "flux-sdnq-quantized-vae",
    )

    assert _lines_of(_plan(pipeline, standalone, vae_dtypes=vaes)) == {(line, ())}


def test_a_flux_sdnq_pipeline_without_the_t5_pair_decodes_with_a_standalone_vae() -> None:
    """Generate requires FLUX.1's `vae` slot unless the SDNQ pipeline ships every component, T5 included."""
    partial = _model("flux-sdnq", BaseModelType.Flux, format=ModelFormat.SDNQQuantized, submodels=_PIPELINE_SUBMODELS)
    vae = [_model("ae", BaseModelType.Flux, ModelType.VAE)]

    assert {case.line for case in _plan(partial, vae, vae_dtypes=FP32_VAES).cases} == {"decode_fp32"}
    assert {case.line for case in _plan(partial, vae).cases} == {"decode_bf16"}


GGUF_KLEIN_9B = _model("klein-9b-gguf", BaseModelType.Flux2, format="gguf_quantized", variant=Flux2VariantType.Klein9B)


@pytest.mark.parametrize(
    "sources, line",
    [
        # The first installed pipeline whose Qwen3 encoder fits the 9B transformer, wherever it is listed.
        (
            [
                _sdnq_pipeline("a-sdnq-4b", BaseModelType.Flux2, variant=Flux2VariantType.Klein4B),
                _model("b-diffusers-9b", BaseModelType.Flux2, format="diffusers", variant=Flux2VariantType.Klein9B),
            ],
            "decode_bf16",
        ),
        # Without a fitting one, the first pipeline at all.
        ([_sdnq_pipeline("a-sdnq-4b", BaseModelType.Flux2, variant=Flux2VariantType.Klein4B)], "decode_fp32"),
        # Without a pipeline, Generate fills the slot with a standalone VAE.
        ([_model("flux2-vae", BaseModelType.Flux2, ModelType.VAE)], "decode_fp16"),
    ],
    ids=["fitting-diffusers-source", "only-an-sdnq-source", "no-source"],
)
def test_a_flux2_model_without_its_own_vae_decodes_with_its_component_source_vae(sources, line) -> None:
    """Generate gives a GGUF or checkpoint FLUX.2 model an installed FLUX.2 pipeline as its component source
    (`getAutoFlux2ComponentSourceModel`), and the FLUX.2 model loader node decodes with that pipeline's VAE."""
    vaes = VaeDtypes(standalone_flux="fp16", z_image_pipeline="bf16", sdnq_quantized_vae=lambda config: False)

    assert _lines_of(_plan(GGUF_KLEIN_9B, sources, vae_dtypes=vaes)) == {(line, ())}


def test_a_flux2_model_decoding_with_its_component_source_picks_no_vae_beside_a_cpu_one() -> None:
    """Generate fills no `vae` slot while a component source supplies the VAE, so a CPU-only VAE is no reason to pin
    another: the pin would decode with a standalone VAE instead of the source's."""
    models = [
        _model("klein-9b-diffusers", BaseModelType.Flux2, format="diffusers", variant=Flux2VariantType.Klein9B),
        _model("a-cpu-vae", BaseModelType.Flux2, ModelType.VAE, cpu_only=True),
        _model("b-gpu-vae", BaseModelType.Flux2, ModelType.VAE, cpu_only=False),
    ]

    assert _lines_of(_plan(GGUF_KLEIN_9B, models, vae_dtypes=FP32_VAES)) == {("decode_bf16", ())}


def test_a_standalone_flux_vae_is_not_calibrated_through_an_sdnq_pipeline_that_drops_the_pick() -> None:
    """Generate sends no VAE for a self-contained FLUX.1 SDNQ pipeline, which decodes with its own, so a VAE target goes
    through another main model, here one that sorts after it; without one it cannot be calibrated."""
    flux_vae = _model("ae", BaseModelType.Flux, ModelType.VAE)
    sdnq = _sdnq_pipeline("a-flux-sdnq", BaseModelType.Flux)
    flux = _model("b-flux", BaseModelType.Flux)

    plan = _plan(flux_vae, [sdnq, flux], vae_dtypes=FP32_VAES)
    alone = _plan(flux_vae, [sdnq])

    assert (plan.calibrated_through, _lines_of(plan)) == ("b-flux", {("decode_fp32", (("vae", "ae"),))})
    assert not alone.supported and "takes a VAE pick" in (alone.reason or "") and alone.cases == []


CARD_16GB = DeviceClass(vendor="amd", name="AMD Radeon RX 9060 XT", total_mib=16304, backend="hip", arch="gfx1200")
OUTSIDE_MIB = 2628
"""What a 16 GB card measured in use besides the torch allocator while its calibration decoded (driver less torch's
reserve, RX 9060 XT on Windows)."""
# About what the shipped SD estimates price a decode at there: 9058 MiB at 2112px fp16, 8521 MiB at 1216px fp32.
SD_16GB_PER_PIXEL = {"fp16": 2100, "fp32": 6000}


def _sd_16gb(estimator: str, height: int, width: int, dtype: str) -> int:
    return SD_16GB_PER_PIXEL[dtype] * height * width


def test_on_a_16gb_card_the_largest_size_leaves_the_vae_fully_resident_beside_its_reservation() -> None:
    """The cache can give 16304 - 2628 = 13676 MiB, below the 14673 MiB the pretile share allows; less 1 GiB for the
    VAE, the reservation (1.5x the estimate) may take 12652 MiB, 13266583552 bytes. fp16 at 2100 B/px: 2048px needs
    13212057600, 2112px 14050713600. fp32 at 6000 B/px: 1152px needs 11943936000, 1216px 13307904000. The share of the
    card alone would have allowed 2112px and 1216px, the sizes that ran partly resident."""
    plan = _plan(
        _model("sdxl", BaseModelType.StableDiffusionXL),
        device_class=CARD_16GB,
        reservation=_sd_16gb,
        capacity_bytes=(CARD_16GB.total_mib - OUTSIDE_MIB) * 2**20,
    )

    assert [(case.line, case.width) for case in plan.cases] == [
        ("decode_fp16", 1024),
        ("decode_fp16", 1408),
        ("decode_fp16", 2048),
        ("decode_fp32", 576),
        ("decode_fp32", 768),
        ("decode_fp32", 1152),
    ]


@pytest.mark.parametrize(
    "process_mib, reserve_gb, cap_gb",
    [(16304 - OUTSIDE_MIB, 0.0, None), (16304, OUTSIDE_MIB / 1024, None), (16304, 0.0, (16304 - OUTSIDE_MIB) / 1024)],
    ids=["outside-use", "reserve-vram", "cache-cap"],
)
def test_the_plan_sizes_for_what_this_process_holds_less_the_cache_reserves(
    tmp_path: Path, mock_invoker: Invoker, process_mib: int, reserve_gb: float, cap_gb: float | None
) -> None:
    sdxl = _model("sdxl", BaseModelType.StableDiffusionXL)
    model_manager = MagicMock()
    model_manager.store.get_model.return_value = sdxl
    model_manager.store.all_models.return_value = [sdxl]
    mock_invoker.services.model_manager = model_manager
    mock_invoker.services.configuration.reserve_vram_gb = reserve_gb
    mock_invoker.services.configuration.max_cache_vram_gb = cap_gb
    service = calibration_service(
        tmp_path,
        device_class=CARD_16GB,
        untiled_decode_reservation=lambda estimator, height, width, dtype, device: _sd_16gb(
            estimator, height, width, dtype
        ),
        device_memory=lambda device: DeviceMemory(process_mib * 2**20, 0),
    )
    service.start(mock_invoker)
    try:
        plan = service.get_plan("sdxl")
    finally:
        service.stop(mock_invoker)

    assert max(case.width for case in plan.cases if case.line == "decode_fp16") == 2048
    assert max(case.width for case in plan.cases if case.line == "decode_fp32") == 1152


def test_the_largest_size_is_capped_at_what_generate_accepts() -> None:
    assert case_sides(lambda side: side * side, 10**15) == [2048, 2880, 4096]


def test_a_card_that_cannot_hold_a_small_decode_with_headroom_is_unsupported() -> None:
    small = DeviceClass(vendor="nvidia", name="Test GPU 2GB", total_mib=2048, backend="cuda", arch="sm_75")

    plan = _plan(
        _model("flux", BaseModelType.Flux), [_model("ae", BaseModelType.Flux, ModelType.VAE)], device_class=small
    )

    assert not plan.supported and "Test GPU 2GB cannot hold" in (plan.reason or "") and plan.cases == []


def test_shared_with_lists_the_models_whose_decode_prices_with_the_same_lines() -> None:
    """On cuDNN the SD, SD3 and FLUX.1 estimators all price with the SD3 entry's values: an SDXL calibration (fp16 and
    fp32) moves SD1 models and an SD3 model running fp16, but not FLUX.1, which decodes in bf16."""
    target = _model("sdxl", BaseModelType.StableDiffusionXL)
    models = [
        _model("sd1", BaseModelType.StableDiffusion1),
        _model("sd3", BaseModelType.StableDiffusion3),
        _model("sdxl-vae", BaseModelType.StableDiffusionXL, ModelType.VAE),
        _model("sdxl-vae-cpu", BaseModelType.StableDiffusionXL, ModelType.VAE, cpu_only=True),
        _model("flux", BaseModelType.Flux),
        _model("lora", BaseModelType.StableDiffusionXL, ModelType.LoRA),
        _model("wan", BaseModelType.Wan),
    ]

    plan = _plan(target, models)

    assert [(shared.key, shared.lines) for shared in plan.shared_with] == [
        ("sd1", ["decode_fp16", "decode_fp32"]),
        ("sd3", ["decode_fp16"]),
        ("sdxl-vae", ["decode_fp16", "decode_fp32"]),
    ]


def test_a_standalone_vae_is_calibrated_through_a_compatible_main_model() -> None:
    flux_vae = _model("ae", BaseModelType.Flux, ModelType.VAE)
    z_image = _model("z-image", BaseModelType.ZImage)
    sdxl_vae = _model("sdxl-vae", BaseModelType.StableDiffusionXL, ModelType.VAE)
    sdxl = _model("sdxl", BaseModelType.StableDiffusionXL)

    flux_plan = _plan(flux_vae, [z_image, sdxl])
    sd_plan = _plan(sdxl_vae, [z_image, sdxl])

    # Z-Image decodes with a FLUX.1 VAE, through its `vae` slot.
    assert (flux_plan.supported, flux_plan.calibrated_through, flux_plan.estimators) == (True, "z-image", ["vae.flux"])
    assert {(case.model_key, tuple(case.components.items())) for case in flux_plan.cases} == {
        ("z-image", (("vae", "ae"),))
    }
    # The SD bases pick a VAE through a setting a case cannot set: the main model's own VAE runs the same lines.
    assert (sd_plan.supported, sd_plan.calibrated_through) == (True, "sdxl")
    assert {(case.model_key, tuple(case.components.items())) for case in sd_plan.cases} == {("sdxl", ())}


def test_a_main_model_picks_a_gpu_vae_where_generate_could_fill_in_one_on_the_cpu() -> None:
    flux = _model("flux", BaseModelType.Flux)
    cpu = _model("a-cpu-vae", BaseModelType.Flux, ModelType.VAE, cpu_only=True)
    gpu = _model("b-gpu-vae", BaseModelType.Flux, ModelType.VAE, cpu_only=False)

    assert {tuple(case.components.items()) for case in _plan(flux, [cpu, gpu]).cases} == {(("vae", "b-gpu-vae"),)}
    assert {tuple(case.components.items()) for case in _plan(flux, [gpu]).cases} == {()}


@pytest.mark.parametrize(
    "target, models, reason",
    [
        (_model("ae", BaseModelType.Flux, ModelType.VAE, cpu_only=True), [_model("f", BaseModelType.Flux)], "CPU"),
        (
            _model("ae", BaseModelType.Flux, ModelType.VAE),
            [_model("s", BaseModelType.StableDiffusionXL)],
            "no compatible",
        ),
        (_model("lora", BaseModelType.Flux, ModelType.LoRA), [], "this is a flux lora model"),
        (_model("qwen", BaseModelType.QwenImage), [], "this is a qwen-image main model"),
        (_model("refiner", BaseModelType.StableDiffusionXLRefiner), [], "plain text prompt"),
        (_model("ideogram", BaseModelType.Ideogram4), [], "plain text prompt"),
        (_model("fill", BaseModelType.Flux, variant=FluxVariantType.DevFill), [], "dev_fill variant"),
        (_model("inpaint", BaseModelType.StableDiffusion1, variant=ModelVariantType.Inpaint), [], "inpaint variant"),
    ],
    ids=["cpu-only-vae", "vae-without-main", "lora", "uncovered-base", "refiner", "ideogram", "fill", "inpaint"],
)
def test_models_a_text_to_image_decode_cannot_calibrate_say_why(target, models, reason) -> None:
    plan = _plan(target, models)

    assert not plan.supported and reason in (plan.reason or "")
    assert plan.cases == [] and plan.shared_with == [] and plan.estimated_seconds is None


_DECODE_NODES = {
    BaseModelType.StableDiffusion1: "latents_to_image",
    BaseModelType.StableDiffusion2: "latents_to_image",
    BaseModelType.StableDiffusionXL: "latents_to_image",
    BaseModelType.StableDiffusionXLRefiner: "latents_to_image",
    BaseModelType.StableDiffusion3: "sd3_latents_to_image",
    BaseModelType.Flux: "flux_vae_decode",
    BaseModelType.ZImage: "z_image_latents_to_image",
    BaseModelType.Flux2: "flux2_vae_decode",
    BaseModelType.Ideogram4: "ideogram4_latents_to_image",
    BaseModelType.CogView4: "cogview4_latents_to_image",
}
"""The node each base's text-to-image graph decodes with (`invokeai/app/invocations/vae/`)."""


@pytest.mark.parametrize("base", sorted(DECODE_LINES, key=str))
def test_each_base_plans_the_estimator_its_decode_node_prices_with(base: BaseModelType) -> None:
    """`DECODE_LINES` restates what the decode nodes do; a node that switches its estimator must fail here."""
    assert set(DECODE_LINES) == set(_DECODE_NODES)
    node = importlib.import_module(f"invokeai.app.invocations.vae.{_DECODE_NODES[base]}")
    [estimate] = [
        value
        for name, value in vars(node).items()
        if name.startswith("estimate_vae_working_memory_") and value is getattr(vae_working_memory, name, None)
    ]
    defaults = {"tile_size": None, "fp32": False, "device": GPU}
    kwargs = {name: defaults[name] for name in inspect.signature(estimate).parameters if name in defaults}
    vae = torch.nn.Linear(1, 1).to(torch.bfloat16)

    with patch.object(vae_working_memory, "sdpa_score_matrix_bytes", return_value=0):
        priced = estimate("decode", torch.zeros(1, 16, 64, 64), vae, **kwargs)

    assert priced.estimator == DECODE_LINES[base].estimator


def test_the_sd_decode_precisions_are_the_dtypes_the_sd_nodes_cast_to() -> None:
    names = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
    cast = {names[vae_working_memory.sd_vae_run_dtype("decode", fp32, GPU)] for fp32 in (False, True)}
    assert set(DECODE_LINES[BaseModelType.StableDiffusionXL].dtypes) == cast


@pytest.mark.parametrize(
    "devices, reason",
    [([GPU, torch.device("cuda:1")], "single generation device"), ([torch.device("cpu")], "GPU memory")],
    ids=["two-gpus", "no-gpu"],
)
def test_a_machine_that_cannot_calibrate_says_why_in_the_plan(
    tmp_path: Path, mock_invoker: Invoker, devices, reason
) -> None:
    flux = _model("flux", BaseModelType.Flux)
    model_manager = MagicMock()
    model_manager.store.get_model.return_value = flux
    model_manager.store.all_models.return_value = [flux]
    mock_invoker.services.model_manager = model_manager
    service = calibration_service(tmp_path, devices=lambda: devices)
    service.start(mock_invoker)
    try:
        plan = service.get_plan("flux")
    finally:
        service.stop(mock_invoker)

    assert not plan.supported and reason in (plan.reason or "")
    assert plan.device_class is None and plan.estimators == ["vae.flux"] and plan.cases == []


# `force_tiled_decode` and a decode with no VAE on the GPU: runs that would measure no decode line.


def _service_plan(tmp_path: Path, mock_invoker: Invoker, target, models=(), **config):
    model_manager = MagicMock()
    model_manager.store.get_model.return_value = target
    model_manager.store.all_models.return_value = [target, *models]
    mock_invoker.services.model_manager = model_manager
    for name, value in config.items():
        setattr(mock_invoker.services.configuration, name, value)
    service = calibration_service(
        tmp_path, untiled_decode_reservation=lambda estimator, h, w, dtype, device: 1000 * h * w
    )
    service.start(mock_invoker)
    try:
        return service.get_plan(target.key)
    finally:
        service.stop(mock_invoker)


@pytest.mark.parametrize(
    "target, models",
    [
        (_model("sdxl", BaseModelType.StableDiffusionXL), []),
        (_model("flux", BaseModelType.Flux), [_model("ae", BaseModelType.Flux, ModelType.VAE)]),
        (_model("z-image", BaseModelType.ZImage, format="diffusers"), []),
        (_model("klein", BaseModelType.Flux2, format="diffusers"), []),
        (_model("ae", BaseModelType.Flux, ModelType.VAE), [_model("flux", BaseModelType.Flux)]),
    ],
    ids=["sdxl", "flux", "z-image", "flux2", "flux-vae"],
)
def test_force_tiled_decode_leaves_nothing_to_calibrate_and_says_how_to_change_that(
    tmp_path: Path, mock_invoker: Invoker, target, models
) -> None:
    """With `force_tiled_decode` every generation decodes tiled, so a case measures no untiled line, and no estimate
    would read one."""
    plan = _service_plan(tmp_path, mock_invoker, target, models, force_tiled_decode=True)

    assert not plan.supported and plan.cases == [] and plan.device_class is not None
    assert "force_tiled_decode" in (plan.reason or "")


@pytest.mark.parametrize(
    "target", [_model("sd3", BaseModelType.StableDiffusion3), _model("cogview4", BaseModelType.CogView4)]
)
def test_force_tiled_decode_leaves_the_decodes_that_ignore_it_calibratable(
    tmp_path: Path, mock_invoker: Invoker, target
) -> None:
    assert _service_plan(tmp_path, mock_invoker, target, force_tiled_decode=True).supported


@pytest.mark.parametrize("base", sorted(DECODE_LINES, key=str))
def test_the_bases_force_tiled_decode_stops_are_those_whose_decode_node_reads_it(base: BaseModelType) -> None:
    node = importlib.import_module(f"invokeai.app.invocations.vae.{_DECODE_NODES[base]}")
    assert ("force_tiled_decode" in inspect.getsource(node)) == (base in FORCE_TILED_BASES)


@pytest.mark.parametrize(
    "main, vae_base",
    [
        (_model("flux", BaseModelType.Flux), BaseModelType.Flux),
        (_model("z-image-gguf", BaseModelType.ZImage, format="gguf_quantized"), BaseModelType.Flux),
        (_model("klein-gguf", BaseModelType.Flux2, format="gguf_quantized"), BaseModelType.Flux2),
    ],
    ids=["flux", "z-image", "flux2"],
)
def test_a_model_that_needs_a_standalone_vae_and_has_none_on_the_gpu_cannot_be_calibrated(main, vae_base) -> None:
    """The decode would run on the CPU, where no calibration window opens; without any VAE the graph cannot compile."""
    cpu_vae = _model("cpu-vae", vae_base, ModelType.VAE, cpu_only=True)

    for models in ([cpu_vae], []):
        plan = _plan(main, models)
        assert not plan.supported and plan.cases == [] and "GPU" in (plan.reason or "")
    assert _plan(main, [cpu_vae, _model("gpu-vae", vae_base, ModelType.VAE)]).supported


def test_a_pipeline_with_its_own_vae_needs_no_gpu_vae_beside_it() -> None:
    cpu_vae = _model("cpu-vae", BaseModelType.Flux2, ModelType.VAE, cpu_only=True)

    assert _plan(_model("klein", BaseModelType.Flux2, format="diffusers"), [cpu_vae]).supported
    # Nor does a FLUX.2 model decoding with its component source's VAE.
    assert _plan(GGUF_KLEIN_9B, [cpu_vae, _model("klein-9b", BaseModelType.Flux2, format="diffusers")]).supported


# The pipeline VAE dtypes the plan restates, held to what their loaders yield (`own_vae_dtype`).


def _save_tiny_vae(folder: Path, vae_class: type, dtype: torch.dtype) -> None:
    vae = vae_class(
        block_out_channels=(32,),
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        norm_num_groups=8,
        latent_channels=4,
        layers_per_block=1,
    )
    vae.to(dtype).save_pretrained(folder)


def _dtype_of(model: torch.nn.Module) -> str:
    return dtype_name(next(model.parameters()).dtype)


@pytest.mark.parametrize(
    "loader_class, vae_class",
    [(Flux2SDNQCheckpointModel, AutoencoderKLFlux2), (ZImageSDNQCheckpointModel, AutoencoderKL)],
    ids=["flux2", "z-image"],
)
def test_an_sdnq_pipelines_own_vae_loads_in_the_dtype_the_plan_sizes_for(tmp_path: Path, loader_class, vae_class):
    _save_tiny_vae(tmp_path / "vae", vae_class, torch.bfloat16)
    loader = loader_class.__new__(loader_class)

    vae = loader._load_vae(SimpleNamespace(path=str(tmp_path), submodels=None))

    assert _dtype_of(vae) == SDNQ_PIPELINE_VAE_DTYPE


def test_a_flux_sdnq_pipelines_plain_vae_loads_in_the_dtype_the_plan_sizes_for(tmp_path: Path):
    _save_tiny_vae(tmp_path / "vae", AutoencoderKL, torch.bfloat16)
    loader = FluxSDNQDiffusersModel.__new__(FluxSDNQDiffusersModel)

    assert _dtype_of(loader._load_vae(tmp_path / "vae")) == SDNQ_PIPELINE_VAE_DTYPE


def test_a_flux2_diffusers_pipelines_own_vae_loads_in_the_dtype_the_plan_sizes_for(tmp_path: Path):
    _save_tiny_vae(tmp_path / "vae", AutoencoderKLFlux2, torch.float32)
    (tmp_path / "model_index.json").write_text(json.dumps({"vae": ["diffusers", "AutoencoderKLFlux2"]}))
    loader = Flux2DiffusersModel.__new__(Flux2DiffusersModel)
    loader._apply_fp8_layerwise_casting = lambda model, config, submodel_type: model

    vae = loader._load_model(SimpleNamespace(path=str(tmp_path), repo_variant=None), SubModelType.VAE)

    assert _dtype_of(vae) == FLUX2_PIPELINE_VAE_DTYPE


@pytest.mark.parametrize("precision, expected", [("float32", "fp32"), ("bfloat16", "bf16"), ("auto", "bf16")])
def test_the_service_reads_a_standalone_flux_vaes_dtype_from_the_precision_setting(
    monkeypatch: pytest.MonkeyPatch, precision: str, expected: str
):
    """On a CUDA card `auto` is fp16, which the FLUX loaders turn into bf16 where the card has it."""
    monkeypatch.setattr(get_config(), "precision", precision)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GeForce RTX 4090")
    empty = torch.empty
    # The bf16 probe on a card that has it; no CUDA device is needed for the answer.
    monkeypatch.setattr(torch, "empty", lambda *args, dtype, device: empty(0, dtype=dtype))

    assert _vae_dtypes(GPU, Path("models")).standalone_flux == expected


def test_the_service_tells_an_sdnq_quantized_vae_folder_where_the_loader_looks(tmp_path: Path):
    quantized, plain = tmp_path / "flux-sdnq" / "vae", tmp_path / "flux-sdnq-plain" / "vae"
    quantized.mkdir(parents=True)
    plain.mkdir(parents=True)
    (quantized / "quantization_config.json").write_text(json.dumps({"quant_method": "sdnq"}))
    vaes = _vae_dtypes(torch.device("cpu"), tmp_path)

    assert vaes.sdnq_quantized_vae(SimpleNamespace(path="flux-sdnq", submodels=None))
    assert not vaes.sdnq_quantized_vae(SimpleNamespace(path="flux-sdnq-plain", submodels=None))
