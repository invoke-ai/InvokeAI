"""The calibration plan: which cases calibrate a model on this GPU, which models share its lines, and when it cannot."""

import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psutil
import pytest
import torch
from diffusers import AutoencoderKL, AutoencoderKLFlux2

from invokeai.app.services.config import get_config
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.working_memory_calibration.working_memory_calibration_default import _vae_dtypes
from invokeai.app.services.working_memory_calibration.working_memory_calibration_plan import (
    DECODE_LINES,
    FLUX2_PIPELINE_VAE_DTYPE,
    FORCE_TILED_BASES,
    SDNQ_PIPELINE_VAE_DTYPE,
    VAE_WEIGHTS_ALLOWANCE_BYTES,
    VaeDtypes,
    build_plan,
    case_sides,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
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
    Qwen3VariantType,
    SubModelType,
)
from invokeai.backend.util import devices, vae_working_memory
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.vae_working_memory import dtype_name
from invokeai.backend.working_memory import counters
from invokeai.backend.working_memory.fingerprint import DeviceClass
from invokeai.backend.working_memory.platform import Platform
from invokeai.backend.working_memory.probe import calibration_reservation
from invokeai.backend.working_memory.spec import import_estimator_modules
from tests.fixtures.working_memory_calibration import GPU, GPU_VAE_DTYPES, calibration_service

GIB = 2**30
MIB = 2**20
CARD = DeviceClass(vendor="nvidia", name="Test GPU 24GB", total_mib=24 * 1024, backend="cuda", arch="sm_89")
CUDNN = Platform("cudnn", "native", "linux")
BYTES_PER_PIXEL = {"fp16": 1000, "bf16": 1000, "fp32": 2000}


def _model(key: str, base: BaseModelType, type: ModelType = ModelType.Main, **fields) -> SimpleNamespace:
    return SimpleNamespace(key=key, name=key.upper(), base=base, type=type, **fields)


def _reservation(estimator: str, height: int, width: int, dtype: str) -> int:
    return BYTES_PER_PIXEL[dtype] * height * width


QWEN3_8B = _model("qwen3-8b", BaseModelType.Any, ModelType.Qwen3Encoder, variant=Qwen3VariantType.Qwen3_8B)
"""The standalone text encoder Generate fills in for a FLUX.2 Klein 9B model, or one without a variant, that is no
pipeline: without one, or a pipeline whose encoder fits, its graph does not compile."""


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
    plan = _plan(_model("main", base), [_model("vae", vae_base, ModelType.VAE), QWEN3_8B], precision=precision)

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
        # Without a fitting one, the first pipeline at all, which gives the VAE beside the standalone encoder.
        ([_sdnq_pipeline("a-sdnq-4b", BaseModelType.Flux2, variant=Flux2VariantType.Klein4B)], "decode_fp32"),
        # Without a pipeline, Generate fills the slot with a standalone VAE.
        ([_model("flux2-vae", BaseModelType.Flux2, ModelType.VAE)], "decode_fp16"),
    ],
    ids=["fitting-diffusers-source", "only-an-sdnq-source", "no-source"],
)
def test_a_flux2_model_without_its_own_vae_decodes_with_its_component_source_vae(sources, line) -> None:
    """Generate gives a GGUF or checkpoint FLUX.2 model an installed FLUX.2 pipeline as its component source
    (`getAutoFlux2ComponentSourceModel`), and the FLUX.2 model loader node decodes with that pipeline's VAE. A standalone
    Qwen3 8B encoder is installed, as one that fits no pipeline needs."""
    vaes = VaeDtypes(standalone_flux="fp16", z_image_pipeline="bf16", sdnq_quantized_vae=lambda config: False)

    assert _lines_of(_plan(GGUF_KLEIN_9B, [*sources, QWEN3_8B], vae_dtypes=vaes)) == {(line, ())}


KLEIN_4B_SDNQ = _sdnq_pipeline("klein-4b-sdnq", BaseModelType.Flux2, variant=Flux2VariantType.Klein4B)
FLUX2_VAE = _model("flux2-vae", BaseModelType.Flux2, ModelType.VAE)
QWEN3_4B = _model("qwen3-4b", BaseModelType.Any, ModelType.Qwen3Encoder, variant=Qwen3VariantType.Qwen3_4B)
GGUF_DEV = _model("dev-gguf", BaseModelType.Flux2, format="gguf_quantized", variant=Flux2VariantType.Dev)


@pytest.mark.parametrize(
    "target, models, encoder",
    [
        (GGUF_KLEIN_9B, [KLEIN_4B_SDNQ], "Qwen3 8B"),
        (GGUF_KLEIN_9B, [KLEIN_4B_SDNQ, FLUX2_VAE], "Qwen3 8B"),
        (GGUF_KLEIN_9B, [FLUX2_VAE], "Qwen3 8B"),
        # The 4B encoder fits the 4B transformer only.
        (GGUF_KLEIN_9B, [KLEIN_4B_SDNQ, QWEN3_4B], "Qwen3 8B"),
        (GGUF_DEV, [KLEIN_4B_SDNQ, FLUX2_VAE, QWEN3_8B], "Mistral"),
        # A VAE goes through the only main model that takes its pick, which cannot generate.
        (FLUX2_VAE, [GGUF_KLEIN_9B], "Qwen3 8B"),
    ],
    ids=["unfitting-source", "unfitting-source-and-vae", "vae-alone", "encoder-of-another-size", "dev", "vae-target"],
)
def test_a_flux2_model_without_a_text_encoder_generate_can_use_cannot_be_calibrated(target, models, encoder) -> None:
    """A FLUX.2 model that is no pipeline compiles only with a text encoder: a fitting pipeline's, or a standalone one of
    its size. A pipeline that does not fit gives only its VAE, so beside it alone, or beside a standalone VAE, no case
    compiles: `pnpm graphs:export` rejects each with "Generate needs a Qwen3 Encoder" ("a Mistral Encoder" for [dev])."""
    plan = _plan(target, models)

    assert not plan.supported and plan.cases == [] and plan.device_class is not None
    assert "text encoder" in (plan.reason or "") and f"{encoder} encoder" in (plan.reason or "")


def test_a_flux2_vae_is_calibrated_through_a_main_model_whose_graph_compiles() -> None:
    """The GGUF model sorts first but has no encoder; the 4B pipeline brings its own and takes the `vae` pick."""
    gguf = _model("a-klein-9b-gguf", BaseModelType.Flux2, format="gguf_quantized", variant=Flux2VariantType.Klein9B)
    pipeline = _sdnq_pipeline("b-klein-4b-sdnq", BaseModelType.Flux2, variant=Flux2VariantType.Klein4B)

    plan = _plan(FLUX2_VAE, [gguf, pipeline])

    assert (plan.supported, plan.calibrated_through) == (True, "b-klein-4b-sdnq")
    assert _lines_of(plan) == {("decode_bf16", (("vae", "flux2-vae"),))}


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


def _cache(device: str, *, reserve_gb: float = 0.0, cap_gb: float | None = None) -> ModelCache:
    """A model cache whose execution device is `device`, built on the CPU: every memory query is patched out."""
    cache = ModelCache(
        execution_device_working_mem_gb=1.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        max_vram_cache_size_gb=cap_gb,
        reserve_vram_gb=reserve_gb,
        execution_device="cpu",
        storage_device="cpu",
        logger=MagicMock(),
    )
    cache._execution_device = torch.device(device)
    return cache


def _plan_with_cache(tmp_path: Path, mock_invoker: Invoker, device: torch.device, device_class: DeviceClass, cache):
    sdxl = _model("sdxl", BaseModelType.StableDiffusionXL)
    model_manager = MagicMock()
    model_manager.store.get_model.return_value = sdxl
    model_manager.store.all_models.return_value = [sdxl]
    model_manager.load.ram_caches = {str(device): cache}
    mock_invoker.services.model_manager = model_manager
    service = calibration_service(
        tmp_path,
        devices=lambda: [device],
        device_class=device_class,
        untiled_decode_reservation=lambda estimator, height, width, dtype, device: _sd_16gb(
            estimator, height, width, dtype
        ),
        device_memory=None,
    )
    service.start(mock_invoker)
    try:
        return service.get_plan("sdxl")
    finally:
        service.stop(mock_invoker)


@pytest.mark.parametrize(
    "process_mib, reserve_gb, cap_gb",
    [(16304 - OUTSIDE_MIB, 0.0, None), (16304, OUTSIDE_MIB / 1024, None), (16304, 0.0, (16304 - OUTSIDE_MIB) / 1024)],
    ids=["outside-use", "reserve-vram", "cache-cap"],
)
def test_the_plan_sizes_for_what_the_cache_gives_a_calibration_lock(
    tmp_path: Path, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch, process_mib: int, reserve_gb, cap_gb
) -> None:
    """A dedicated card: the calibration's offload moves every other model out, so what they hold counts; the cache's
    reserve and cap apply."""
    monkeypatch.setattr(
        TorchDevice, "cuda_mem_get_info", classmethod(lambda cls, device: ((process_mib - 2048) * MIB, 16304 * MIB))
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 2048 * MIB)
    cache = _cache("cuda:0", reserve_gb=reserve_gb, cap_gb=cap_gb)

    plan = _plan_with_cache(tmp_path, mock_invoker, GPU, CARD_16GB, cache)

    assert max(case.width for case in plan.cases if case.line == "decode_fp16") == 2048
    assert max(case.width for case in plan.cases if case.line == "decode_fp32") == 1152


def test_a_working_memory_floor_beyond_what_the_cache_gives_leaves_nothing_to_plan(
    tmp_path: Path, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock never reserves less than `device_working_mem_gb`. Set beyond what the cache can give a calibration on
    the card (the 16 GB card of the test above, 2048px at the default), even the smallest case's lock would leave the
    VAE partly loaded: the plan says the card cannot calibrate rather than queue cases whose windows fit no line."""
    monkeypatch.setattr(TorchDevice, "cuda_mem_get_info", classmethod(lambda cls, device: (14256 * MIB, 16304 * MIB)))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 2048 * MIB)
    monkeypatch.setattr(mock_invoker.services.configuration, "device_working_mem_gb", 15.0)

    plan = _plan_with_cache(tmp_path, mock_invoker, GPU, CARD_16GB, _cache("cuda:0"))

    assert plan.cases == []
    assert plan.supported is False


@pytest.mark.parametrize("device", ["mps", "xpu:0"], ids=["mps", "integrated-xpu"])
def test_a_device_that_shares_system_memory_sizes_for_the_ram_the_host_has_available(
    tmp_path: Path, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """32 GiB of unified memory: the host has 10 GiB available, the backend holds 2 GiB for this process, and models
    and tensors take 3 GiB of that now. The calibration's offload moves nothing on such a device, so the lock gets 9
    GiB, as the cache counts it, and a case may take 8 GiB beside the VAE's 1 GiB: at 2100 B/px (fp16) the largest
    square is 1600px, at 6000 B/px (fp32) 960px. Without the live allocations taken off: 1920px and 1088px; with 90%
    of the total, what the plan read first: 3072px and 1792px."""
    gpu = torch.device(device)
    shared = DeviceClass(vendor="test", name="Unified 32GB", total_mib=32 * 1024, backend=gpu.type, arch=None)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=32 * GIB, available=10 * GIB))
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 2 * GIB)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 3 * GIB)
    monkeypatch.setattr(torch.xpu, "memory_reserved", lambda device=None: 2 * GIB, raising=False)
    monkeypatch.setattr(torch.xpu, "memory_allocated", lambda device=None: 3 * GIB, raising=False)
    monkeypatch.setattr(torch.xpu, "memory_stats", lambda device=None: {}, raising=False)
    monkeypatch.setattr(counters, "xpu_memory_info", lambda device: None)
    monkeypatch.setattr(devices, "xpu_device_is_integrated", lambda device: True)
    cache = _cache(device)

    plan = _plan_with_cache(tmp_path, mock_invoker, gpu, shared, cache)

    assert cache.calibration_capacity() == 9 * GIB
    assert max(case.width for case in plan.cases if case.line == "decode_fp16") == 1600
    assert max(case.width for case in plan.cases if case.line == "decode_fp32") == 960
    # Beside each case's working memory the lock still has room for the VAE, as the cache budgets it.
    for case in plan.cases:
        reservation = calibration_reservation(_sd_16gb(case.estimator, case.height, case.width, case.line[7:]))
        assert cache._get_vram_available(reservation) >= VAE_WEIGHTS_ALLOWANCE_BYTES


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


def _with_default_vae(model: SimpleNamespace, vae_key: str) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(model), "default_settings": SimpleNamespace(vae=vae_key)})


SDXL_CPU_VAE = _model("sdxl-cpu-vae", BaseModelType.StableDiffusionXL, ModelType.VAE, cpu_only=True)


def test_an_sd_model_whose_default_vae_runs_on_the_cpu_cannot_be_calibrated() -> None:
    """Selecting the model applies its default VAE (`getModelDefaultVae`), and the SD graph decodes with it; an SD case
    has no `vae` slot to pick another. A default the model does not accept is not applied, so its own VAE decodes."""
    sdxl = _with_default_vae(_model("sdxl", BaseModelType.StableDiffusionXL), "sdxl-cpu-vae")
    gpu_vae = _model("sdxl-gpu-vae", BaseModelType.StableDiffusionXL, ModelType.VAE)
    sd1_cpu_vae = _model("sd1-cpu-vae", BaseModelType.StableDiffusion1, ModelType.VAE, cpu_only=True)

    plan = _plan(sdxl, [SDXL_CPU_VAE, gpu_vae])

    assert not plan.supported and plan.cases == []
    assert "default VAE, SDXL-CPU-VAE, which runs on the CPU" in (plan.reason or "")
    assert _plan(_with_default_vae(sdxl, "sdxl-gpu-vae"), [SDXL_CPU_VAE, gpu_vae]).supported
    assert _plan(_with_default_vae(sdxl, "sd1-cpu-vae"), [sd1_cpu_vae]).supported


def test_an_sd_vae_is_calibrated_through_a_main_model_whose_default_vae_runs_on_the_gpu() -> None:
    sdxl_vae = _model("sdxl-vae", BaseModelType.StableDiffusionXL, ModelType.VAE)
    on_the_cpu = _with_default_vae(_model("a-sdxl", BaseModelType.StableDiffusionXL), "sdxl-cpu-vae")
    own_vae = _model("b-sdxl", BaseModelType.StableDiffusionXL)

    plan = _plan(sdxl_vae, [on_the_cpu, own_vae, SDXL_CPU_VAE])
    alone = _plan(sdxl_vae, [on_the_cpu, SDXL_CPU_VAE])

    assert (plan.supported, plan.calibrated_through) == (True, "b-sdxl")
    assert not alone.supported and "A-SDXL decodes with its default VAE" in (alone.reason or "")


def test_a_model_with_a_vae_slot_picks_a_gpu_vae_over_its_default_one_on_the_cpu() -> None:
    sd3 = _with_default_vae(_model("sd3", BaseModelType.StableDiffusion3), "a-cpu-vae")
    cpu = _model("a-cpu-vae", BaseModelType.StableDiffusion3, ModelType.VAE, cpu_only=True)
    gpu = _model("b-gpu-vae", BaseModelType.StableDiffusion3, ModelType.VAE)

    assert {tuple(case.components.items()) for case in _plan(sd3, [cpu, gpu]).cases} == {(("vae", "b-gpu-vae"),)}
    alone = _plan(sd3, [cpu])
    assert not alone.supported and "no compatible VAE that runs on the GPU" in (alone.reason or "")


def test_a_pipeline_with_a_default_vae_decodes_with_it_in_the_standalone_vaes_dtype() -> None:
    """The FLUX.2 and Z-Image model loader nodes take the `vae` setting over the pipeline's own VAE. Under `precision:
    float32` a standalone FLUX VAE stays fp32 where the FLUX.2 pipeline's own loads in bf16; one on the CPU gives way to
    a pick on the GPU."""
    klein = _model("klein", BaseModelType.Flux2, format="diffusers")
    cpu = _model("a-cpu-vae", BaseModelType.Flux2, ModelType.VAE, cpu_only=True)
    gpu = _model("b-gpu-vae", BaseModelType.Flux2, ModelType.VAE)

    assert _lines_of(_plan(klein, [cpu, gpu], vae_dtypes=FP32_VAES)) == {("decode_bf16", ())}
    assert _lines_of(_plan(_with_default_vae(klein, "b-gpu-vae"), [cpu, gpu], vae_dtypes=FP32_VAES)) == {
        ("decode_fp32", ())
    }
    assert _lines_of(_plan(_with_default_vae(klein, "a-cpu-vae"), [cpu, gpu], vae_dtypes=FP32_VAES)) == {
        ("decode_fp32", (("vae", "b-gpu-vae"),))
    }


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
    """The decode would run on the CPU, where no calibration window opens; without any VAE the graph cannot compile.
    (The FLUX.2 model takes its text encoder from the standalone one.)"""
    cpu_vae = _model("cpu-vae", vae_base, ModelType.VAE, cpu_only=True)

    for models in ([cpu_vae], []):
        plan = _plan(main, [*models, QWEN3_8B])
        assert not plan.supported and plan.cases == [] and "GPU" in (plan.reason or "")
    assert _plan(main, [cpu_vae, _model("gpu-vae", vae_base, ModelType.VAE), QWEN3_8B]).supported


def test_a_pipeline_with_its_own_vae_needs_no_gpu_vae_beside_it() -> None:
    cpu_vae = _model("cpu-vae", BaseModelType.Flux2, ModelType.VAE, cpu_only=True)
    klein_9b = _model("klein-9b", BaseModelType.Flux2, format="diffusers", variant=Flux2VariantType.Klein9B)

    assert _plan(_model("klein", BaseModelType.Flux2, format="diffusers"), [cpu_vae]).supported
    # Nor does a FLUX.2 model decoding with its component source's VAE.
    assert _plan(GGUF_KLEIN_9B, [cpu_vae, klein_9b]).supported


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
