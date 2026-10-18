"""The calibration plan: which cases calibrate a model on this GPU, which models share its lines, and when it cannot."""

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.working_memory_calibration.working_memory_calibration_plan import (
    DECODE_LINES,
    build_plan,
    case_sides,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, FluxVariantType, ModelType, ModelVariantType
from invokeai.backend.util import vae_working_memory
from invokeai.backend.working_memory.fingerprint import DeviceClass
from invokeai.backend.working_memory.platform import Platform
from invokeai.backend.working_memory.spec import import_estimator_modules
from tests.fixtures.working_memory_calibration import GPU, calibration_service

GIB = 2**30
CARD = DeviceClass(vendor="nvidia", name="Test GPU 24GB", total_mib=24 * 1024, backend="cuda", arch="sm_89")
CUDNN = Platform("cudnn", "native", "linux")
BYTES_PER_PIXEL = {"fp16": 1000, "bf16": 1000, "fp32": 2000}


def _model(key: str, base: BaseModelType, type: ModelType = ModelType.Main, **fields) -> SimpleNamespace:
    return SimpleNamespace(key=key, name=key.upper(), base=base, type=type, **fields)


def _reservation(estimator: str, height: int, width: int, dtype: str) -> int:
    return BYTES_PER_PIXEL[dtype] * height * width


def _plan(target, models=(), *, device_class=CARD, unavailable=None, precision="auto", reservation=_reservation):
    return build_plan(
        target,
        [target, *models],
        device_class=device_class if unavailable is None else None,
        unavailable=unavailable,
        platform=CUDNN if unavailable is None else None,
        precision=precision,
        reservation=reservation,
        specs=import_estimator_modules(),
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
    plan = _plan(_model("main", base), precision=precision)

    assert plan.supported and plan.estimators == [estimator]
    assert sorted({case.line for case in plan.cases}) == lines
    # Only the SD nodes cast the VAE; elsewhere the loader decides the dtype.
    assert all(bool(case.settings) == (base == BaseModelType.StableDiffusion1) for case in plan.cases)


def test_the_largest_size_is_capped_at_what_generate_accepts() -> None:
    assert case_sides(lambda side: side * side, 10**15) == [2048, 2880, 4096]


def test_a_card_that_cannot_hold_a_small_decode_with_headroom_is_unsupported() -> None:
    small = DeviceClass(vendor="nvidia", name="Test GPU 2GB", total_mib=2048, backend="cuda", arch="sm_75")

    plan = _plan(_model("flux", BaseModelType.Flux), device_class=small)

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
