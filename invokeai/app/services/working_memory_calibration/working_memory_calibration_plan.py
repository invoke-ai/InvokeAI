"""The cases a client runs to calibrate one model's working memory, and which other models that calibration moves.

Covered: the decode of main models whose decode node prices with one of the five AutoencoderKL estimators of
`backend/util/vae_working_memory.py` (`DECODE_LINES`), and standalone VAEs of those bases, calibrated by generating
with a compatible installed main model. Each case is a two-step text-to-image generation, which runs the whole
pipeline and ends in an untiled decode; image-to-image would add an encode, which v1 does not plan.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, Optional

from invokeai.app.invocations.flux2.flux2_klein_model_loader import KLEIN_TO_QWEN3_VARIANT
from invokeai.app.invocations.model import is_self_contained_sdnq_flux1_pipeline, is_self_contained_sdnq_pipeline
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    WorkingMemoryCalibrationCase,
    WorkingMemoryCalibrationDeviceClassRef,
    WorkingMemoryCalibrationPlan,
    WorkingMemoryCalibrationRepetitions,
    WorkingMemoryCalibrationSharedModel,
)
from invokeai.backend.architectures import VaeFacet
from invokeai.backend.architectures import get as get_facet
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    Flux2VariantType,
    FluxVariantType,
    MistralVariantType,
    ModelType,
    ModelVariantType,
    Qwen3VariantType,
)
from invokeai.backend.util.vae_working_memory import VAE_PRETILE_VRAM_FRACTION
from invokeai.backend.working_memory import user_fit
from invokeai.backend.working_memory.fingerprint import DeviceClass
from invokeai.backend.working_memory.fit import MIN_AREA_RATIO, EntryKey
from invokeai.backend.working_memory.platform import Platform
from invokeai.backend.working_memory.probe import calibration_reservation
from invokeai.backend.working_memory.spec import EstimatorSpec

ModelConfig = Any
"""An installed model's config. Read by attribute: `key`, `name`, `base`, `type`, and where present `variant` and
`cpu_only`."""

ReservationFunction = Callable[[str, int, int, str], int]
"""The reservation an untiled decode of (estimator, height, width, dtype) gets on the calibration device."""


@dataclass(frozen=True)
class DecodeLines:
    estimator: str
    dtypes: tuple[str, ...] = ()
    """The dtypes the decode runs in, where they are fixed. Where the node casts the VAE (`vaePrecision`), each is a
    case. Empty: the VAE's loader decides (`decode_dtypes`)."""
    precision_setting: bool = False


@dataclass(frozen=True)
class VaeDtypes:
    """What the VAE loaders choose on the calibration device, with dtypes named as the lines are (`fp16`, `bf16`,
    `fp32`). The service reads them through the loaders' own rules, so the plan sizes the decode that runs."""

    standalone_flux: str
    """A standalone FLUX.1 or FLUX.2 VAE, and the FLUX.1 VAE Z-Image decodes with: the configured dtype, except fp16
    (`TorchDevice.dtype_avoiding_float16`). A configured fp32 stays fp32."""
    z_image_pipeline: str
    """The VAE of a Z-Image Diffusers pipeline: `TorchDevice.choose_bfloat16_safe_dtype`."""
    sdnq_quantized_vae: Callable[[Any], bool]
    """Whether the VAE folder of a FLUX.1 SDNQ pipeline holds SDNQ weights, as `FluxSDNQDiffusersModel._load_vae`
    tells (`is_sdnq_folder`)."""


FLUX2_PIPELINE_VAE_DTYPE: Final = "bf16"
"""The VAE of a FLUX.2 Diffusers pipeline, which `Flux2DiffusersModel` loads in bf16 whatever the configured dtype."""

SDNQ_PIPELINE_VAE_DTYPE: Final = "fp32"
"""The VAE of a self-contained SDNQ pipeline, which `Flux2SDNQCheckpointModel._load_vae`,
`ZImageSDNQCheckpointModel._load_vae` and, for a VAE folder of plain weights, `FluxSDNQDiffusersModel._load_vae` load
through `from_pretrained` without a `torch_dtype`: fp32, whatever the weights on disk are."""

SDNQ_QUANTIZED_VAE_DTYPE: Final = "bf16"
"""A FLUX.1 SDNQ pipeline's SDNQ-quantized VAE, which `FluxSDNQDiffusersModel._load_sdnq_vae` dequantizes at bf16."""

_SD = DecodeLines("vae.sd", ("fp16", "fp32"), precision_setting=True)
_FLUX = DecodeLines("vae.flux")
_FLUX2 = DecodeLines("vae.flux2")

DECODE_LINES: Final[Mapping[BaseModelType, DecodeLines]] = {
    BaseModelType.StableDiffusion1: _SD,
    BaseModelType.StableDiffusion2: _SD,
    BaseModelType.StableDiffusionXL: _SD,
    BaseModelType.StableDiffusionXLRefiner: _SD,
    # The SD3 VAE runs in the configured precision (`_sd3_dtype`).
    BaseModelType.StableDiffusion3: DecodeLines("vae.sd3"),
    # Z-Image decodes with a FLUX.1 VAE.
    BaseModelType.Flux: _FLUX,
    BaseModelType.ZImage: _FLUX,
    BaseModelType.Flux2: _FLUX2,
    # Decodes with the FLUX.2 VAE; shares its lines, but generates from structured prompts the plan does not write.
    BaseModelType.Ideogram4: _FLUX2,
    # The CogView4 loader forces bf16.
    BaseModelType.CogView4: DecodeLines("vae.cogview4", ("bf16",)),
}
"""Per base, the estimator its decode node prices with and, where fixed, the dtypes it runs in."""

FORCE_TILED_BASES: Final = frozenset(
    {
        BaseModelType.StableDiffusion1,
        BaseModelType.StableDiffusion2,
        BaseModelType.StableDiffusionXL,
        BaseModelType.StableDiffusionXLRefiner,
        BaseModelType.Flux,
        BaseModelType.ZImage,
        BaseModelType.Flux2,
    }
)
"""Bases whose decode node tiles every decode under `force_tiled_decode` (`latents_to_image`, `flux_vae_decode`,
`z_image_latents_to_image`, `flux2_vae_decode`). The SD3, CogView4 and Ideogram 4 nodes ignore the setting."""

_OWN_VAE_BASES: Final = frozenset({BaseModelType.Flux2, BaseModelType.ZImage})
"""Bases whose Diffusers and self-contained SDNQ pipelines decode with their own VAE when Generate's `vae` slot is
empty. A FLUX.1 Diffusers pipeline does not: Generate requires a standalone VAE for it."""

PLANNABLE_BASES: Final = frozenset(
    {
        BaseModelType.StableDiffusion1,
        BaseModelType.StableDiffusion2,
        BaseModelType.StableDiffusionXL,
        BaseModelType.StableDiffusion3,
        BaseModelType.Flux,
        BaseModelType.Flux2,
        BaseModelType.ZImage,
        BaseModelType.CogView4,
    }
)
"""Bases whose main models a text-to-image case can calibrate. The SDXL refiner generates from no text, and Ideogram 4
from structured prompts only."""

VAE_SLOT_BASES: Final = frozenset(
    {BaseModelType.Flux, BaseModelType.Flux2, BaseModelType.StableDiffusion3, BaseModelType.ZImage}
)
"""Bases whose Generate widget has a `vae` component slot. The SD bases pick a VAE through a setting a case cannot
override, and CogView4 has none: their standalone VAEs are calibrated through a main model's own VAE, which prices on
the same lines."""

NOT_TEXT_TO_IMAGE_VARIANTS: Final = frozenset(
    {ModelVariantType.Inpaint.value, ModelVariantType.Depth.value, FluxVariantType.DevFill.value}
)

CASE_STEPS: Final = 2
CASE_SEED: Final = 1
CASE_PROMPT: Final = "a mountain lake at dawn, detailed photograph"
REPETITIONS: Final = WorkingMemoryCalibrationRepetitions(warm_up=1, measured=2)

AREA_FRACTIONS: Final = (0.25, 0.5, 1.0)
SIDE_MULTIPLE: Final = 64
"""A multiple of every covered base's dimension grid (8, 16 or 32)."""
MAX_SIDE: Final = 4096
"""The largest width or height Generate accepts; no covered architecture declares a smaller limit."""
MIN_LARGEST_SIDE: Final = 512
VAE_WEIGHTS_ALLOWANCE_BYTES: Final = 2**30
"""What the VAE's own weights take on the device while it decodes: under 1 GiB for every covered VAE."""

SECONDS_MODEL_LOAD: Final = 30
SECONDS_PER_RUN: Final = 3
SECONDS_PER_MEGAPIXEL_STEP: Final = 1.0
SECONDS_PER_MEGAPIXEL_DECODE: Final = 1.0

ENCODE_NOT_COVERED: Final = (
    "encode: text-to-image runs no encode, so image-to-image and inpainting encodes keep the shipped values"
)


def sd3_dtype(precision: str) -> str:
    """The dtype the SD3 VAE runs in under the `precision` setting: `auto` is fp16 on a GPU."""
    return {"bfloat16": "bf16", "float32": "fp32"}.get(precision, "fp16")


def _value(field: object) -> str:
    return str(getattr(field, "value", field))


def ships_its_vae(model: ModelConfig) -> bool:
    """Whether a main model decodes with its own VAE while Generate's `vae` slot is empty, as Generate
    (`isBundledMainForBase`) and the model loader nodes treat it: a FLUX.2 or Z-Image Diffusers pipeline, or a
    self-contained SDNQ pipeline (`is_self_contained_sdnq_pipeline`; for FLUX.1 `is_self_contained_sdnq_flux1_pipeline`,
    which also needs the T5 pair)."""
    if not _is_main(model):
        return False
    if model.base == BaseModelType.Flux:
        return is_self_contained_sdnq_flux1_pipeline(model)
    if model.base not in _OWN_VAE_BASES:
        return False
    return _value(getattr(model, "format", None)) == "diffusers" or is_self_contained_sdnq_pipeline(model)


def _ignores_vae_pick(model: ModelConfig) -> bool:
    """Whether Generate drops a `vae` pick for the main model: a self-contained FLUX.1 SDNQ pipeline always decodes
    with its own VAE (`buildFLUXGraph` sends none). The FLUX.2 and Z-Image model loader nodes take a picked VAE first."""
    return model.base == BaseModelType.Flux and ships_its_vae(model)


_KLEIN_QWEN3: Final = {variant.value: encoder.value for variant, encoder in KLEIN_TO_QWEN3_VARIANT.items()}
"""The Qwen3 encoder variant each FLUX.2 Klein variant takes."""

_QWEN3_NAMES: Final = {Qwen3VariantType.Qwen3_4B.value: "Qwen3 4B", Qwen3VariantType.Qwen3_8B.value: "Qwen3 8B"}
"""How a reason names the Qwen3 encoder a Klein variant takes."""


def _fits_as_source(main: ModelConfig, source: ModelConfig) -> bool:
    """Whether a FLUX.2 pipeline is a component source Generate prefers for `main` (`isFlux2DiffusersSourceForModel`):
    [dev] for [dev], else one whose Qwen3 encoder matches the Klein variant's (`KLEIN_TO_QWEN3_VARIANT`)."""
    if _variant(main) == Flux2VariantType.Dev.value:
        return _variant(source) == Flux2VariantType.Dev.value
    required = _KLEIN_QWEN3.get(_variant(main) or "")
    return required is None or _KLEIN_QWEN3.get(_variant(source) or "") == required


def _flux2_pipelines(models: Sequence[ModelConfig]) -> list[ModelConfig]:
    """The installed FLUX.2 pipelines Generate offers as component sources, in `models`' order."""
    return [model for model in models if model.base == BaseModelType.Flux2 and ships_its_vae(model)]


def _has_standalone_encoder(main: ModelConfig, models: Sequence[ModelConfig]) -> bool:
    """Whether a standalone text encoder Generate can fill in for a FLUX.2 main model is installed: a Mistral encoder
    other than the Ministral 3B for [dev] (`isFlux2MistralEncoder`), else a Qwen3 encoder other than the 0.6B, of the
    size the Klein variant takes where it names one (`isFlux2Qwen3EncoderForModel`)."""
    if _variant(main) == Flux2VariantType.Dev.value:
        return any(
            model.type == ModelType.MistralEncoder and _variant(model) != MistralVariantType.Ministral3B.value
            for model in models
        )
    required = _KLEIN_QWEN3.get(_variant(main) or "")
    return any(
        model.type == ModelType.Qwen3Encoder
        and _variant(model) != Qwen3VariantType.Qwen3_06B.value
        and (required is None or _variant(model) == required)
        for model in models
    )


def _standalone_encoder_name(main: ModelConfig) -> str:
    """The standalone encoder `_has_standalone_encoder` looks for, as a reason names it."""
    if _variant(main) == Flux2VariantType.Dev.value:
        return "Mistral"
    return _QWEN3_NAMES.get(_KLEIN_QWEN3.get(_variant(main) or "", ""), "Qwen3")


def _lacks_text_encoder(main: ModelConfig, models: Sequence[ModelConfig]) -> bool:
    """Whether Generate cannot compile a FLUX.2 main model's graph for want of a text encoder: the model is no pipeline
    of its own, no installed pipeline fits it as a component source, and no standalone encoder for it is installed.
    A pipeline that does not fit gives only its VAE (`getFlux2DiffusersComponentSource`; the model loader node takes
    a source's encoder only from a fitting one)."""
    if main.base != BaseModelType.Flux2 or not _is_main(main) or ships_its_vae(main):
        return False
    if any(_fits_as_source(main, source) for source in _flux2_pipelines(models)):
        return False
    return not _has_standalone_encoder(main, models)


def vae_source(model: ModelConfig, models: Sequence[ModelConfig]) -> Optional[ModelConfig]:
    """The pipeline whose own VAE decodes a main model's generations while its `vae` slot is empty: the model itself
    (`ships_its_vae`), or for another FLUX.2 model the component source Generate then picks on its own and its graph
    keeps, whose VAE the FLUX.2 model loader node takes: the first fitting installed FLUX.2 pipeline, else the first
    one (`getAutoFlux2ComponentSourceModel`), which the graph keeps only beside a standalone text encoder
    (`getFlux2DiffusersComponentSource`). Generate lists the models in the order the server does, which is `models`'
    order. None: a standalone VAE decodes, or, for a FLUX.2 model without an encoder, no graph compiles
    (`_lacks_text_encoder`)."""
    if ships_its_vae(model):
        return model
    if not _is_main(model) or model.base != BaseModelType.Flux2:
        return None
    sources = _flux2_pipelines(models)
    fitting = next((source for source in sources if _fits_as_source(model, source)), None)
    if fitting is not None:
        return fitting
    return sources[0] if sources and _has_standalone_encoder(model, models) else None


def own_vae_dtype(pipeline: ModelConfig, vae: VaeDtypes) -> str:
    """The dtype a pipeline's own VAE loads in (`ships_its_vae`), as its loader yields it."""
    if _value(getattr(pipeline, "format", None)) == "diffusers":
        return vae.z_image_pipeline if pipeline.base == BaseModelType.ZImage else FLUX2_PIPELINE_VAE_DTYPE
    if pipeline.base == BaseModelType.Flux and vae.sdnq_quantized_vae(pipeline):
        return SDNQ_QUANTIZED_VAE_DTYPE
    return SDNQ_PIPELINE_VAE_DTYPE


def decode_dtypes(model: ModelConfig, precision: str, vae: VaeDtypes, models: Sequence[ModelConfig]) -> tuple[str, ...]:
    """The dtypes the decode of `model` (a main model or a VAE of a covered base, among the installed `models`) runs
    in: fixed per base, the `precision` setting for SD3, else the dtype the VAE that decodes loads in, a standalone one
    or a pipeline's own (`vae_source`)."""
    lines = DECODE_LINES[model.base]
    if lines.dtypes:
        return lines.dtypes
    if model.base == BaseModelType.StableDiffusion3:
        return (sd3_dtype(precision),)
    source = vae_source(model, models)
    return (vae.standalone_flux if source is None else own_vae_dtype(source, vae),)


def _variant(config: ModelConfig) -> Optional[str]:
    variant = getattr(config, "variant", None)
    return None if variant is None else _value(variant)


def _runs_on_gpu(vae: ModelConfig) -> bool:
    return getattr(vae, "cpu_only", None) is not True


def _accepts(main_base: BaseModelType, vae_base: BaseModelType) -> bool:
    facet = get_facet(main_base, VaeFacet)
    if facet is None:
        return main_base == vae_base
    return any(compatibility.base == vae_base for compatibility in facet.accepted)


def _is_main(config: ModelConfig) -> bool:
    return bool(config.type == ModelType.Main)


def _is_vae(config: ModelConfig) -> bool:
    return bool(config.type == ModelType.VAE)


def _plannable_main(config: ModelConfig) -> bool:
    return _is_main(config) and config.base in PLANNABLE_BASES and _variant(config) not in NOT_TEXT_TO_IMAGE_VARIANTS


def decode_budget(device_class: DeviceClass, capacity_bytes: Optional[int]) -> int:
    """The largest calibration reservation a case may take: within the share of the card a decode may claim before the
    node tiles it up front (`VAE_PRETILE_VRAM_FRACTION`), and within `capacity_bytes`, what the model cache can give on
    the device; each less the VAE's own weights, which must stay fully resident beside it. Nothing else does: a
    calibration lock first moves every other unlocked model out (`LoadedModelWithoutConfig.model_on_device`). Past the
    capacity the cache loads the VAE partly (`partial_residency`), and such a window fits no line. On a device that
    shares system memory (MPS, an integrated XPU) the capacity follows the RAM the host has available, which can be far
    below the share of the device's total. `capacity_bytes` None: no reading, and the share of the card alone bounds
    it."""
    ceiling = int(VAE_PRETILE_VRAM_FRACTION * device_class.total_mib * 2**20)
    if capacity_bytes is not None:
        ceiling = min(ceiling, capacity_bytes)
    return ceiling - VAE_WEIGHTS_ALLOWANCE_BYTES


def case_sides(reservation: Callable[[int], int], budget_bytes: int, floor_bytes: int = 0) -> list[int]:
    """The square sides of the three cases: about `AREA_FRACTIONS` of the largest area that decodes untiled here.

    The largest area is the largest square, in steps of `SIDE_MULTIPLE` and at most `MAX_SIDE`, whose calibration
    reservation (the estimate plus calibration headroom, `probe.calibration_reservation`, and never below
    `floor_bytes`, `device_working_mem_gb`, as the lock reserves it) fits `budget_bytes` (`decode_budget`). A larger
    decode would be tiled or partly loaded, and neither fits a line. Each fraction of that area is rounded down to a
    square side that is a multiple of `SIDE_MULTIPLE`; repeated sides are dropped. Returns no sides when the largest is
    below `MIN_LARGEST_SIDE` or the sides do not span two sizes `MIN_AREA_RATIO` apart.
    """
    largest = 0
    for side in range(SIDE_MULTIPLE, MAX_SIDE + 1, SIDE_MULTIPLE):
        if max(calibration_reservation(reservation(side)), floor_bytes) > budget_bytes:
            break
        largest = side
    if largest < MIN_LARGEST_SIDE:
        return []
    sides: list[int] = []
    for fraction in AREA_FRACTIONS:
        side = max(SIDE_MULTIPLE, int(math.sqrt(fraction) * largest) // SIDE_MULTIPLE * SIDE_MULTIPLE)
        if side not in sides:
            sides.append(side)
    if sides[-1] ** 2 < MIN_AREA_RATIO * sides[0] ** 2:
        return []
    return sides


def _square(reservation: ReservationFunction, estimator: str, dtype: str, side: int) -> int:
    return reservation(estimator, side, side, dtype)


def decode_lines_of(
    model: ModelConfig, precision: str, vae: VaeDtypes, models: Sequence[ModelConfig]
) -> list[tuple[str, str]]:
    """The (estimator, ``decode_<dtype>``) lines a model's decode prices on: a main model or a GPU VAE of a base in
    `DECODE_LINES`; none for any other model."""
    if getattr(model, "base", None) not in DECODE_LINES:
        return []
    if not (_is_main(model) or (_is_vae(model) and _runs_on_gpu(model))):
        return []
    estimator = DECODE_LINES[model.base].estimator
    return [(estimator, f"decode_{dtype}") for dtype in decode_dtypes(model, precision, vae, models)]


def estimated_seconds(cases: Sequence[WorkingMemoryCalibrationCase]) -> int:
    """A rough order of magnitude: one model load, then per run a fixed cost and a cost per megapixel for each step
    and for the decode. Not measured; real times vary several-fold with the GPU and the model."""
    runs = REPETITIONS.warm_up + REPETITIONS.measured
    seconds: float = SECONDS_MODEL_LOAD
    for case in cases:
        megapixels = case.width * case.height / 1e6
        per_run = SECONDS_PER_RUN + megapixels * (
            case.steps * SECONDS_PER_MEGAPIXEL_STEP + SECONDS_PER_MEGAPIXEL_DECODE
        )
        seconds += runs * per_run
    return math.ceil(seconds)


def _lines(
    model: ModelConfig,
    precision: str,
    vae: VaeDtypes,
    models: Sequence[ModelConfig],
    platform: Platform,
    specs: Mapping[str, EstimatorSpec],
) -> set[tuple[EntryKey, str]]:
    """The lines a model's decode prices with on `platform`."""
    estimator = DECODE_LINES[model.base].estimator
    spec = specs.get(estimator)
    carrier = user_fit.carrier_key(estimator, spec.version, platform) if spec is not None else None
    if carrier is None:
        return set()
    return {(carrier, f"decode_{dtype}") for dtype in decode_dtypes(model, precision, vae, models)}


def shared_with(
    target: ModelConfig,
    models: Sequence[ModelConfig],
    precision: str,
    vae: VaeDtypes,
    platform: Platform,
    specs: Mapping[str, EstimatorSpec],
) -> list[WorkingMemoryCalibrationSharedModel]:
    """The other installed main models and GPU VAEs of a covered base whose decode prices with a line the target's
    does, on this platform: a line is the shipped entry that carries the values (one estimator may borrow another's)
    and the operation and dtype."""
    mine = _lines(target, precision, vae, models, platform, specs)
    shared = []
    for model in models:
        if model.key == target.key or model.base not in DECODE_LINES:
            continue
        if not (_is_main(model) or (_is_vae(model) and _runs_on_gpu(model))):
            continue
        common = sorted(suffix for _, suffix in mine & _lines(model, precision, vae, models, platform, specs))
        if common:
            shared.append(
                WorkingMemoryCalibrationSharedModel(
                    key=model.key, name=model.name, base=_value(model.base), type=_value(model.type), lines=common
                )
            )
    return sorted(shared, key=lambda model: (model.name, model.key))


_STANDALONE_VAE_BASES: Final = frozenset({BaseModelType.Flux, BaseModelType.Flux2, BaseModelType.ZImage})
"""Bases whose Generate requires a `vae` pick, and fills one, unless a pipeline's own VAE decodes (`vae_source`)."""


def _accepted_vaes(main: ModelConfig, models: Sequence[ModelConfig]) -> list[ModelConfig]:
    return [model for model in models if _is_vae(model) and _accepts(main.base, model.base)]


def _needs_gpu_vae(main: ModelConfig, models: Sequence[ModelConfig]) -> bool:
    """Whether a main model decodes with a standalone VAE Generate fills in, and none it accepts runs on the GPU: a
    decode on the CPU opens no calibration window, and without any VAE the graph does not compile."""
    if main.base not in _STANDALONE_VAE_BASES or vae_source(main, models) is not None:
        return False
    return not any(_runs_on_gpu(vae) for vae in _accepted_vaes(main, models))


def _gpu_vae_pick(main: ModelConfig, models: Sequence[ModelConfig]) -> dict[str, str]:
    """A `vae` pick where Generate's own fill could land on a VAE that runs on the CPU: the first GPU VAE the main
    model accepts. None needed where every accepted VAE runs on the GPU, where the base has no VAE slot, or where a
    pipeline's own VAE decodes (`vae_source`), so Generate fills none."""
    if main.base not in VAE_SLOT_BASES or vae_source(main, models) is not None:
        return {}
    accepted = _accepted_vaes(main, models)
    if all(_runs_on_gpu(vae) for vae in accepted):
        return {}
    gpu = sorted((vae for vae in accepted if _runs_on_gpu(vae)), key=lambda vae: (vae.name, vae.key))
    return {"vae": gpu[0].key} if gpu else {}


def _unsupported(
    target: ModelConfig, reason: str, device: Optional[DeviceClass], estimators: list[str]
) -> WorkingMemoryCalibrationPlan:
    return WorkingMemoryCalibrationPlan(
        model_key=target.key,
        supported=False,
        reason=reason,
        device_class=WorkingMemoryCalibrationDeviceClassRef(id=device.id, name=device.name) if device else None,
        calibrated_through=None,
        estimators=estimators,
        cases=[],
        repetitions=REPETITIONS,
        estimated_seconds=None,
        shared_with=[],
        not_covered=[ENCODE_NOT_COVERED] if estimators else [],
    )


def build_plan(
    target: ModelConfig,
    models: Sequence[ModelConfig],
    *,
    device_class: Optional[DeviceClass],
    unavailable: Optional[str],
    platform: Optional[Platform],
    precision: str,
    vae_dtypes: VaeDtypes,
    reservation: ReservationFunction,
    specs: Mapping[str, EstimatorSpec],
    capacity_bytes: Optional[int] = None,
    working_memory_floor_bytes: int = 0,
    force_tiled_decode: bool = False,
) -> WorkingMemoryCalibrationPlan:
    """The plan for calibrating `target`, among the installed `models`, on the GPU of `device_class`.

    `unavailable` says why this machine cannot calibrate (no GPU, several generation devices); `device_class` and
    `platform` are then None. `reservation` prices an untiled decode on the calibration device; `precision` is the
    `precision` setting; `vae_dtypes` are the dtypes the VAE loaders choose on the device, so each case sizes the
    decode in the dtype it runs in; `capacity_bytes` is what the cache can give on the device (`decode_budget`), and
    `working_memory_floor_bytes` the least working memory a lock reserves there (`device_working_mem_gb`);
    `force_tiled_decode` is the setting, under which the decode nodes of `FORCE_TILED_BASES` run no untiled decode.
    """
    covered = target.base in DECODE_LINES and (_is_main(target) or _is_vae(target))
    estimators = [DECODE_LINES[target.base].estimator] if covered else []
    if not covered:
        kind = f"{_value(target.base)} {_value(target.type)}"
        return _unsupported(
            target,
            f"Calibration covers the decode of SD1, SD2, SDXL, SD3, FLUX.1, FLUX.2, Z-Image and CogView4 main models "
            f"and their VAEs; this is a {kind} model.",
            device_class,
            estimators,
        )

    main = target
    if _is_vae(target):
        if not _runs_on_gpu(target):
            return _unsupported(
                target,
                "This VAE runs on the CPU (cpu_only); calibration measures GPU memory.",
                device_class,
                estimators,
            )
        candidates = sorted(
            (
                model
                for model in models
                if _plannable_main(model) and _accepts(model.base, target.base) and not _ignores_vae_pick(model)
            ),
            key=lambda model: (model.name, model.key),
        )
        if not candidates:
            return _unsupported(
                target,
                "A VAE is calibrated by generating with a main model that decodes with it, and no compatible main "
                "model that takes a VAE pick is installed.",
                device_class,
                estimators,
            )
        # One whose graph compiles, where there is one; otherwise the first says what it lacks.
        main = next((model for model in candidates if not _lacks_text_encoder(model, models)), candidates[0])
    elif target.base not in PLANNABLE_BASES:
        return _unsupported(
            target,
            "This model shares its decode with others but cannot generate from a plain text prompt; calibrate one of "
            "the models listed as sharing it.",
            device_class,
            estimators,
        )
    elif _variant(target) in NOT_TEXT_TO_IMAGE_VARIANTS:
        return _unsupported(
            target, f"The {_variant(target)} variant does not generate from text alone.", device_class, estimators
        )

    if unavailable is not None or device_class is None or platform is None:
        return _unsupported(target, unavailable or "This machine has no GPU to calibrate.", None, estimators)

    if force_tiled_decode and main.base in FORCE_TILED_BASES:
        # Running the cases untiled would override the setting for these items only, and measure decodes none of this
        # server's generations runs: no estimate would read the fit.
        return _unsupported(
            target,
            "force_tiled_decode is on, so every decode of this model runs tiled: a calibration measures no untiled "
            "decode, and no estimate would read one. Set force_tiled_decode: false in invokeai.yaml and restart "
            "InvokeAI to calibrate.",
            device_class,
            estimators,
        )
    if _lacks_text_encoder(main, models):
        encoder = _standalone_encoder_name(main)
        return _unsupported(
            target,
            f"{main.name} has no text encoder Generate can use, so no case compiles: Generate takes it from a FLUX.2 "
            f"pipeline whose encoder fits the model or from a standalone {encoder} encoder, and a pipeline that does "
            f"not fit gives only its VAE. Install a {encoder} encoder or a fitting FLUX.2 pipeline.",
            device_class,
            estimators,
        )
    if not _is_vae(target) and _needs_gpu_vae(main, models):
        return _unsupported(
            target,
            f"{main.name} decodes with a standalone VAE, and no compatible VAE that runs on the GPU is installed: a "
            "decode on the CPU measures no GPU memory. Install a compatible VAE, or clear cpu_only on one.",
            device_class,
            estimators,
        )

    if _is_vae(target) and main.base in VAE_SLOT_BASES:
        components = {"vae": target.key}
    else:
        components = _gpu_vae_pick(main, models)

    lines = DECODE_LINES[target.base]
    budget = decode_budget(device_class, capacity_bytes)
    cases: list[WorkingMemoryCalibrationCase] = []
    # A standalone VAE target decodes as itself; a main model with a pipeline's own VAE, or what fills its slot.
    decoding = target if _is_vae(target) else main
    for dtype in decode_dtypes(decoding, precision, vae_dtypes, models):
        sides = case_sides(partial(_square, reservation, lines.estimator, dtype), budget, working_memory_floor_bytes)
        for side in sides:
            cases.append(
                WorkingMemoryCalibrationCase(
                    id=f"{lines.estimator}-decode-{dtype}-{side}x{side}",
                    model_key=main.key,
                    components=components,
                    width=side,
                    height=side,
                    steps=CASE_STEPS,
                    seed=CASE_SEED,
                    positive_prompt=CASE_PROMPT,
                    negative_prompt="",
                    settings={"vaePrecision": dtype} if lines.precision_setting else {},
                    estimator=lines.estimator,
                    line=f"decode_{dtype}",
                )
            )
    if not cases:
        return _unsupported(
            target,
            f"{device_class.name} cannot hold an untiled decode of {MIN_LARGEST_SIDE}x{MIN_LARGEST_SIDE} with "
            "calibration headroom and the VAE fully loaded, in the memory free on it now.",
            device_class,
            estimators,
        )

    return WorkingMemoryCalibrationPlan(
        model_key=target.key,
        supported=True,
        reason=None,
        device_class=WorkingMemoryCalibrationDeviceClassRef(id=device_class.id, name=device_class.name),
        calibrated_through=main.key if main is not target else None,
        estimators=estimators,
        cases=cases,
        repetitions=REPETITIONS,
        estimated_seconds=estimated_seconds(cases),
        shared_with=shared_with(target, models, precision, vae_dtypes, platform, specs),
        not_covered=[ENCODE_NOT_COVERED],
    )
