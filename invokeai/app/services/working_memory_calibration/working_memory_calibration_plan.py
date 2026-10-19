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
    FluxVariantType,
    ModelType,
    ModelVariantType,
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
    dtypes: tuple[str, ...]
    """The dtypes the decode runs in by default. Where the node casts the VAE (`vaePrecision`), each is a case."""
    precision_setting: bool = False


_SD = DecodeLines("vae.sd", ("fp16", "fp32"), precision_setting=True)
_FLUX = DecodeLines("vae.flux", ("bf16",))
_FLUX2 = DecodeLines("vae.flux2", ("bf16",))

DECODE_LINES: Final[Mapping[BaseModelType, DecodeLines]] = {
    BaseModelType.StableDiffusion1: _SD,
    BaseModelType.StableDiffusion2: _SD,
    BaseModelType.StableDiffusionXL: _SD,
    BaseModelType.StableDiffusionXLRefiner: _SD,
    # The SD3 VAE runs in the configured precision (`_sd3_dtype`).
    BaseModelType.StableDiffusion3: DecodeLines("vae.sd3", ()),
    # The FLUX.1 loaders avoid fp16, and Z-Image decodes with a FLUX.1 VAE.
    BaseModelType.Flux: _FLUX,
    BaseModelType.ZImage: _FLUX,
    BaseModelType.Flux2: _FLUX2,
    # Decodes with the FLUX.2 VAE; shares its lines, but generates from structured prompts the plan does not write.
    BaseModelType.Ideogram4: _FLUX2,
    # The CogView4 loader forces bf16.
    BaseModelType.CogView4: DecodeLines("vae.cogview4", ("bf16",)),
}
"""Per base, the estimator its decode node prices with and the dtypes it runs in."""

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


def decode_dtypes(base: BaseModelType, precision: str) -> tuple[str, ...]:
    lines = DECODE_LINES[base]
    return lines.dtypes or (sd3_dtype(precision),)


def _value(field: object) -> str:
    return str(getattr(field, "value", field))


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
    capacity the cache loads the VAE partly (`partial_residency`), and such a window fits no line. `capacity_bytes`
    None: the device does not say (shared memory), and the share of the card alone bounds it."""
    ceiling = int(VAE_PRETILE_VRAM_FRACTION * device_class.total_mib * 2**20)
    if capacity_bytes is not None:
        ceiling = min(ceiling, capacity_bytes)
    return ceiling - VAE_WEIGHTS_ALLOWANCE_BYTES


def case_sides(reservation: Callable[[int], int], budget_bytes: int) -> list[int]:
    """The square sides of the three cases: about `AREA_FRACTIONS` of the largest area that decodes untiled here.

    The largest area is the largest square, in steps of `SIDE_MULTIPLE` and at most `MAX_SIDE`, whose calibration
    reservation (the estimate plus calibration headroom, `probe.calibration_reservation`) fits `budget_bytes`
    (`decode_budget`). A larger decode would be tiled or partly loaded, and neither fits a line. Each fraction of that
    area is rounded down to a square side that is a multiple of `SIDE_MULTIPLE`; repeated sides are dropped. Returns
    no sides when the largest is below `MIN_LARGEST_SIDE` or the sides do not span two sizes `MIN_AREA_RATIO` apart.
    """
    largest = 0
    for side in range(SIDE_MULTIPLE, MAX_SIDE + 1, SIDE_MULTIPLE):
        if calibration_reservation(reservation(side)) > budget_bytes:
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


def decode_lines_of(model: ModelConfig, precision: str) -> list[tuple[str, str]]:
    """The (estimator, ``decode_<dtype>``) lines a model's decode prices on: a main model or a GPU VAE of a base in
    `DECODE_LINES`; none for any other model."""
    if getattr(model, "base", None) not in DECODE_LINES:
        return []
    if not (_is_main(model) or (_is_vae(model) and _runs_on_gpu(model))):
        return []
    estimator = DECODE_LINES[model.base].estimator
    return [(estimator, f"decode_{dtype}") for dtype in decode_dtypes(model.base, precision)]


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
    base: BaseModelType, precision: str, platform: Platform, specs: Mapping[str, EstimatorSpec]
) -> set[tuple[EntryKey, str]]:
    """The lines a base's decode prices with on `platform`."""
    estimator = DECODE_LINES[base].estimator
    spec = specs.get(estimator)
    carrier = user_fit.carrier_key(estimator, spec.version, platform) if spec is not None else None
    if carrier is None:
        return set()
    return {(carrier, f"decode_{dtype}") for dtype in decode_dtypes(base, precision)}


def shared_with(
    target: ModelConfig,
    models: Sequence[ModelConfig],
    precision: str,
    platform: Platform,
    specs: Mapping[str, EstimatorSpec],
) -> list[WorkingMemoryCalibrationSharedModel]:
    """The other installed main models and GPU VAEs of a covered base whose decode prices with a line the target's
    does, on this platform: a line is the shipped entry that carries the values (one estimator may borrow another's)
    and the operation and dtype."""
    mine = _lines(target.base, precision, platform, specs)
    shared = []
    for model in models:
        if model.key == target.key or model.base not in DECODE_LINES:
            continue
        if not (_is_main(model) or (_is_vae(model) and _runs_on_gpu(model))):
            continue
        common = sorted(suffix for _, suffix in mine & _lines(model.base, precision, platform, specs))
        if common:
            shared.append(
                WorkingMemoryCalibrationSharedModel(
                    key=model.key, name=model.name, base=_value(model.base), type=_value(model.type), lines=common
                )
            )
    return sorted(shared, key=lambda model: (model.name, model.key))


def _gpu_vae_pick(main: ModelConfig, models: Sequence[ModelConfig]) -> dict[str, str]:
    """A `vae` pick where Generate's own fill could land on a VAE that runs on the CPU: the first GPU VAE the main
    model accepts. None needed where every accepted VAE runs on the GPU, or where the base has no VAE slot."""
    if main.base not in VAE_SLOT_BASES:
        return {}
    accepted = [model for model in models if _is_vae(model) and _accepts(main.base, model.base)]
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
    reservation: ReservationFunction,
    specs: Mapping[str, EstimatorSpec],
    capacity_bytes: Optional[int] = None,
) -> WorkingMemoryCalibrationPlan:
    """The plan for calibrating `target`, among the installed `models`, on the GPU of `device_class`.

    `unavailable` says why this machine cannot calibrate (no GPU, several generation devices); `device_class` and
    `platform` are then None. `reservation` prices an untiled decode on the calibration device; `precision` is the
    `precision` setting; `capacity_bytes` is what the cache can give on the device (`decode_budget`).
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
            (model for model in models if _plannable_main(model) and _accepts(model.base, target.base)),
            key=lambda model: (model.name, model.key),
        )
        if not candidates:
            return _unsupported(
                target,
                "A VAE is calibrated by generating with a main model that decodes with it, and no compatible main "
                "model is installed.",
                device_class,
                estimators,
            )
        main = candidates[0]
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

    if _is_vae(target) and main.base in VAE_SLOT_BASES:
        components = {"vae": target.key}
    else:
        components = _gpu_vae_pick(main, models)

    lines = DECODE_LINES[target.base]
    budget = decode_budget(device_class, capacity_bytes)
    cases: list[WorkingMemoryCalibrationCase] = []
    for dtype in decode_dtypes(target.base, precision):
        sides = case_sides(partial(_square, reservation, lines.estimator, dtype), budget)
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
        shared_with=shared_with(target, models, precision, platform, specs),
        not_covered=[ENCODE_NOT_COVERED],
    )
