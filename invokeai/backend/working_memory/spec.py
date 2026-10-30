import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import torch

from invokeai.backend.working_memory import user_fit
from invokeai.backend.working_memory.coefficients import ResolvedCoefficients, shipped_coefficients
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.fit import LineFunction
from invokeai.backend.working_memory.platform import platform_for

TermsFunction = Callable[[Mapping[str, FeatureValue], Mapping[str, float]], tuple[WorkingMemoryTerm, ...]]
"""An estimator's pure part: its features and its coefficient values in, its terms out."""

ESTIMATOR_MODULES: Final = (
    "invokeai.backend.moss_image.working_memory",
    "invokeai.backend.qwen_image_2_1.vae",
    "invokeai.backend.util.vae_working_memory",
    "invokeai.backend.z_image.denoise_working_memory",
    "invokeai.backend.flux2.denoise_working_memory",
)
"""Every module that registers an `EstimatorSpec`. The registry holds only the specs of modules imported so far, so a
caller that needs all of them imports these (`import_estimator_modules`); `tests/backend/working_memory/` fails on a
module that declares a spec without being listed here."""


@dataclass(frozen=True)
class EstimatorSpec:
    """What a working-memory estimator declares: its id and version, the coefficients its fitted terms read, the
    platform keys those coefficients are shipped per, and how its features map to the line those terms price on.

    Registered at import of the module that defines the estimator. `terms` is that estimator's pure part, so an
    estimate can be evaluated again from recorded features alone -- which is how the measured points under `data/` are
    checked. `line` places features on the estimator's fitted line for the shipped fit (`fit.py`), the ids of measured
    points (`bounds.point_id`) and a user calibration. Both are behaviour rather than declaration and stay out of
    equality and hashing.
    """

    id: str
    version: int
    """Bumped whenever `terms` changes the formula, so coefficients fitted to the old one no longer resolve."""
    coefficients: frozenset[str]
    platform_keys: tuple[str, ...]
    terms: TermsFunction = field(compare=False, repr=False)
    line: LineFunction = field(compare=False, repr=False)
    """The line an operation's features price on, their size on it, and whether a measurement of them fits it: the
    AutoencoderKL convolution line in output pixels for every VAE estimator (`fit.convolution_line`), one in attended
    tokens for each denoise estimator. Declared by every estimator, so a new one cannot inherit another's line, and with it
    which of its measurements join a fit."""
    user_calibrated: bool = False
    """Whether a user's calibration moves this estimator: its windows fit user lines, its estimates read them, the
    calibration summary shows its lines, and what its windows found held outside torch sizes the calibration plan
    (`recorded_outside_torch_bytes`). Without it, its lines are the shipped fit's alone: a calibration run still records
    its windows, and the summary counts them among the windows that fit no line.

    The summary and the calibration plan are built on pixel lines: the summary keys a line's sizes by its windows'
    `width` and `height`, and the plan sizes square decodes. A spec whose line is another, such as the Z-Image
    denoise's in tokens, must not opt in until they key sizes by `fit.Line.size`."""

    def estimate(self, features: Mapping[str, FeatureValue], device: torch.device) -> WorkingMemoryEstimate:
        """The estimate for an operation on `device`, priced with the shipped coefficients of the platform it runs on,
        and, for a `user_calibrated` estimator, with the user's own line where a calibration of this device measured it
        (`user_fit.resolve`)."""
        platform = platform_for(device)
        shipped = shipped_coefficients(self, platform)
        fit = user_fit.user_fit_for(self, platform, device)
        if fit is None:
            return self.evaluate(features, shipped)
        return user_fit.resolve(self, features, shipped, fit)

    def evaluate(
        self, features: Mapping[str, FeatureValue], coefficients: ResolvedCoefficients
    ) -> WorkingMemoryEstimate:
        """The estimate for `features` with the given coefficients."""
        return WorkingMemoryEstimate(
            estimator=self.id,
            version=self.version,
            features=features,
            terms=self.terms(features, coefficients.values),
            source=coefficients.source,
        )


_SPECS: Final[dict[str, EstimatorSpec]] = {}


def _defined_at(terms: TermsFunction) -> str:
    return f"{terms.__module__}.{getattr(terms, '__qualname__', repr(terms))}"


def register(spec: EstimatorSpec) -> EstimatorSpec:
    """Declare an estimator, once per id, from the module that defines it.

    `importlib.reload` of that module declares the same spec again, from a new `terms` function of the same module and
    name: a no-op that adopts the re-executed function. Anything else under an id already taken raises -- differing
    declared data, the same data from another function, or the very same function object again (two specs of one module
    sharing a `terms` function and, by a copy-paste slip, an id). Each would otherwise silently replace one estimator's
    formula or coefficients with another's.

    jurigged (`--dev_reload`) patches function bodies in place and re-runs only the top-level statements that changed:
    an edited `terms` body takes effect with nothing re-registered, but an edited `register(...)` statement raises even
    when its data is unchanged (a reformat, a comment), because it hands over the same function object. Restart the
    server then.
    """
    declared = _SPECS.get(spec.id)
    if declared is not None:
        if declared != spec:
            raise ValueError(f"Working-memory estimator '{spec.id}' is already registered as {declared}.")
        if declared.terms is spec.terms or _defined_at(declared.terms) != _defined_at(spec.terms):
            raise ValueError(
                f"Working-memory estimator '{spec.id}' is already registered by {_defined_at(declared.terms)}; "
                "ids are unique per estimator."
            )
    _SPECS[spec.id] = spec
    return spec


def registered_specs() -> Mapping[str, EstimatorSpec]:
    """Every spec registered so far, by id: those of the estimator modules imported in this process."""
    return MappingProxyType(_SPECS)


def import_estimator_modules() -> Mapping[str, EstimatorSpec]:
    """Every spec there is: imports `ESTIMATOR_MODULES` first, so the answer does not depend on import order."""
    for name in ESTIMATOR_MODULES:
        importlib.import_module(name)
    return registered_specs()
