import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import torch

from invokeai.backend.working_memory.coefficients import ResolvedCoefficients, shipped_coefficients
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate, WorkingMemoryTerm
from invokeai.backend.working_memory.platform import platform_for

TermsFunction = Callable[[Mapping[str, FeatureValue], Mapping[str, float]], tuple[WorkingMemoryTerm, ...]]
"""An estimator's pure part: its features and its coefficient values in, its terms out."""

ESTIMATOR_MODULES: Final = ("invokeai.backend.util.vae_working_memory",)
"""Every module that registers an `EstimatorSpec`. The registry holds only the specs of modules imported so far, so a
caller that needs all of them imports these (`import_estimator_modules`); `tests/backend/working_memory/` fails on a
module that declares a spec without being listed here."""


@dataclass(frozen=True)
class EstimatorSpec:
    """What a working-memory estimator declares: its id and version, the coefficients its fitted terms read, and the
    platform keys those coefficients are shipped per.

    Registered at import of the module that defines the estimator. `terms` is that estimator's pure part, so an
    estimate can be evaluated again from recorded features alone -- which is how the measured points under `data/` are
    checked. It is behaviour rather than declaration and stays out of equality and hashing.
    """

    id: str
    version: int
    """Bumped whenever `terms` changes the formula, so coefficients fitted to the old one no longer resolve."""
    coefficients: frozenset[str]
    platform_keys: tuple[str, ...]
    terms: TermsFunction = field(compare=False, repr=False)

    def estimate(self, features: Mapping[str, FeatureValue], device: torch.device) -> WorkingMemoryEstimate:
        """The estimate for an operation on `device`, priced with the shipped coefficients of the platform it runs on."""
        return self.evaluate(features, shipped_coefficients(self, platform_for(device)))

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
