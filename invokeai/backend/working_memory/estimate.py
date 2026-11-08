from collections.abc import Mapping
from dataclasses import dataclass, field

FeatureValue = int | float | str | bool | None
"""One runtime fact an estimator is evaluated on: a size, an element width, a tile, a byte count it measured."""


@dataclass(frozen=True)
class WorkingMemoryTerm:
    """One part of a working-memory estimate.

    A fitted term is proportional to a data coefficient; `fitted` builds it, so its `bytes` are exactly the
    coefficient's value times `scale`. An analytic term (a score matrix, an image buffer) carries neither. `included` is
    False for a term the estimate priced but does not reserve, such as the smaller of two phases that peak apart.
    """

    name: str
    bytes: float
    included: bool
    coefficient: str | None = None
    scale: float | None = None

    @classmethod
    def fitted(cls, name: str, coefficient: str, scale: float, values: Mapping[str, float]) -> "WorkingMemoryTerm":
        """A term of `scale` units at `values[coefficient]` bytes per unit."""
        return cls(name, values[coefficient] * scale, True, coefficient, scale)


@dataclass(frozen=True)
class WorkingMemoryEstimate:
    """What an operation is expected to need on its compute device, and the terms that add up to it."""

    estimator: str
    """The id of the `EstimatorSpec` that produced it, e.g. ``vae.flux2``."""
    version: int
    features: Mapping[str, FeatureValue]
    """Everything the estimate was evaluated from, runtime decisions such as the tile and the score-matrix size
    included, so it can be evaluated again from these alone."""
    terms: tuple[WorkingMemoryTerm, ...]
    source: str
    """Where the coefficients came from, e.g. ``shipped:conv_backend=miopen (values of vae.flux2)``."""
    reservation_bytes: int = field(init=False)
    """What the model cache keeps free: the included terms' bytes, truncated. Derived from the terms, never given."""

    def __post_init__(self) -> None:
        # One term at a time in term order rather than `sum()`, which compensates float rounding since Python 3.12,
        # so the reservation is exactly the running total the terms describe.
        total: float = 0
        for term in self.terms:
            if term.included:
                total += term.bytes
        object.__setattr__(self, "reservation_bytes", int(total))
