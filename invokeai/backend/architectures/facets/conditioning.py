"""Which conditioning type an architecture serializes.

Text encoders write a `ConditioningFieldData` to disk and denoise nodes read it back. The reader
unpickles under `torch.load(weights_only=True)`, so every conditioning class has to be passed to
`torch.serialization.add_safe_globals` first — a process-global list built once, in
`ApiDependencies.initialize`.

Forgetting to add a new architecture's class there produces no error at boot and no error at encode.
It fails at *load*, inside the denoise node, after the text encoder has run and its output has been
written: an `UnpicklingError` naming a class the user has never heard of, halfway through a graph.

`add_safe_globals` mutates process-global state at a fixed point during startup, which is why the
registry has to be filled by the time `dependencies` is imported rather than on first use. That is
the constraint the module-scope import in `dependencies.py` was put there for; this is what redeems
it.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

from invokeai.backend.architectures.facet import Facet
from invokeai.backend.architectures.registry import generative_bases, get
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData


@dataclass(frozen=True)
class ConditioningFacet(Facet):
    """The `*ConditioningInfo` an architecture's text encoder produces."""

    REQUIRED: ClassVar[bool] = True

    info: type[Any]
    """The class itself, not its name. A name would have to be resolved back to a class to be
    registered as a safe global, and a typo would then fail at exactly the moment this facet exists
    to protect — during deserialization, mid-graph."""


def conditioning_infos() -> tuple[type[Any], ...]:
    """Every distinct conditioning class any architecture declares.

    Sorted by name so the resulting `safe_globals` list is stable across runs. Registry order is
    insertion order, which is the order the `defs/` modules were discovered in — reproducible in
    practice but incidental, and this list is worth being able to diff.

    Thirteen classes serve sixteen architectures: SD 1.x and 2.x share `BasicConditioningInfo`,
    SDXL and its refiner share `SDXLConditioningInfo`, and FLUX.2 encodes to `FLUXConditioningInfo`
    just as FLUX.1 does.
    """
    infos = {facet.info for base in generative_bases() if (facet := get(base, ConditioningFacet)) is not None}
    return tuple(sorted(infos, key=lambda cls: cls.__name__))


def conditioning_safe_globals() -> list[type[Any]]:
    """The exact list `dependencies` hands the conditioning `ObjectSerializerDisk`.

    Assembled here rather than at the call site so that a test can assert the call site passes
    *this* list. A test that rebuilt `[ConditioningFieldData, *conditioning_infos()]` itself would
    only ever agree with itself, and would stay green if the call site were shortened.

    `ConditioningFieldData` leads because it is the envelope every saved conditioning is wrapped in;
    the per-architecture classes it holds follow.
    """
    return [ConditioningFieldData, *conditioning_infos()]
