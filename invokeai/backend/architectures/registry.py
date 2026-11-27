"""The architecture registry: which facets each `BaseModelType` declares.

Adding a model architecture means editing a long tail of core files — a `step_callback` branch, a
`safe_globals` entry, a variant-enum lookup — and forgetting one of them fails at *generation* time
rather than at boot. The registry moves those facts next to each other, in one file per
architecture, and `validate()` turns "you forgot one" into a startup error.

Storage and error behaviour follow `model_manager.load.model_loader_registry`: a plain dict, and a
double registration raises with the full context rather than overwriting. Re-stating the *same*
declaration is the exception — see `register()`.
"""

from typing import Final, TypeVar

from invokeai.backend.architectures.facet import Facet
from invokeai.backend.model_manager.taxonomy import BaseModelType

FacetT = TypeVar("FacetT", bound=Facet)

_ARCHITECTURES: Final[dict[BaseModelType, dict[type[Facet], Facet]]] = {}

_NOT_ARCHITECTURES: Final = frozenset(
    {
        # A fallback for models with no architecture association at all (CLIP, and the like).
        BaseModelType.Any,
        # Not an architecture but a hosting mode: the model runs at an external provider.
        BaseModelType.External,
        # Identification failed. Nothing can be declared about it by definition.
        BaseModelType.Unknown,
    }
)
"""The `BaseModelType` members that are not architectures. The single definition of that set: both
`register()` and `validate()` read it, so there is no second list to keep in sync."""

_PACKAGE = "invokeai/backend/architectures"
_FACETS_MODULE: Final = "invokeai.backend.architectures.facets"
"""Where a facet has to live to be enforced by `validate()`. See the filter there."""


class ArchitectureError(ValueError):
    """Raised for a missing or malformed architecture declaration.

    Subclasses `ValueError` deliberately: it replaces `raise ValueError(f"Unsupported base model:
    ...")` at the call sites it takes over, so existing `except ValueError` handlers keep behaving
    exactly as they did.
    """


def defs_module_path(base: BaseModelType) -> str:
    """The file a contributor has to edit to change what `base` declares.

    Derived from `base.value` — the one identifier that cannot change, because it is persisted in
    the model database — rather than read from a second table that could disagree with reality.
    """
    return f"{_PACKAGE}/defs/{base.value.replace('-', '_')}.py"


def register(base: BaseModelType, *facets: Facet) -> None:
    """Declare what `base` is. Called once per architecture, from its own module under `defs/`.

    Re-executing a `defs/` module — `importlib.reload`, or jurigged re-running the file on save
    under `--dev_reload` — calls this again. An identical declaration is the same fact stated twice,
    so it is a no-op; a *differing* one raises, which is the collision this guard exists for. Facets
    are frozen dataclasses, so "identical" is by value.

    A `defs/` module is therefore not usefully hot-reloadable: you reload it because you edited it,
    and an edited declaration is exactly the differing case, so it raises and needs a restart. The
    no-op path only keeps an *unedited* re-execution — a reload that swept this module up with the
    one that changed — from failing for no reason.
    """
    if base in _NOT_ARCHITECTURES:
        raise ArchitectureError(
            f"'{base.value}' is not a model architecture and cannot be registered. "
            f"It is one of {sorted(b.value for b in _NOT_ARCHITECTURES)}."
        )

    by_type: dict[type[Facet], Facet] = {}
    for facet in facets:
        facet_type = type(facet)
        if facet_type in by_type:
            raise ArchitectureError(
                f"Architecture '{base.value}' declares {facet_type.__name__} more than once in the "
                f"same register() call. See {defs_module_path(base)}."
            )
        by_type[facet_type] = facet

    declared = _ARCHITECTURES.get(base)
    if declared is not None and declared != by_type:
        raise ArchitectureError(
            f"Architecture '{base.value}' is already registered with a different declaration. Every "
            f"architecture is declared exactly once, in {defs_module_path(base)}."
        )
    _ARCHITECTURES[base] = by_type


def get(base: BaseModelType, facet_type: type[FacetT]) -> FacetT | None:
    """The facet of that type declared by `base`, or None. Use `require` unless None is meaningful."""
    facet = _ARCHITECTURES.get(base, {}).get(facet_type)
    # `isinstance` narrows `Facet | None` to `FacetT`. The registry keys by exact runtime type, so a
    # hit always passes; the check is here to satisfy the type checker without a cast.
    return facet if isinstance(facet, facet_type) else None


def require(base: BaseModelType, facet_type: type[FacetT]) -> FacetT:
    """The facet of that type declared by `base`. Raises, naming the file to edit, if it is missing.

    The message is the point of this function. It is read by someone whose new architecture just
    failed mid-generation, and it has to say which file to open — not merely which base was
    unsupported.
    """
    facet = get(base, facet_type)
    if facet is not None:
        return facet
    if base not in _ARCHITECTURES:
        raise ArchitectureError(
            f"Architecture '{base.value}' is not registered, so it cannot declare "
            f"{facet_type.__name__}. Create {defs_module_path(base)} with a "
            f"`register(BaseModelType.{base.name}, {facet_type.__name__}(...))` call. It is picked "
            f"up automatically; there is no import list to edit."
        )
    raise ArchitectureError(
        f"Architecture '{base.value}' does not declare {facet_type.__name__}. Add "
        f"{facet_type.__name__}(...) to the `register(...)` call in {defs_module_path(base)}."
    )


def generative_bases() -> frozenset[BaseModelType]:
    """Every registered architecture."""
    return frozenset(_ARCHITECTURES)


def facets_of(base: BaseModelType) -> tuple[Facet, ...]:
    """Everything `base` declares. For diagnostics and tests, not for dispatch."""
    return tuple(_ARCHITECTURES.get(base, {}).values())


def validate() -> None:
    """Check the registry is complete. Called at boot; raises rather than warns.

    Two things are checked, and both are boot errors rather than CI-only assertions because the
    whole point is that an incompletely declared architecture cannot start the app:

    1. Every `BaseModelType` that is not a sentinel has a module under `defs/`. Discovery is
       automatic, so the way to fail here is to add an enum member and no file.
    2. Every registered architecture declares every facet marked `REQUIRED`.

    Called from `ApiDependencies.initialize`, which is the one place every entry point that builds
    services passes through. Unlike the custom-node checks in `run_app`, which warn, this raises:
    architectures are first-party and the set is closed, so an incomplete one is a bug here.
    """
    missing_bases = sorted(
        base.value for base in BaseModelType if base not in _NOT_ARCHITECTURES and base not in _ARCHITECTURES
    )
    if missing_bases:
        raise ArchitectureError(
            "These architectures are not registered: "
            + ", ".join(f"'{b}'" for b in missing_bases)
            + ". Each needs a module under "
            + f"{_PACKAGE}/defs/ calling `register(...)`; the filename is the base value with "
            + "'-' replaced by '_'."
        )

    # `Facet.FACET_TYPES` collects every `Facet` subclass created anywhere in the interpreter,
    # custom-node packs included. A pack that merely defines a REQUIRED facet would otherwise make
    # this raise sixteen times over, naming first-party files under defs/ as the culprit — and
    # `load_custom_nodes` has already swallowed the pack's own errors by then, so nothing would
    # point at the pack. Only this package's facets are requirements of this package.
    required = [
        facet_type
        for facet_type in Facet.FACET_TYPES
        if facet_type.REQUIRED and facet_type.__module__.startswith(f"{_FACETS_MODULE}.")
    ]
    problems = [
        f"'{base.value}' does not declare {facet_type.__name__} (add it in {defs_module_path(base)})"
        for base in sorted(_ARCHITECTURES, key=lambda b: b.value)
        for facet_type in required
        if facet_type not in _ARCHITECTURES[base]
    ]
    if problems:
        raise ArchitectureError("Incomplete architecture declarations:\n  " + "\n  ".join(problems))
