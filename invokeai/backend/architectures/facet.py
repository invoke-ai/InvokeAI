"""The marker base class for architecture facets.

Deliberately its own module rather than part of `registry`: `facets/*` needs both, and putting
`Facet` in `registry` would make every facet module import the registry it is registered into.
"""

from typing import Any, ClassVar


class Facet:
    """One immutable, self-contained fact about a model architecture.

    Concrete facets are frozen dataclasses carrying plain data — never services, never model
    instances. They are keyed in the registry by their *exact* runtime type, so a facet class must
    not be subclassed by another facet class.

    Subclasses are collected at class-creation time so `registry.validate()` can check every
    registered architecture against the facets that declare themselves required. This mirrors
    `Config_Base.CONFIG_CLASSES`, but uses a dict rather than a set: a set iterates in a
    non-deterministic order, which has already produced a real bug in this codebase (see
    tests/backend/model_manager/configs/test_wan_lora_probe_independence.py). Error messages built
    by walking this collection would otherwise reorder between runs.
    """

    REQUIRED: ClassVar[bool] = False
    """Whether every architecture must declare this facet. Checked at boot, not at first use."""

    FACET_TYPES: ClassVar[dict[type["Facet"], None]] = {}
    """Every concrete facet class, in definition order. A dict used as an ordered set."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Facet.FACET_TYPES[cls] = None
