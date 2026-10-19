"""Concrete facets — the kinds of fact an architecture can declare.

Imported automatically, like `defs/`, and for a sharper reason than symmetry: `validate()` checks
each architecture against the facets that marked themselves `REQUIRED`, and it learns which those
are from `Facet.FACET_TYPES`, which is filled at class-creation time. A facet module that nothing
happened to import would be absent from that collection, so its requirement would go unchecked —
silently, and precisely for a facet so new that no architecture declares it yet.
"""

from importlib import import_module
from pathlib import Path
from types import ModuleType

from invokeai.backend.util.module_discovery import discover_modules

_MODULES: dict[str, ModuleType] = {
    name: import_module(name) for name in discover_modules(Path(__file__).parent, f"{__name__}.")
}
