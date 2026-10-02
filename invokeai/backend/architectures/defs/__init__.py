"""One module per model architecture, each calling `registry.register(...)` exactly once.

The modules here are discovered and imported automatically — there is no list to add to. The
filename is the base value with `-` replaced by `_`, which is what `registry.defs_module_path()`
computes, so an error message can name the file to edit without a second table to keep in sync.

Modules here import `architectures.registry` directly and never the `architectures` package. Doing
the latter would ask a partially-initialized package for an attribute, since it is that package's
own import that brings us here.
"""

from importlib import import_module
from pathlib import Path
from types import ModuleType

from invokeai.backend.util.module_discovery import discover_modules

_MODULES: dict[str, ModuleType] = {
    name: import_module(name) for name in discover_modules(Path(__file__).parent, f"{__name__}.")
}
