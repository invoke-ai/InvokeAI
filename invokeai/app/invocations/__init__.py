"""Core invocation modules.

Every module in this package registers its `@invocation`-decorated classes with
`InvocationRegistry` as it is imported, so the app is only correct once *all* of them have been
imported. `load_all_modules()` does that, and `invokeai.app.services.shared.graph` calls it.

Importing them here, in the package body, would be the obvious shortcut and is wrong: it makes
`import invokeai.app.invocations.anything` -- including the `baseinvocation` import that
`invokeai.invocation_api` starts with -- pull in the whole tree. A node module that imports
`invocation_api` back (`composition-nodes.py` does) then closes a cycle, and
`import invokeai.invocation_api` fails outright with a partially initialized module. That is the
first import in the documented node-pack guide, so it has to stay cheap.

Discovery walks the whole package tree rather than globbing `*.py` in this directory. Node modules
are grouped into per-architecture subpackages (`flux/`, `wan/`, ...) and cross-cutting ones (`vae/`,
`text_encoder/`, `pid/`), and a flat glob would skip every one of them silently -- the failure would
not surface at boot but later, as an "unknown node type" when a user opens a workflow that uses one.
"""

from importlib import import_module
from pathlib import Path
from types import ModuleType

from invokeai.app.invocations._legacy_module_compatibility import LEGACY_MODULE_MAP
from invokeai.backend.util.module_discovery import discover_modules


def load_all_modules() -> dict[str, ModuleType]:
    """Import every node module in this package, registering the invocations it declares.

    Idempotent: `import_module` returns the cached module on later calls, so callers do not have to
    coordinate who invokes it first.
    """
    return {
        name: import_module(name)
        for name in discover_modules(
            Path(__file__).parent,
            f"{__name__}.",
            excluded_modules=LEGACY_MODULE_MAP,
        )
    }
