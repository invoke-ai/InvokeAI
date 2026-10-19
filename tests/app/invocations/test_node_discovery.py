"""Every node module on disk must actually have been imported.

The walker itself is exercised against a synthetic tree in
`tests/backend/util/test_module_discovery.py`. What is left to check here is that this package's
real layout agrees with it — the realistic mistake being a new architecture folder that never got an
`__init__.py`, which is not a package and therefore contributes no nodes at all.

Calling `load_all_modules()` here rather than reading a module-level dict is the point of the
split: the package body must stay cheap so that `import invokeai.invocation_api` -- the first
import in the node-pack guide -- does not pull in this tree and close an import cycle.
"""

from pathlib import Path

from invokeai.app.invocations import load_all_modules
from invokeai.app.invocations._legacy_module_compatibility import LEGACY_MODULE_MAP

PACKAGE = "invokeai.app.invocations"


def test_every_node_module_on_disk_was_imported() -> None:
    root = Path(__file__).parents[3] / "invokeai" / "app" / "invocations"
    on_disk = {
        f"{PACKAGE}." + p.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        for p in root.rglob("*.py")
        if not any(part.startswith("_") for part in p.relative_to(root).parts)
    }
    assert on_disk - set(LEGACY_MODULE_MAP) == set(load_all_modules()), "walker and filesystem disagree"
