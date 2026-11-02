"""Every node module on disk must actually have been imported, and importing them all must stay cheap.

The walker itself is exercised against a synthetic tree in
`tests/backend/util/test_module_discovery.py`. What is left to check here is that this package's
real layout agrees with it — the realistic mistake being a new architecture folder that never got an
`__init__.py`, which is not a package and therefore contributes no nodes at all.

Calling `load_all_modules()` here rather than reading a module-level dict is the point of the
split: the package body must stay cheap so that `import invokeai.invocation_api` -- the first
import in the node-pack guide -- does not pull in this tree and close an import cycle.
"""

import subprocess
import sys
from pathlib import Path

import pytest

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


@pytest.mark.platform_independent
def test_loading_every_node_module_leaves_transformers_qwen3_5_unloaded() -> None:
    """Every app start loads every node module, so their module-level imports run at boot.

    transformers' Qwen3.5 modeling costs about 0.25 s there, and it imports the `causal_conv1d` and `fla` kernels
    whenever they are installed: a wheel built against another torch would stop the server from starting. Nodes that
    run Qwen3.5 (Anima's and MOSS-Image's prompt nodes) import it when they run. Indirect imports count as well:
    diffusers' FLUX.3 prompt-upsampler module loads it, which is why FLUX.3's layout code ports the box helpers it
    needs. A fresh interpreter, because this one has imported far more.
    """
    program = (
        "import sys\n"
        "from invokeai.app.invocations import load_all_modules\n"
        "load_all_modules()\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('transformers.models.qwen3_5'))\n"
        "assert not loaded, loaded\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"stderr:\n{result.stderr[-3000:]}"
    assert "OK" in result.stdout
