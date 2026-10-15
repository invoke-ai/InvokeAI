"""`import invokeai.invocation_api` must work as the first invokeai import in a process.

It is the first import in the node-pack guide, so every custom node pack starts here. It broke once:
the node modules were imported in `invokeai/app/invocations/__init__.py`, which made touching *any*
submodule -- including the `baseinvocation` import that `invocation_api` itself begins with -- pull
in the whole tree. `composition-nodes.py` imports `invocation_api` back, and the cycle closed:
`ImportError: cannot import name 'BaseInvocation' from partially initialized module`.

The app never noticed, because by the time it loads custom nodes everything is already imported.
Only a fresh process starting at `invocation_api` sees it, which is why this runs in a subprocess.
"""

import subprocess
import sys

PROGRAM = """
import sys

import invokeai.invocation_api as api

assert "BaseInvocation" in api.__all__, "invocation_api did not export its own surface"
assert api.BaseInvocation is not None

# The point of the split: importing the public surface must not drag in every node module. If this
# ever has to change, the cycle above is what will break.
loaded = [m for m in sys.modules if m.startswith("invokeai.app.invocations.") and "flux" in m]
assert not loaded, f"invocation_api pulled in node modules: {sorted(loaded)[:5]}"

print("OK")
"""


def test_invocation_api_is_importable_on_its_own() -> None:
    result = subprocess.run([sys.executable, "-c", PROGRAM], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-3000:]}"
    assert "OK" in result.stdout


def test_the_documented_node_pack_imports_work_on_their_own() -> None:
    """What `docs/development/Guides/creating-node-pack.mdx` tells authors to write, verbatim."""
    program = (
        "from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation\n"
        "from invokeai.app.invocations.fields import InputField, OutputField\n"
        "from invokeai.invocation_api import BaseInvocationOutput, invocation_output\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"stderr:\n{result.stderr[-3000:]}"
    assert "OK" in result.stdout
