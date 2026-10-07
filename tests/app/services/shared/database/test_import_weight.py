"""The database layer stays light to import: the user CLIs, migrations and the copy tool load it without the app."""

import subprocess
import sys


def test_the_layer_does_not_load_torch_or_the_invocations() -> None:
    # A query module that imported invocation fields once made every database-only program load torch (~9 s).
    probe = (
        "import sys\n"
        "import invokeai.app.services.shared.database.database\n"
        "import invokeai.app.services.shared.database.startup\n"
        "print(sorted(m for m in ('torch', 'invokeai.app.invocations') if m in sys.modules))\n"
    )

    loaded = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout

    assert loaded.strip() == "[]"
