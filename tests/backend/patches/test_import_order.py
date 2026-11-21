import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["invokeai.backend.patches.layers.utils", "invokeai.backend.patches.layers.dora_layer"],
)
def test_patch_layers_import_in_fresh_interpreter(module: str):
    # Regression for #10063: these must not depend on model_manager.load being imported first.
    result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
