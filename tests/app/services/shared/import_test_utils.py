import subprocess
import sys
import textwrap


def assert_module_imports_without_graph(module_name: str) -> None:
    """Import a shared execution-engine leaf while rejecting graph-facade imports."""

    script = textwrap.dedent(
        """
        import builtins
        import importlib
        import sys

        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "invokeai.app.services.shared.graph" or name.startswith("invokeai.app.services.shared.graph."):
                raise ModuleNotFoundError("graph import blocked")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import
        importlib.import_module(MODULE_NAME)
        assert "invokeai.app.services.shared.graph" not in sys.modules
        """
    ).replace("MODULE_NAME", repr(module_name))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr
