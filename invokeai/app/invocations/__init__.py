import importlib
from pathlib import Path

# import any devcontainer mounted nodes within 'custom_nodes' as modules (devcontainer mounts them here instead of /invokeai/nodes to avoid confusing development tools like the debugger or pylance or whatever).
importlib.import_module("invokeai.app.invocations.custom_nodes")

# add core nodes to __all__
python_files = filter(lambda f: not f.name.startswith("_"), Path(__file__).parent.glob("*.py"))
__all__ = [f.stem for f in python_files]  # type: ignore
