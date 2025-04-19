from pathlib import Path

# import any folders within 'custom_nodes/src/' as modules (for better devcontainer development).
from .custom_nodes import *  # type: ignore[reportUnusedImport]

# add core nodes to __all__
python_files = filter(lambda f: not f.name.startswith("_"), Path(__file__).parent.glob("*.py"))
__all__ = [f.stem for f in python_files]  # type: ignore