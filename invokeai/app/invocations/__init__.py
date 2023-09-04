import pathlib

from ..extensions.extension_manager import InvokeAIExtensionManager

__all__ = []
extension_manager = InvokeAIExtensionManager()

native_nodes_dir = pathlib.Path(__file__).parent
assert native_nodes_dir.is_dir()

# Load Native Nodes
for file in native_nodes_dir.iterdir():
    if file.is_file() and file.name != "__init__.py" and file.suffix == ".py":
        __all__.append(file.stem)

# Load Extensions
loaded_extensions = extension_manager.load_extensions()
__all__.extend(loaded_extensions)
