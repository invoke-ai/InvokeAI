import pathlib

from ...backend.util.logging import InvokeAILogger
from ..services.config import get_invokeai_config

logger = InvokeAILogger.getLogger()

__all__ = []


def load_nodes():
    native_nodes_dir = pathlib.Path(__file__).parent
    assert native_nodes_dir.is_dir()

    community_nodes_dir = pathlib.Path(get_invokeai_config().root / 'nodes')
    assert community_nodes_dir.is_dir()

    # Load Native Nodes
    for file in native_nodes_dir.iterdir():
        if file.is_file() and file.name != '__init__.py' and file.suffix == ".py":
            __all__.append(file.stem)

    # Load Community Nodes
    for extension in community_nodes_dir.iterdir():
        if extension.is_dir() and extension.name != '__pycache__':
            # Search for py files that are not named __init__.py in extensions root directory
            py_files = list(extension.glob('*.py'))
            py_files = [
                file for file in py_files
                if not file.name.endswith("__init__.py")]

            # If no py files are found, extensions is not loaded.
            if len(py_files) == 0:
                logger.warn(
                    f'Extension: "{extension.name}" failed to load. No nodes found.')
                return

            # Every py file in the root directory of the extension is loaded as a node
            # This will allow people to pack multiple nodes in different files in the same extension
            # All subfolders are ignored. These can be used for secondary operations needed for the node.
            for file in extension.iterdir():
                if file.is_file() and file.name != '__init__.py' and file.suffix == ".py":
                    __all__.append((file.parent / file.stem).__str__())
                    logger.info(f'Extension: "{extension.name}" loaded')


load_nodes()
