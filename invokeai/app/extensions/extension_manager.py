import pathlib
from typing import List

from ...backend.util.logging import getLogger
from ..services.config import get_invokeai_config


class InvokeAIExtensionManager():
    def __init__(self) -> None:
        self.logger = getLogger('Extension Manager')
        self.config = get_invokeai_config()
        self.community_nodes_dir = pathlib.Path(self.config.root / 'nodes')
        assert self.community_nodes_dir.is_dir()

    def load_extensions(self) -> List:
        loaded_extensions = []
        for extension in self.community_nodes_dir.iterdir():
            if extension.is_dir() and not extension.name.startswith('__'):
                # Search for py files that are not named __init__.py in extensions root directory
                py_files = list(extension.glob('*.py'))
                py_files = [
                    file for file in py_files if not file.name == "__init__.py"]

                # If no py files are found, extension is not loaded.
                if len(py_files) == 0:
                    self.logger.warn(
                        f'Extension: "{extension.name}" failed to load. No nodes found.')
                    continue

                # Every py file in the root directory of the extension is loaded as an invocation
                # This will allow people to pack multiple nodes in different files in the same extension
                # All subfolders are ignored. These can be used for secondary operations needed for the node.
                for file in extension.iterdir():
                    if file.is_file() and file.name != '__init__.py' and file.suffix == ".py":
                        loaded_extensions.append((file.parent / file.stem).__str__())
                        self.logger.info(
                            f'Extension: "{extension.name}" loaded')
        
        return loaded_extensions