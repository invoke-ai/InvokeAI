import pathlib

from ..services.config import get_invokeai_config

__all__ = []


def load_nodes(directory: str | pathlib.Path, extensions=False):
    directory = directory if isinstance(
        directory, pathlib.Path) else pathlib.Path(directory)
    assert directory.is_dir()
    for file in directory.iterdir():
        if (file.is_file() and file.name != '__init__.py' and file.name[-3:] == ".py"):
            if extensions:
                __all__.append((file.parent / file.name[:-3]).__str__())
            else:
                __all__.append(file.name[:-3])


# Load Native Nodes
native_nodes_dir = pathlib.Path(__file__).parent
load_nodes(native_nodes_dir)

# Load Community Nodes
community_nodes_dir = pathlib.Path(get_invokeai_config().root / 'nodes')
load_nodes(community_nodes_dir, extensions=True)
