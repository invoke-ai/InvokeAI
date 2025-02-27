import shutil
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_custom_nodes(custom_nodes_path: Path):
    """
    Loads all custom nodes from the custom_nodes_path directory.

    This function copies a custom __init__.py file to the custom_nodes_path directory, effectively turning it into a
    python module.

    The custom __init__.py file itself imports all the custom node packs as python modules from the custom_nodes_path
    directory.

    Then,the custom __init__.py file is programmatically imported using importlib. As it executes, it imports all the
    custom node packs as python modules.
    """
    custom_nodes_path.mkdir(parents=True, exist_ok=True)

    custom_nodes_init_path = str(custom_nodes_path / "__init__.py")
    custom_nodes_readme_path = str(custom_nodes_path / "README.md")

    # copy our custom nodes __init__.py to the custom nodes directory
    shutil.copy(Path(__file__).parent / "custom_nodes/init.py", custom_nodes_init_path)
    shutil.copy(Path(__file__).parent / "custom_nodes/README.md", custom_nodes_readme_path)

    # set the same permissions as the destination directory, in case our source is read-only,
    # so that the files are user-writable
    for p in custom_nodes_path.glob("**/*"):
        p.chmod(custom_nodes_path.stat().st_mode)

    # Import custom nodes, see https://docs.python.org/3/library/importlib.html#importing-programmatically
    spec = spec_from_file_location("custom_nodes", custom_nodes_init_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load custom nodes from {custom_nodes_init_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
