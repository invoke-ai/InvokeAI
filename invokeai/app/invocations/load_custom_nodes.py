import logging
import shutil
import sys
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_custom_nodes(custom_nodes_path: Path, logger: logging.Logger):
    """
    Loads all custom nodes from the custom_nodes_path directory.

    If custom_nodes_path does not exist, it creates it.

    It also copies the custom_nodes/README.md file to the custom_nodes_path directory. Because this file may change,
    it is _always_ copied to the custom_nodes_path directory.

    Then, it crawls the custom_nodes_path directory and imports all top-level directories as python modules.

    If the directory does not contain an __init__.py file or starts with an `_` or `.`, it is skipped.
    """

    # create the custom nodes directory if it does not exist
    custom_nodes_path.mkdir(parents=True, exist_ok=True)

    # Copy the README file to the custom nodes directory
    source_custom_nodes_readme_path = Path(__file__).parent / "custom_nodes/README.md"
    target_custom_nodes_readme_path = Path(custom_nodes_path) / "README.md"

    # copy our custom nodes README to the custom nodes directory
    shutil.copy(source_custom_nodes_readme_path, target_custom_nodes_readme_path)

    loaded_packs: list[str] = []
    failed_packs: list[str] = []

    # Import custom nodes, see https://docs.python.org/3/library/importlib.html#importing-programmatically
    for d in custom_nodes_path.iterdir():
        # skip files
        if not d.is_dir():
            continue

        # skip hidden directories
        if d.name.startswith("_") or d.name.startswith("."):
            continue

        # skip directories without an `__init__.py`
        init = d / "__init__.py"
        if not init.exists():
            continue

        module_name = init.parent.stem

        # skip if already imported
        if module_name in globals():
            continue

        # load the module
        spec = spec_from_file_location(module_name, init.absolute())

        if spec is None or spec.loader is None:
            logger.warning(f"Could not load {init}")
            continue

        logger.info(f"Loading node pack {module_name}")

        try:
            module = module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            loaded_packs.append(module_name)
        except Exception:
            failed_packs.append(module_name)
            full_error = traceback.format_exc()
            logger.error(f"Failed to load node pack {module_name} (may have partially loaded):\n{full_error}")

        del init, module_name

    loaded_count = len(loaded_packs)
    if loaded_count > 0:
        logger.info(
            f"Loaded {loaded_count} node pack{'s' if loaded_count != 1 else ''} from {custom_nodes_path}: {', '.join(loaded_packs)}"
        )
