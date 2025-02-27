"""
Invoke-managed custom node loader. See README.md for more information.
"""

import sys
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()
loaded_packs: list[str] = []
failed_packs: list[str] = []

custom_nodes_dir = Path(__file__).parent

for d in custom_nodes_dir.iterdir():
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

    # load the module, appending adding a suffix to identify it as a custom node pack
    spec = spec_from_file_location(module_name, init.absolute())

    if spec is None or spec.loader is None:
        logger.warn(f"Could not load {init}")
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
        f"Loaded {loaded_count} node pack{'s' if loaded_count != 1 else ''} from {custom_nodes_dir}: {', '.join(loaded_packs)}"
    )
