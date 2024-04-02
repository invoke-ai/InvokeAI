"""
Invoke-managed custom node loader. See README.md for more information.
"""

import sys
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()
loaded_count = 0


for d in Path(__file__).parent.iterdir():
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

        loaded_count += 1
    except Exception:
        full_error = traceback.format_exc()
        logger.error(f"Failed to load node pack {module_name}:\n{full_error}")

    del init, module_name

if loaded_count > 0:
    logger.info(f"Loaded {loaded_count} node packs from {Path(__file__).parent}")
