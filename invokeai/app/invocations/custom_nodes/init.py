"""
Invoke-managed custom node loader. See README.md for more information.
"""

import sys
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

    # we have a legit module to import
    spec = spec_from_file_location(module_name, init.absolute())

    if spec is None or spec.loader is None:
        logger.warn(f"Could not load {init}")
        continue

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    loaded_count += 1

    del init, module_name


logger.info(f"Loaded {loaded_count} modules from {Path(__file__).parent}")
