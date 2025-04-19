import importlib
import os
import pkgutil
import sys

from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()

# Compute the path to `src/` directory
here = os.path.dirname(__file__)
src_path = os.path.join(here, 'src')

# Tell Python to treat src/ as part of this package
#    (so that pkgutil.iter_modules will see the sub‑packages)
sys.path.insert(0, str(src_path))

loaded = 0
# Iterate over every entry in src/, and for each package, import it
for finder, pkg_name, is_pkg in pkgutil.iter_modules([src_path]):
    if not is_pkg:
        continue

    # avoid double‑import
    if pkg_name in sys.modules:
        continue

    try:
        logger.info(f"Importing custom node package {pkg_name!r}")
        importlib.import_module(pkg_name)
        loaded += 1
    except Exception as e:
        logger.error(f"Failed to import {pkg_name!r}: {e}")

if loaded:
    logger.info(f"Loaded {loaded} custom node package(s) from {src_path!r}")