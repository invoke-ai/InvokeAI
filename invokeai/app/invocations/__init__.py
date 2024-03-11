import shutil
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from invokeai.app.services.config.config_default import get_config

custom_nodes_path = Path(get_config().custom_nodes_path.resolve())
custom_nodes_path.mkdir(parents=True, exist_ok=True)

custom_nodes_init_path = str(custom_nodes_path / "__init__.py")
custom_nodes_readme_path = str(custom_nodes_path / "README.md")

# copy our custom nodes __init__.py to the custom nodes directory
shutil.copy(Path(__file__).parent / "custom_nodes/init.py", custom_nodes_init_path)
shutil.copy(Path(__file__).parent / "custom_nodes/README.md", custom_nodes_readme_path)

# Import custom nodes, see https://docs.python.org/3/library/importlib.html#importing-programmatically
spec = spec_from_file_location("custom_nodes", custom_nodes_init_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load custom nodes from {custom_nodes_init_path}")
module = module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

# add core nodes to __all__
python_files = filter(lambda f: not f.name.startswith("_"), Path(__file__).parent.glob("*.py"))
__all__ = [f.stem for f in python_files]  # type: ignore
