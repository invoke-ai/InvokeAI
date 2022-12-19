'''
ldm.invoke.globals defines a small number of global variables that would
otherwise have to be passed through long and complex call chains.

It defines a Namespace object named "Globals" that contains
the attributes:

  - root           - the root directory under which e.g. "models" and "outputs" can be found
  - initfile       - path to the initialization file
  - outdir         - output directory
  - config         - models config file
  - try_patchmatch - option to globally disable loading of 'patchmatch' module
  - always_use_cpu - force use of CPU even if GPU is available
'''

from argparse import Namespace
from .paths import InvokePaths

Globals = Namespace()
Paths = InvokePaths()

Globals.root = Paths.root
Globals.initfile = Paths.initfile
Globals.outdir = Paths.outdir
Globals.config = Paths.config
Globals.config_dir = Paths.configdir
Globals.models_dir = Paths.models
Globals.autoscan_dir = Paths.default_weights

# Try loading patchmatch
Globals.try_patchmatch = True

# Use CPU even if GPU is available (main use case is for debugging MPS issues)
Globals.always_use_cpu = False

# Whether the internet is reachable for dynamic downloads
# The CLI will test connectivity at startup time.
Globals.internet_available = True