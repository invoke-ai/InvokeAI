'''
ldm.invoke.globals defines a small number of global variables that would
otherwise have to be passed through long and complex call chains.

It defines a Namespace object named "Globals" that contains
the attributes:

  - root           - the root directory under which "models" and "outputs" can be found
  - initfile       - path to the initialization file
  - try_patchmatch - option to globally disable loading of 'patchmatch' module
'''

import os
import os.path as osp
from argparse import Namespace

Globals = Namespace()

# This is usually overwritten by the command line and/or environment variables
if os.environ.get('INVOKEAI_ROOT'):
    Globals.root = osp.abspath(os.environ.get('INVOKEAI_ROOT'))
elif os.environ.get('VIRTUAL_ENV'):
    Globals.root = osp.abspath(osp.join(os.environ.get('VIRTUAL_ENV'), '..'))
else:
    Globals.root = osp.abspath(osp.expanduser('~/invokeai'))

# Where to look for the initialization file
Globals.initfile = 'invokeai.init'

# Try loading patchmatch
Globals.try_patchmatch = True
