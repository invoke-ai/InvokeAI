'''
ldm.invoke.globals defines a small number of global variables that would
otherwise have to be passed through long and complex call chains.

It defines a Namespace object named "Globals" that contains
the attributes:

  - root    - the root directory under which "models" and "outputs" can be found
'''

import os
from argparse import Namespace

Globals = Namespace()

# This is usually overwritten by the command line and/or environment variables
Globals.root = '.'

# Where to look for the initialization file
Globals.initfile = os.path.expanduser('~/.invokeai')

# Awkward workaround to disable attempted loading of pypatchmatch
# which is causing CI tests to error out.
Globals.try_patchmatch = True
