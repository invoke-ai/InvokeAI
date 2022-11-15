'''
ldm.invoke.globals defines a small number of global variables that would
otherwise have to be passed through long and complex call chains.

It defines a Namespace object named "Globals" that contains
the attributes:

  - root    - the root directory under which "models" and "outputs" can be found
'''

from argparse import Namespace

Globals = Namespace()
Globals.root = '.'

