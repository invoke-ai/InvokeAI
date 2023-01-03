#!/usr/bin/env python

import sys
import os
if sys.platform == 'darwin':
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import ldm.invoke.CLI
ldm.invoke.CLI.main()

