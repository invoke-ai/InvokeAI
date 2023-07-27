#!/bin/env python

import sys
from pathlib import Path
from invokeai.backend.model_management.model_probe import ModelProbe

info = ModelProbe().probe(Path(sys.argv[1]))
print(info)

      
      
