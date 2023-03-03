#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import warnings
from invokeai.frontend.CLI import invokeai_command_line_interface as main
warnings.warn("dream.py is being deprecated, please run invoke.py for the "
              "new UI/API or legacy_api.py for the old API",
              DeprecationWarning)
main()

