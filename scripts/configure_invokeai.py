#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import warnings
from ldm.invoke.config import configure_invokeai

if __name__ == '__main__':
    warnings.warn("configire_invokeai.py is deprecated, please run 'invoke'", DeprecationWarning)
    configure_invokeai.main()
