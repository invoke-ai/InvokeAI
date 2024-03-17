#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import warnings

from invokeai.frontend.install.invokeai_configure import run_configure as configure

if __name__ == "__main__":
    warnings.warn(
        "configure_invokeai.py is deprecated, running 'invokeai-configure'...", DeprecationWarning, stacklevel=2
    )
    configure()
