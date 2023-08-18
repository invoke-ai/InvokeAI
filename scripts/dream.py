#!/usr/bin/env python
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import warnings

from invokeai.app.cli_app import invoke_cli

warnings.warn(
    "dream.py is being deprecated, please run invoke.py for the " "new UI/API or legacy_api.py for the old API",
    DeprecationWarning,
)


invoke_cli()
