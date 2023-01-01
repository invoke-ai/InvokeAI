#!/usr/bin/env python3

import os
from sys import platform


if platform == "darwin":
    add_environ: dict = {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}


def child_setup(add_environ: dict):
    """Called to setup the child process before exec()
    @add_environ is a dict for extra env variables
    """
    for key, val in add_environ.items():
        if val is None:
            val = ""
        os.putenv(key, val)

child_setup(add_environ)

import ldm.invoke.CLI
ldm.invoke.CLI.main()
