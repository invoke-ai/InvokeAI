#!/usr/bin/env python

# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import logging
import os

from invokeai.app.run_app import run_app

logging.getLogger("xformers").addFilter(lambda record: "A matching Triton is not available" not in record.getMessage())


def main():
    # Let run_app set the working directory to the repo root if needed.
    # Avoid global chdir to prevent unintended side effects on path resolution.
    run_app()


if __name__ == "__main__":
    main()
