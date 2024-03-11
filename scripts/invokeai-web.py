#!/usr/bin/env python

# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import logging
import os

from invokeai.frontend.cli.app_arg_parser import app_arg_parser

logging.getLogger("xformers").addFilter(lambda record: "A matching Triton is not available" not in record.getMessage())


def main():
    # Parse CLI args immediately to handle `version` and `help` commands. Once the app starts up, we will parse the
    # args again to get configuration args.
    app_arg_parser.parse_args()

    # Change working directory to the repo root
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from invokeai.app.api_app import invoke_api

    invoke_api()


if __name__ == "__main__":
    main()
