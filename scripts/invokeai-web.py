#!/usr/bin/env python

# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import os
import logging

logging.getLogger("xformers").addFilter(lambda record: "A matching Triton is not available" not in record.getMessage())


def main():
    # Change working directory to the repo root
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from invokeai.app.api_app import invoke_api

    invoke_api()


if __name__ == "__main__":
    main()
