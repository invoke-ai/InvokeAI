# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

import os
import sys

def main():
    # Change working directory to the repo root
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    if '--api' in sys.argv:
        from ldm.dream.app.api_app import invoke_api
        invoke_api()
    else:
        # TODO: Parse some top-level args here.
        from ldm.dream.app.cli_app import invoke_cli
        invoke_cli()


if __name__ == '__main__':
    main()
