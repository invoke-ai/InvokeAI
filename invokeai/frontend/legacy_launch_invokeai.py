import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true")
    opts, _ = parser.parse_known_args()

    if opts.web:
        sys.argv.pop(sys.argv.index("--web"))
        from invokeai.app.api_app import invoke_api

        invoke_api()
    else:
        from invokeai.app.cli_app import invoke_cli

        invoke_cli()


if __name__ == "__main__":
    main()
