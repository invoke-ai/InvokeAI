import json
import os
import sys


def main():
    # Change working directory to the repo root
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from invokeai.app.api_app import app
    from invokeai.app.util.custom_openapi import get_openapi_func

    schema = get_openapi_func(app)()
    json.dump(schema, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
