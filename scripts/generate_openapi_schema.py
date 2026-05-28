import json
import os
import sys
from pathlib import Path


def main():
    # Change working directory to the repo root
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    from invokeai.app.api_app import app
    from invokeai.app.util.custom_openapi import get_openapi_func

    schema = get_openapi_func(app)()

    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
            f.write("\n")
    else:
        json.dump(schema, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()