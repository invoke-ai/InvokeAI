import json
import os
import sys
from pathlib import Path


def main():
    # Resolve the output path against the caller's working directory *before* chdir'ing to the repo root,
    # so a relative path (e.g. `generate_openapi_schema.py openapi.json` run from invokeai/frontend/web)
    # lands where the caller expects rather than at the repo root.
    output_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None

    # Change working directory to the repo root
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    from invokeai.app.api_app import app
    from invokeai.app.util.custom_openapi import get_openapi_func

    schema = get_openapi_func(app)()

    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
            f.write("\n")
    else:
        json.dump(schema, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
