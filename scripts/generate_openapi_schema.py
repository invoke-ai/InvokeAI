import json
import os
import sys


def main():
    # Change working directory to the repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    # When invoked as a script, sys.path[0] is this script's directory rather than the repo
    # root, so ``import invokeai`` would resolve via the venv install — which on systems with
    # multi-worktree editable installs can pick up an `invokeai` namespace package missing
    # some of this worktree's invocation modules. Prepending the repo root ensures we always
    # import the local sources and so register every invocation declared here.
    if sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)

    from invokeai.app.api_app import app
    from invokeai.app.util.custom_openapi import get_openapi_func

    schema = get_openapi_func(app)()
    json.dump(schema, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
