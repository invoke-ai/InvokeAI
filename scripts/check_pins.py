"""Check that pins.json is consistent with pyproject.toml.

``pins.json`` is not used anywhere in this repo — it is fetched (at the release
tag) by the Invoke Launcher (https://github.com/invoke-ai/launcher), which uses
its ``torchIndexUrl`` entries to pick the torch wheel index for legacy
(pre-6.14.0) installs. Because nothing in-repo consumes it, it can silently
drift from the ``[[tool.uv.index]]`` URLs in pyproject.toml — which is exactly
what happened when ROCm moved from 6.3 to 7.1 (issue #9328).

This script fails if any ``torchIndexUrl`` entry in pins.json differs from the
URL of the corresponding ``torch-<backend>`` index in pyproject.toml.

Run from anywhere: python scripts/check_pins.py
"""

import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    pins = json.loads((REPO_ROOT / "pins.json").read_text())
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    indexes = {i["name"]: i["url"] for i in pyproject["tool"]["uv"]["index"]}

    errors: list[str] = []

    for platform, backends in pins["torchIndexUrl"].items():
        for backend, pinned_url in backends.items():
            index_name = f"torch-{backend}"
            expected_url = indexes.get(index_name)
            if expected_url is None:
                errors.append(
                    f"pins.json torchIndexUrl.{platform}.{backend}: no [[tool.uv.index]] named '{index_name}' in pyproject.toml"
                )
            elif pinned_url != expected_url:
                errors.append(
                    f"pins.json torchIndexUrl.{platform}.{backend} is '{pinned_url}' but pyproject.toml index '{index_name}' is '{expected_url}'"
                )

    if errors:
        print("pins.json is out of sync with pyproject.toml:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nUpdate pins.json to match the [[tool.uv.index]] URLs in pyproject.toml (or vice versa).",
            file=sys.stderr,
        )
        return 1

    print("pins.json is consistent with pyproject.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
