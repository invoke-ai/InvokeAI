"""Check that pins.json is consistent with pyproject.toml.

``pins.json`` is not used anywhere in this repo — it is fetched (at the release
tag) by the Invoke Launcher (https://github.com/invoke-ai/launcher), which uses
its ``torchIndexUrl`` entries to pick the torch wheel index for legacy
(pre-6.14.0) installs. Because nothing in-repo consumes it, it can silently
drift from the ``[[tool.uv.index]]`` URLs in pyproject.toml — which is exactly
what happened when ROCm moved from 6.3 to 7.1 (issue #9328).

The launcher's schema makes every backend key optional and its install path only
passes ``--index`` when the selected entry exists, so a *missing* entry is just
as damaging as a stale one: the install silently falls back to the default PyPI
index and resolves wheels for the wrong backend. This script therefore checks
both halves:

  1. pins.json carries exactly the platform/backend entries in REQUIRED_BACKENDS
     — no more, no less.
  2. Each of those URLs matches the corresponding ``torch-<backend>``
     ``[[tool.uv.index]]`` URL in pyproject.toml.

Run from anywhere: python scripts/check_pins.py
"""

import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every platform/backend combination pins.json must carry, and no others.
#
# The launcher offers the same four GPU choices on every OS and maps them to a
# torch backend (amd -> rocm, either nvidia option -> cuda, no GPU -> cpu), then
# looks up `torchIndexUrl[sys.platform][backend]`. Any combination it can look up
# and that Invoke actually supports needs an entry here, because a missing one
# degrades silently rather than failing loudly.
#
# - darwin is deliberately empty: macOS uses MPS, for which PyTorch publishes no
#   separate index, so the launcher installs the default PyPI wheels.
# - win32 has no rocm entry: PyTorch publishes no ROCm wheels for Windows, and
#   the `rocm` extra in pyproject.toml is marked `sys_platform == 'linux'`.
REQUIRED_BACKENDS: dict[str, set[str]] = {
    "win32": {"cpu", "cuda"},
    "linux": {"cpu", "cuda", "rocm"},
    "darwin": set(),
}


def check_pins(pins: dict, pyproject: dict) -> list[str]:
    """Return a list of human-readable problems; empty means pins.json is fine."""

    indexes = {i["name"]: i["url"] for i in pyproject["tool"]["uv"]["index"]}
    torch_index_url = pins.get("torchIndexUrl", {})

    errors: list[str] = []

    for platform in sorted(set(REQUIRED_BACKENDS) - set(torch_index_url)):
        errors.append(f"pins.json is missing the torchIndexUrl.{platform} section")
    for platform in sorted(set(torch_index_url) - set(REQUIRED_BACKENDS)):
        errors.append(
            f"pins.json torchIndexUrl.{platform} is not a platform the launcher installs on; "
            f"expected only {sorted(REQUIRED_BACKENDS)}"
        )

    for platform, required in sorted(REQUIRED_BACKENDS.items()):
        backends = torch_index_url.get(platform)
        if backends is None:
            continue

        for backend in sorted(required - set(backends)):
            errors.append(
                f"pins.json torchIndexUrl.{platform}.{backend} is missing; the launcher would omit "
                f"--index for {backend} installs on {platform} and resolve torch from the default index"
            )
        for backend in sorted(set(backends) - required):
            errors.append(
                f"pins.json torchIndexUrl.{platform}.{backend} is not supported on {platform}; "
                f"expected only {sorted(required)}"
            )

        for backend in sorted(required & set(backends)):
            pinned_url = backends[backend]
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

    return errors


def main(repo_root: Path = REPO_ROOT) -> int:
    pins = json.loads((repo_root / "pins.json").read_text())
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())

    errors = check_pins(pins, pyproject)

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
