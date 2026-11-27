"""Every data file a backend module reads at runtime ships in the wheel.

Vendored tokenizers and configs are read from the installed package, so a file that
`[tool.setuptools.package-data]` does not match is absent from a wheel install: the loader raises
`FileNotFoundError`, or -- for a tokenizer missing only its vocabulary -- encodes every prompt to
nothing. A source checkout has every file, so nothing but this test notices.
"""

import tomllib
from fnmatch import fnmatch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = REPO_ROOT / "invokeai" / "backend"
DATA_SUFFIXES = (".json", ".json.gz")


def _package_data() -> dict[str, list[str]]:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["setuptools"]["package-data"]


def _is_shipped(path: Path, package_data: dict[str, list[str]]) -> bool:
    """True if a package-data glob of `path`'s package, or of any package above it, matches it."""
    relative = path.relative_to(REPO_ROOT)
    parts = relative.parts
    for depth in range(len(parts) - 1, 0, -1):
        package = ".".join(parts[:depth])
        member = "/".join(parts[depth:])
        for pattern in package_data.get(package, []):
            # setuptools globs: `**` spans directories, `*` stays within one.
            if fnmatch(member, pattern) and (pattern.count("/") == member.count("/") or "**" in pattern):
                return True
    return False


def test_every_vendored_backend_data_file_is_package_data() -> None:
    package_data = _package_data()
    vendored = [
        p for p in BACKEND.rglob("*") if p.is_file() and p.name.endswith(DATA_SUFFIXES) and "__pycache__" not in p.parts
    ]
    assert vendored, "found no vendored data files -- the scan is broken"

    missing = sorted(str(p.relative_to(REPO_ROOT)) for p in vendored if not _is_shipped(p, package_data))
    assert not missing, "not matched by [tool.setuptools.package-data] in pyproject.toml:\n  " + "\n  ".join(missing)
