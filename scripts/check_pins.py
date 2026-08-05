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
index and resolves wheels for the wrong backend. This script therefore checks:

  1. pins.json carries exactly the platform/backend entries in REQUIRED_BACKENDS
     — no more, no less.
  2. Each of those URLs matches the corresponding ``torch-<backend>``
     ``[[tool.uv.index]]`` URL in pyproject.toml.
  3. pins.json's ``python`` — the version the launcher builds the venv with,
     before it installs anything — satisfies ``project.requires-python``.
     Otherwise the launcher creates an interpreter that the package metadata it
     is about to install rejects, and the install fails at the last step.

It also prints a *warning* when the pin is a version pyproject.toml's classifiers
do not mention. That is advisory only: classifiers are optional in PEP 621 and
informational on PyPI, so they cannot decide whether a pin is installable —
``requires-python`` does. One consequence is deliberate and worth stating: a
version that satisfies an open-ended ``requires-python`` but that no interpreter
has (``>=3.11`` with a ``3.99`` pin) is not caught here. Catching it would mean
gating on non-normative metadata, and a checker that rejects a legal pin is worse
than one that misses an implausible typo.

The repo root is derived from this file's own location, so the working directory
does not matter — only the path you hand to python does::

    python3 scripts/check_pins.py                     # from the repo root
    python3 /path/to/InvokeAI/scripts/check_pins.py   # from anywhere else

Kept dependency-free (stdlib only) so CI can run it with a bare ``python3``,
which is why the ``requires-python`` handling below is hand-rolled rather than
using ``packaging``.
"""

import json
import re
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

# The version the launcher is told to build the venv with, which must be exactly
# major.minor ("3.12"). A patch-level pin is rejected on purpose: `uv venv --python
# 3.12` resolves to the newest 3.12.x available, whereas `--python 3.12.7` demands
# one exact build, which freezes users on an unpatched interpreter and stops working
# outright once that build leaves uv's index. The launcher's own reinstall check
# compares only `major()`/`minor()` of this field against the existing venv, so a
# patch component is inert there in any case. Forbidding the third component also
# makes a bogus patch such as "3.12.9999" - which satisfies every specifier but which
# `uv python find` cannot resolve - unrepresentable rather than merely unlikely. A
# bogus major.minor is caught instead by requires-python, as far as its bounds reach.
# Leading zeros are rejected too, so the pin has exactly one spelling: "03.12" would
# otherwise compare equal to "3.12" here while reaching the launcher verbatim.
_VERSION_RE = re.compile(r"(?:0|[1-9][0-9]?)\.(?:0|[1-9][0-9]?)\Z")

# A single requires-python clause ("<3.13", ">=3.11.4"). Anything else legal in PEP
# 440 but not matched here (epochs, pre-releases, `~=`, `.*` wildcards) is reported as
# unevaluatable rather than guessed at, so a specifier this script cannot reason about
# fails loudly.
_CLAUSE_RE = re.compile(r"(==|!=|>=|<=|>|<)\s*([0-9]{1,4}(?:\.[0-9]{1,4}){0,2})\Z")

# "Programming Language :: Python :: 3.12" - a *whole* classifier naming exactly one
# major.minor version. The required dot excludes "... :: 3" and "... :: 3 :: Only";
# the anchor excludes "... :: 3.12 :: Only" and "... :: 3.1.4", neither of which
# declares support for the version it appears to name. Leading zeros are excluded for
# the same reason as in _VERSION_RE: "... :: 3.012" is not a trove classifier, so
# reading it as a declaration of 3.12 support would silence the advisory in exactly the
# case it exists to name. Digits are bounded like the two patterns above, so a 4000-digit
# "version" is not read as one and cannot end up quoted back in the advisory.
_CLASSIFIER_RE = re.compile(r"Programming Language :: Python :: ((?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3}))\Z")

_SUPPORTED_OPERATORS = "==, !=, >=, <=, > and <"


def _parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _satisfies(version: tuple[int, ...], operator: str, bound: tuple[int, ...]) -> bool:
    """Compare two dotted versions, zero-padding the shorter one (3.12 == 3.12.0)."""

    width = max(len(version), len(bound))
    version += (0,) * (width - len(version))
    bound += (0,) * (width - len(bound))

    match operator:
        case "==":
            return version == bound
        case "!=":
            return version != bound
        case ">=":
            return version >= bound
        case "<=":
            return version <= bound
        case ">":
            return version > bound
        case "<":
            return version < bound
        case _:
            # Unreachable via _CLAUSE_RE. Raising rather than falling through keeps a
            # newly-added operator from silently inheriting some other operator's answer.
            raise ValueError(f"unsupported operator {operator!r}")


def _requires_python_clauses(pyproject: dict) -> tuple[list[str], list[str]]:
    """Split project.requires-python into clauses. Returns (clauses, errors)."""

    project = pyproject.get("project")
    requires_python = project.get("requires-python") if isinstance(project, dict) else None

    if requires_python is None:
        return [], [
            "pyproject.toml has no project.requires-python, so there is nothing to validate "
            "pins.json's python pin against"
        ]
    if not isinstance(requires_python, str):
        return [], [f"pyproject.toml project.requires-python is {requires_python!r}, not a string"]

    # Empty clauses come from a trailing or doubled comma, which PEP 440 tolerates.
    clauses = [clause for clause in (part.strip() for part in requires_python.split(",")) if clause]
    if not clauses:
        return [], [
            f"pyproject.toml project.requires-python is {requires_python!r}, which constrains "
            "nothing, so pins.json's python pin cannot be validated"
        ]
    return clauses, []


def _classifier_versions(pyproject: dict) -> list[str]:
    """The major.minor versions project.classifiers claims support for, in declared order.

    Tolerates every malformed shape - a missing [project] table, a non-list value, non-string
    entries - by returning nothing rather than raising. Nothing is a fine answer: classifiers
    are optional, so their absence and their malformation mean the same thing here, which is
    that there is no advisory to give.
    """

    project = pyproject.get("project")
    classifiers = project.get("classifiers") if isinstance(project, dict) else None
    if not isinstance(classifiers, list):
        return []
    matches = (_CLASSIFIER_RE.match(entry) for entry in classifiers if isinstance(entry, str))
    # dict.fromkeys de-duplicates without reordering; a repeated classifier is legal but would
    # otherwise be listed twice in the warning.
    return list(dict.fromkeys(match.group(1) for match in matches if match is not None))


def check_python(pins: dict, pyproject: dict) -> list[str]:
    """Check pins.json's `python` against pyproject.toml's `requires-python`.

    Never raises: every malformed shape is reported as an error string, because this
    runs before the torchIndexUrl checks and a raise here would hide them.
    """

    clauses, errors = _requires_python_clauses(pyproject)
    pinned = pins.get("python")

    if pinned is None:
        errors.append(
            "pins.json is missing the 'python' field; the launcher uses it to decide which python "
            "version to build the venv with, and it must satisfy pyproject.toml's requires-python"
        )
        return errors
    if not isinstance(pinned, str) or _VERSION_RE.match(pinned) is None:
        errors.append(
            f"pins.json python is {pinned!r}; expected a major.minor version the launcher can hand "
            "to `uv venv --python`, like '3.12' (a patch component is deliberately not accepted - "
            "uv already picks the newest patch for a major.minor version)"
        )
        return errors

    version = _parse_version(pinned)
    requires_python = ", ".join(clauses)

    for clause in clauses:
        match = _CLAUSE_RE.match(clause)
        if match is None:
            errors.append(
                f"pyproject.toml requires-python clause '{clause}' is not one this script can "
                f"evaluate (it understands {_SUPPORTED_OPERATORS} against dotted numeric versions); "
                "teach scripts/check_pins.py this form rather than assuming the pin is still valid"
            )
            continue
        operator, bound = match.groups()
        if not _satisfies(version, operator, _parse_version(bound)):
            errors.append(
                f"pins.json python is '{pinned}' but pyproject.toml requires-python is "
                f"'{requires_python}' (fails '{clause}'); the launcher would create a python {pinned} "
                "venv and then install package metadata that rejects that interpreter"
            )

    return errors


def check_python_classifiers(pins: dict, pyproject: dict) -> list[str]:
    """Return advisory notes - never errors - about the python pin vs project.classifiers.

    `requires-python` is what actually gates installation, and it is the only authority this
    script fails on. Classifiers are optional in PEP 621 and purely informational on PyPI, so a
    pin they don't mention is a documentation gap, not a broken install: `requires-python =
    ">=3.11, <3.13"` genuinely permits a 3.11 pin whether or not a 3.11 classifier exists.
    Saying so out loud is still useful - the classifiers are what we publish - but it must not
    fail the build, and their *absence* must not be treated as a finding at all.

    Never raises, for the same reason check_python doesn't.
    """

    pinned = pins.get("python") if isinstance(pins, dict) else None
    if not isinstance(pinned, str) or _VERSION_RE.match(pinned) is None:
        return []  # check_python already reports the shape

    # A plain string comparison is enough: both patterns forbid leading zeros, so a version has
    # exactly one spelling on either side and there is nothing left to normalize.
    supported = _classifier_versions(pyproject)
    if not supported or pinned in supported:
        return []

    return [
        f"pins.json python is '{pinned}', which pyproject.toml's classifiers do not mention "
        f"(they list {', '.join(supported)}). That is allowed - classifiers are informational, and "
        "requires-python is what gates installation - but consider adding "
        f"'Programming Language :: Python :: {pinned}' so the published metadata matches."
    ]


def _uv_indexes(pyproject: dict) -> dict[str, str]:
    """Map [[tool.uv.index]] name -> url, tolerating a malformed or absent table."""

    tool = pyproject.get("tool")
    uv = tool.get("uv") if isinstance(tool, dict) else None
    entries = uv.get("index") if isinstance(uv, dict) else None
    if not isinstance(entries, list):
        return {}
    return {
        entry["name"]: entry["url"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("name"), str) and isinstance(entry.get("url"), str)
    }


def check_pins(pins: dict, pyproject: dict) -> list[str]:
    """Return a list of human-readable problems; empty means pins.json is fine."""

    if not isinstance(pins, dict):
        # Everything below indexes into it; report the shape rather than raising.
        return [f"pins.json is {pins!r}, not an object with 'python' and 'torchIndexUrl' keys"]

    indexes = _uv_indexes(pyproject)

    errors: list[str] = check_python(pins, pyproject)

    torch_index_url = pins.get("torchIndexUrl", {})
    if not isinstance(torch_index_url, dict):
        errors.append(f"pins.json torchIndexUrl is {torch_index_url!r}, not an object mapping platform to backend URLs")
        torch_index_url = {}

    for platform in sorted(set(REQUIRED_BACKENDS) - set(torch_index_url)):
        errors.append(f"pins.json is missing the torchIndexUrl.{platform} section")
    for platform in sorted(set(torch_index_url) - set(REQUIRED_BACKENDS)):
        errors.append(
            f"pins.json torchIndexUrl.{platform} is not a platform the launcher installs on; "
            f"expected only {sorted(REQUIRED_BACKENDS)}"
        )

    for platform, required in sorted(REQUIRED_BACKENDS.items()):
        if platform not in torch_index_url:
            continue  # already reported above as a missing section

        backends = torch_index_url[platform]
        if not isinstance(backends, dict):
            errors.append(
                f"pins.json torchIndexUrl.{platform} is {backends!r}, not an object mapping backend "
                "to index URL; the launcher would fail to parse pins.json"
            )
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
            "\nUpdate pins.json to match the [[tool.uv.index]] URLs and requires-python in "
            "pyproject.toml (or vice versa).",
            file=sys.stderr,
        )

    # Deliberately after the errors have already been printed: advice is worth less than a real
    # problem, so computing it must not be able to come between one and its report. It is still
    # printed on a failing run - a wrong pin and an unmentioned pin are usually the same edit.
    for warning in check_python_classifiers(pins, pyproject):
        print(f"warning: {warning}", file=sys.stderr)

    if errors:
        return 1

    print("pins.json is consistent with pyproject.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
