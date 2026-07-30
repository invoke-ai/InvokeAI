"""Assert that `uv.lock` gives linux/aarch64 an installable torch and torchvision.

Nothing else in CI covers aarch64. Three separate mechanisms in `pyproject.toml` conspire to make torch
resolve from PyPI there instead of from the PyTorch WHL indexes (which ship no aarch64 torchvision wheel):
`tool.uv.environments`, the `[tool.uv.sources]` platform markers, and the aarch64 fallback pins in each
torch extra. Break the sources markers and `uv lock` fails loudly, but break either of the other two and
the lockfile simply stops mentioning torch on aarch64 while `uv lock --locked` stays green -- which is how
aarch64 support was silently lost once before, when a ROCm bump narrowed `tool.uv.environments`.

So assert against the lockfile itself, which is the artifact users actually install from.

Usage: check_aarch64_lock.py [path/to/uv.lock]   (needs `packaging`)
"""

import sys
import tomllib
from pathlib import Path
from typing import Any

from packaging.markers import Marker
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename

# A linux/aarch64 interpreter, for evaluating the lockfile's environment markers. `python_version` is
# filled in per supported minor version below.
AARCH64_ENV = {
    "sys_platform": "linux",
    "platform_machine": "aarch64",
    "platform_system": "Linux",
    "os_name": "posix",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}

REQUIRED = ("torch", "torchvision")


def env_for(python_version: str, extra: str | None = None) -> dict[str, str]:
    env = {**AARCH64_ENV, "python_version": python_version, "python_full_version": f"{python_version}.0"}
    if extra is not None:
        env["extra"] = extra
    return env


def matches(marker: str | None, env: dict[str, str]) -> bool:
    return marker is None or Marker(marker).evaluate(env)


def supported_python_versions(requires_python: str) -> list[str]:
    """The `3.x` versions admitted by the lockfile's `requires-python` (e.g. ">=3.11, <3.13" -> 3.11, 3.12)."""
    spec = SpecifierSet(requires_python)
    return [f"3.{minor}" for minor in range(8, 30) if spec.contains(f"3.{minor}.0")]


def has_aarch64_wheel(package: dict[str, Any], python_version: str) -> bool:
    """Whether `package` ships a linux/aarch64 wheel usable by CPython `python_version`.

    Only linux aarch64 counts -- macOS arm64 wheels (`macosx_11_0_arm64`) are a different platform.
    """
    minor = int(python_version.split(".")[1])
    accepted = {f"cp3{minor}", "py3", f"py3{minor}"}
    for wheel in package.get("wheels", []):
        filename = wheel.get("url", wheel.get("path", "")).rsplit("/", 1)[-1]
        if not filename.endswith(".whl"):
            continue
        for tag in parse_wheel_filename(filename)[3]:
            if "aarch64" not in tag.platform:
                continue
            if tag.interpreter in accepted:
                return True
            # An abi3 wheel built for an older CPython also works on this one.
            if tag.abi == "abi3" and tag.interpreter.startswith("cp3") and tag.interpreter[3:].isdigit():
                if int(tag.interpreter[3:]) <= minor:
                    return True
    return False


def main() -> int:
    lock_path = Path(sys.argv[1] if len(sys.argv) > 1 else "uv.lock")
    if not lock_path.is_file():
        print(f"File not found: {lock_path}", file=sys.stderr)
        return 1

    text = lock_path.read_text()
    lock = tomllib.loads(text)

    python_versions = supported_python_versions(lock["requires-python"])
    packages = {(p["name"], p.get("version"), str(p.get("source"))): p for p in lock["package"]}
    problems: list[str] = []

    # 1. `tool.uv.environments` must still admit aarch64 -- if it doesn't, there is no aarch64 resolution
    #    to inspect and the per-extra checks below would report a confusing pile of missing torch.
    #    An empty `supported-markers` means uv was given no restriction, which is fine.
    markers = lock.get("supported-markers", [])
    if markers and not any(matches(m, env_for(v)) for m in markers for v in python_versions):
        print("uv.lock excludes linux/aarch64 entirely. supported-markers:")
        for m in markers:
            print(f"  {m}")
        print("\nWiden `tool.uv.environments` in pyproject.toml to include")
        print("(sys_platform == 'linux' and platform_machine == 'aarch64'), then re-run `uv lock`.")
        return 1

    # 2. Every conflicting torch extra must resolve torch and torchvision on aarch64, from a registry that
    #    actually has aarch64 wheels for them.
    root_name = "invokeai"
    root = next((p for p in lock["package"] if p["name"] == root_name), None)
    if root is None:
        print(f"no {root_name!r} package in {lock_path} -- has the project been renamed?")
        return 1
    extras = sorted({e["extra"] for group in lock.get("conflicts", []) for e in group if e["package"] == root_name})
    if not extras:
        print("no conflicting extras found in uv.lock -- has the cpu/cuda/rocm extra layout changed?")
        return 1

    # uv encodes conflicting extras into markers as `extra-<len(package)>-<package>-<extra>`. That is uv's
    # internal spelling, so confirm it is still what the lockfile uses -- otherwise the marker evaluation
    # below would silently match nothing and we would report a torch problem that isn't real. (A broken
    # extra can legitimately drop the token from its own markers, so look at the whole lockfile.)
    keys = {extra: f"extra-{len(root_name)}-{root_name}-{extra}" for extra in extras}
    unknown = sorted(k for k in keys.values() if k not in text)
    if unknown:
        print(f"uv.lock never mentions {', '.join(repr(k) for k in unknown)}.")
        print("uv's encoding of conflicting extras has changed; update scripts/check_aarch64_lock.py.")
        return 1

    for extra in extras:
        deps = root.get("optional-dependencies", {}).get(extra, [])
        key = keys[extra]
        for name in REQUIRED:
            for python_version in python_versions:
                env = env_for(python_version, extra=key)
                resolved = [d for d in deps if d["name"] == name and matches(d.get("marker"), env)]
                if not resolved:
                    problems.append(f"  [{extra}] py{python_version}: no {name} resolves on linux/aarch64")
                    continue
                dep = resolved[0]
                package = packages.get((dep["name"], dep.get("version"), str(dep.get("source"))))
                registry = (dep.get("source") or {}).get("registry", "?")
                if package is None:
                    problems.append(f"  [{extra}] py{python_version}: {name}=={dep.get('version')} not in uv.lock")
                elif not has_aarch64_wheel(package, python_version):
                    problems.append(
                        f"  [{extra}] py{python_version}: {name}=={dep.get('version')} from {registry}"
                        " has no linux/aarch64 wheel"
                    )
                else:
                    print(f"  [{extra}] py{python_version}: {name}=={dep.get('version')} from {registry}")

    if problems:
        print("\nuv.lock does not give linux/aarch64 an installable torch/torchvision:")
        print("\n".join(problems))
        print("\nCheck `tool.uv.environments`, the `[tool.uv.sources]` aarch64 markers and the aarch64")
        print("fallback pins in the cpu/cuda/rocm extras in pyproject.toml, then re-run `uv lock`.")
        return 1

    print("\naarch64 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
