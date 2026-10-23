"""Assert that `uv.lock` gives the platforms the x64 test jobs cannot vouch for an installable torch stack.

Two platforms are covered:

- linux/aarch64. Three separate mechanisms in `pyproject.toml` conspire to make torch resolve from PyPI
  there instead of from the PyTorch WHL indexes (which ship no aarch64 torchvision wheel):
  `tool.uv.environments`, the `[tool.uv.sources]` platform markers, and the aarch64 fallback pins in each
  torch extra. Break the sources markers and `uv lock` fails loudly, but break either of the other two and
  the lockfile simply stops mentioning torch on aarch64 while `uv lock --locked` stays green -- which is how
  aarch64 support was silently lost once before, when a ROCm bump narrowed `tool.uv.environments`. No CI job
  installs on aarch64, so this check is its only coverage. Each extra must carry its own aarch64 torch edge:
  the base dependencies also resolve a torch there today, but that is incidental to how uv forks the
  resolution, so it must not be what keeps an extra installable.
- win32/ARM64 (Windows on ARM, NVIDIA RTX Spark). torch and torchvision must come from NVIDIA's
  out-of-tree index with `win_arm64` wheels whichever extra is selected, and -- because most of the
  platform's wheel gaps are papered over with version forks in `pyproject.toml` rather than enforced by
  uv -- every package in the platform's dependency closure must ship a `win_arm64` (or pure) wheel. uv's
  `required-environments` only enforces that for packages without an sdist; a package with an sdist and
  no ARM64 wheel locks fine and fails at install time on the user's machine. The windows-arm64 test job
  installs only the `test` extra; this check covers every torch extra and the install with none.

So assert against the lockfile itself, which is the artifact users actually install from.

Usage: check_platform_locks.py [path/to/uv.lock] [--platform {aarch64,win_arm64}]   (needs `packaging`)
"""

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

from packaging.markers import Marker
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

ROOT_NAME = "invokeai"
REQUIRED = ("torch", "torchvision")
NVIDIA_WIN_ARM64_REGISTRY = "https://pypi.nvidia.com/nvtorch_oot/"


@dataclass(frozen=True)
class Platform:
    name: str
    description: str
    # Interpreter facts for evaluating the lockfile's environment markers; `python_version` is added per check.
    env: dict[str, str]
    # A wheel tag this platform can install (pure `any` wheels are accepted separately).
    accepts_tag: Callable[[Tag], bool]
    # Restrict the Python versions checked (None: every version admitted by `requires-python`).
    python_versions: tuple[str, ...] | None = None
    # torch/torchvision must resolve from this registry (None: any registry with acceptable wheels).
    torch_registry: str | None = None
    # Also check torch/torchvision with no extra selected (platforms whose torch does not come from an extra).
    check_no_extra: bool = False
    # Count the base dependencies' torch edges toward each extra's torch (False: the extra must carry its own).
    torch_from_base: bool = True
    # Walk the whole closure for wheel availability.
    check_closure: bool = False
    # Non-conflicting extras whose closure is walked too (what the platform's CI lane installs).
    closure_extras: tuple[str, ...] = ()
    # Packages allowed to be sdist-only in the closure (pure Python, built without a compiler).
    sdist_allowlist: frozenset[str] = field(default_factory=frozenset)


PLATFORMS: dict[str, Platform] = {
    "aarch64": Platform(
        name="linux/aarch64",
        description="torch and torchvision from a registry with aarch64 wheels, per extra",
        env={
            "sys_platform": "linux",
            "platform_machine": "aarch64",
            "platform_system": "Linux",
            "os_name": "posix",
            "implementation_name": "cpython",
            "platform_python_implementation": "CPython",
        },
        # Only linux aarch64 counts -- macOS arm64 wheels (`macosx_11_0_arm64`) are a different platform.
        accepts_tag=lambda tag: "aarch64" in tag.platform,
        torch_from_base=False,
    ),
    "win_arm64": Platform(
        name="win32/ARM64",
        description="torch and torchvision from NVIDIA's index, and win_arm64 wheels for the whole closure",
        env={
            "sys_platform": "win32",
            "platform_machine": "ARM64",
            "platform_system": "Windows",
            "os_name": "nt",
            "implementation_name": "cpython",
            "platform_python_implementation": "CPython",
        },
        accepts_tag=lambda tag: tag.platform == "win_arm64",
        # The only supported interpreter there: PyWavelets and PyYAML ship win_arm64 wheels for 3.12+ only.
        python_versions=("3.12",),
        torch_registry=NVIDIA_WIN_ARM64_REGISTRY,
        check_no_extra=True,
        check_closure=True,
        closure_extras=("test",),
        # imageio-ffmpeg's wheels exist only to carry an ffmpeg binary; the sdist builds a pure wheel and
        # the app uses the ffmpeg on PATH. requests-testadapter (test extra) is a pure-Python sdist.
        sdist_allowlist=frozenset({"imageio-ffmpeg", "requests-testadapter"}),
    ),
}


def env_for(platform: Platform, python_version: str, extra: str | None = None) -> dict[str, str]:
    env = {**platform.env, "python_version": python_version, "python_full_version": f"{python_version}.0"}
    # `extra` must be defined for the `extra != '...'` clauses uv writes; "" means no extra selected.
    env["extra"] = extra if extra is not None else ""
    return env


def matches(marker: str | None, env: dict[str, str]) -> bool:
    return marker is None or Marker(marker).evaluate(env)


def supported_python_versions(requires_python: str) -> list[str]:
    """The `3.x` versions admitted by the lockfile's `requires-python`.

    The specifier may admit several versions - `">=3.10, <3.13"` yields 3.10, 3.11 and 3.12 -
    even where the repo's own bound currently admits exactly one.
    """
    spec = SpecifierSet(requires_python)
    return [f"3.{minor}" for minor in range(8, 30) if spec.contains(f"3.{minor}.0")]


def wheel_filenames(package: dict[str, Any]) -> list[str]:
    names = []
    for wheel in package.get("wheels", []):
        # The PyTorch WHL indexes percent-encode the `+` of local versions in wheel URLs (`torch-2.7.1%2Bcpu-...`).
        filename = unquote(wheel.get("url", wheel.get("path", "")).rsplit("/", 1)[-1])
        if filename.endswith(".whl"):
            names.append(filename)
    return names


def has_wheel(package: dict[str, Any], python_version: str, platform: Platform, allow_pure: bool) -> bool:
    """Whether `package` ships a wheel CPython `python_version` can install on `platform`."""
    minor = int(python_version.split(".")[1])
    accepted = {f"cp3{minor}", "py3", f"py3{minor}"}
    for filename in wheel_filenames(package):
        try:
            tags = parse_wheel_filename(filename)[3]
        except InvalidWheelFilename as e:
            # The exception message names the offending filename.
            sys.exit(f"unparseable wheel filename in uv.lock ({e}) -- update scripts/check_platform_locks.py")
        for tag in tags:
            if not (platform.accepts_tag(tag) or (allow_pure and tag.platform == "any")):
                continue
            if tag.interpreter in accepted:
                return True
            # An abi3 wheel built for an older CPython also works on this one.
            if tag.abi == "abi3" and tag.interpreter.startswith("cp3") and tag.interpreter[3:].isdigit():
                if int(tag.interpreter[3:]) <= minor:
                    return True
    return False


class Lock:
    def __init__(self, path: Path) -> None:
        self.text = path.read_text()
        self.data = tomllib.loads(self.text)
        self.packages = {(p["name"], p.get("version"), str(p.get("source"))): p for p in self.data["package"]}
        self.by_name: dict[str, list[dict[str, Any]]] = {}
        for p in self.data["package"]:
            self.by_name.setdefault(p["name"], []).append(p)

    def resolve(self, dep: dict[str, Any]) -> list[dict[str, Any]]:
        """The lock package entries a dependency edge refers to."""
        if dep.get("version") is not None:
            package = self.packages.get((dep["name"], dep["version"], str(dep.get("source"))))
            return [package] if package is not None else []
        # uv omits version/source from a dependency entry when the package resolves to a single version
        # across the whole lockfile.
        candidates = self.by_name.get(dep["name"], [])
        return candidates if len(candidates) == 1 else []


def universes(
    root: dict[str, Any], keys: dict[str, str], include_no_extra: bool, include_base: bool = True
) -> list[tuple[str, str | None, list[dict[str, Any]]]]:
    """(label, uv extra marker token, dependency edges) for each resolution universe to check.

    With `include_base` False, an extra's universe holds only the extra's own edges.
    """
    base = root.get("dependencies", [])
    extra_base = base if include_base else []
    result: list[tuple[str, str | None, list[dict[str, Any]]]] = []
    if include_no_extra:
        result.append(("no extra", None, base))
    result += [
        (extra, keys[extra], extra_base + root.get("optional-dependencies", {}).get(extra, [])) for extra in keys
    ]
    return result


def check_torch(
    lock: Lock, root: dict[str, Any], platform: Platform, python_versions: list[str], keys: dict[str, str]
) -> list[str]:
    """torch/torchvision resolve, per extra (and with no extra where the platform asks), from acceptable wheels."""
    problems: list[str] = []
    checked = universes(root, keys, include_no_extra=platform.check_no_extra, include_base=platform.torch_from_base)
    for label, key, deps in checked:
        for name in REQUIRED:
            for python_version in python_versions:
                env = env_for(platform, python_version, extra=key)
                resolved = [d for d in deps if d["name"] == name and matches(d.get("marker"), env)]
                if not resolved:
                    problems.append(f"  [{label}] py{python_version}: no {name} resolves on {platform.name}")
                    continue
                # The base and the extra each carry an edge to the same package wherever the extra pins what the
                # base already resolves, so the matches are checked per package, not per edge. Distinct packages
                # (overlapping markers, which uv shouldn't emit) are each checked.
                reported: set[tuple[str, Any, str]] = set()
                for dep in resolved:
                    packages = lock.resolve(dep)
                    version = dep.get("version") or (packages[0].get("version") if packages else "?")
                    if not packages:
                        problems.append(f"  [{label}] py{python_version}: {name}=={version} not in uv.lock")
                        continue
                    for package in packages:
                        ident = (package["name"], package.get("version"), str(package.get("source")))
                        if ident in reported:
                            continue
                        reported.add(ident)
                        registry = (package.get("source") or {}).get("registry", "?")
                        where = f"{name}=={package.get('version')} from {registry}"
                        if platform.torch_registry is not None and registry != platform.torch_registry:
                            problems.append(
                                f"  [{label}] py{python_version}: {where} -- expected {platform.torch_registry}"
                            )
                        elif not has_wheel(package, python_version, platform, allow_pure=False):
                            problems.append(f"  [{label}] py{python_version}: {where} has no {platform.name} wheel")
                        else:
                            print(f"  [{label}] py{python_version}: {where}")
    return problems


def walk_closure(
    lock: Lock, edges: list[dict[str, Any]], env: dict[str, str]
) -> tuple[dict[tuple[str, Any, str], dict[str, Any]], list[str]]:
    """Every lock package reachable from `edges` in environment `env`, keyed by (name, version, source), plus
    the edges that name no package in the lock (a corrupt lock would otherwise shrink the closure silently)."""
    seen: dict[tuple[str, Any, str], dict[str, Any]] = {}
    unresolved: list[str] = []
    # A package's extras are walked per (package, extra), independently of whether the package itself was
    # already reached through a plain edge -- `P` and then `P[x]` must still enqueue `x`'s dependencies.
    seen_extras: set[tuple[tuple[str, Any, str], str]] = set()
    pending = list(edges)
    while pending:
        dep = pending.pop()
        if not matches(dep.get("marker"), env):
            continue
        packages = lock.resolve(dep)
        if not packages:
            unresolved.append(f"{dep['name']}=={dep.get('version', '?')}")
        for package in packages:
            ident = (package["name"], package.get("version"), str(package.get("source")))
            if ident not in seen:
                seen[ident] = package
                pending += package.get("dependencies", [])
            for extra in dep.get("extra", []):
                if (ident, extra) not in seen_extras:
                    seen_extras.add((ident, extra))
                    pending += package.get("optional-dependencies", {}).get(extra, [])
    return seen, unresolved


def check_closure(
    lock: Lock, root: dict[str, Any], platform: Platform, python_version: str, keys: dict[str, str]
) -> list[str]:
    """Every package reachable on the platform (any single extra, or none) ships an installable wheel."""
    problems: list[str] = []
    base = root.get("dependencies", [])
    checked = universes(root, keys, include_no_extra=True)
    checked += [
        (extra, None, base + root.get("optional-dependencies", {}).get(extra, [])) for extra in platform.closure_extras
    ]
    for label, key, deps in checked:
        seen, unresolved = walk_closure(lock, deps, env_for(platform, python_version, extra=key))
        problems += [f"  [{label}] py{python_version}: dependency {edge} is not in uv.lock" for edge in unresolved]
        missing = [
            f"{name}=={version} from {(p.get('source') or {}).get('registry', '?')}"
            for (name, version, _), p in sorted(seen.items())
            if name not in platform.sdist_allowlist and not has_wheel(p, python_version, platform, allow_pure=True)
        ]
        problems += [f"  [{label}] py{python_version}: {m} has no {platform.name} wheel" for m in missing]
        print(f"  [{label}] py{python_version}: {len(seen)} packages in the closure, {len(missing)} without a wheel")
    return problems


def check_platform(lock: Lock, key: str) -> int:
    platform = PLATFORMS[key]
    print(f"{platform.name}: {platform.description}")
    python_versions = list(platform.python_versions or supported_python_versions(lock.data["requires-python"]))

    # 1. `tool.uv.environments` must still admit the platform -- if it doesn't, there is no resolution to
    #    inspect and the per-extra checks below would report a confusing pile of missing torch.
    #    An empty `supported-markers` means uv was given no restriction, which is fine.
    markers = lock.data.get("supported-markers", [])
    if markers and not any(matches(m, env_for(platform, v)) for m in markers for v in python_versions):
        print(f"uv.lock excludes {platform.name} entirely. supported-markers:")
        for m in markers:
            print(f"  {m}")
        print("\nWiden `tool.uv.environments` in pyproject.toml to include the platform, then re-run `uv lock`.")
        return 1

    root = next((p for p in lock.data["package"] if p["name"] == ROOT_NAME), None)
    if root is None:
        print(f"no {ROOT_NAME!r} package in uv.lock -- has the project been renamed?")
        return 1
    extras = sorted(
        {e["extra"] for group in lock.data.get("conflicts", []) for e in group if e["package"] == ROOT_NAME}
    )
    if not extras:
        print("no conflicting extras found in uv.lock -- has the cpu/cuda/rocm extra layout changed?")
        return 1

    # uv encodes conflicting extras into markers as `extra-<len(package)>-<package>-<extra>`. That is uv's
    # internal spelling, so confirm it is still what the lockfile uses -- otherwise the marker evaluation
    # below would silently match nothing and we would report a torch problem that isn't real. (A broken
    # extra can legitimately drop the token from its own markers, so look at the whole lockfile.)
    keys = {extra: f"extra-{len(ROOT_NAME)}-{ROOT_NAME}-{extra}" for extra in extras}
    unknown = sorted(k for k in keys.values() if k not in lock.text)
    if unknown:
        print(f"uv.lock never mentions {', '.join(repr(k) for k in unknown)}.")
        print("uv's encoding of conflicting extras has changed; update scripts/check_platform_locks.py.")
        return 1

    problems = check_torch(lock, root, platform, python_versions, keys)
    if platform.check_closure:
        for python_version in python_versions:
            problems += check_closure(lock, root, platform, python_version, keys)

    if problems:
        print(f"\nuv.lock does not give {platform.name} an installable environment:")
        print("\n".join(problems))
        print("\nCheck `tool.uv.environments`, the `[tool.uv.sources]` platform markers and the platform")
        print("fallback pins in pyproject.toml, then re-run `uv lock`.")
        return 1

    print(f"\n{platform.name} OK\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check uv.lock for platforms the x64 test jobs cannot vouch for.")
    parser.add_argument("lock", nargs="?", default="uv.lock", type=Path)
    parser.add_argument("--platform", choices=sorted(PLATFORMS), action="append", help="default: all")
    args = parser.parse_args(argv)
    if not args.lock.is_file():
        print(f"File not found: {args.lock}", file=sys.stderr)
        return 1
    lock = Lock(args.lock)
    return max(check_platform(lock, key) for key in (args.platform or sorted(PLATFORMS)))


if __name__ == "__main__":
    sys.exit(main())
