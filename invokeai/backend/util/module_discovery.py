"""Import-for-side-effect discovery of the modules in a package tree.

Two registries in this codebase are filled by importing modules rather than from a hand-maintained
list: node invocations (`invokeai.app.invocations`) and model architectures
(`invokeai.backend.architectures.defs`). Both fail the same way when discovery is subtly wrong —
they find nothing, register nothing, and stay green — so both go through this one function, and its
pitfalls are handled and tested in one place.
"""

import pkgutil
from collections.abc import Collection, Iterator
from pathlib import Path


def _holds_modules(directory: Path) -> bool:
    """Whether `directory` contains any module that discovery would be expected to reach.

    Ignores `_`-prefixed path components, so a stale `__pycache__` does not make an ordinary data
    directory look like a package someone forgot to finish.
    """
    return any(
        not any(part.startswith("_") for part in module.relative_to(directory).parts)
        for module in directory.rglob("*.py")
    )


def _orphan_directories(package_dir: Path) -> Iterator[Path]:
    """Directories under `package_dir` that hold modules but have no `__init__.py`.

    The walk does not descend into such a directory, and says nothing about it: every
    module below it is simply absent from the result, which is indistinguishable from there being
    nothing to find. That is the discovery mistake that actually happens — one `__init__.py`
    forgotten in a new subdirectory — so it is looked for rather than waited for.
    """
    for child in sorted(package_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "__init__.py").exists():
            yield from _orphan_directories(child)
        elif _holds_modules(child):
            yield child


def discover_modules(root: Path, prefix: str, excluded_modules: Collection[str] | None = None) -> list[str]:
    """Fully-qualified names of every non-private module in the package tree rooted at `root`.

    `prefix` is the dotted path of the package that lives at `root`, trailing dot included; it is
    what the returned names are prefixed with. `excluded_modules` contains fully-qualified module
    names to omit from the result.

    A path component starting with `_` excludes the module and everything below it: that covers
    `__pycache__` and marks a module or package as internal. Packages themselves are not returned --
    importing a module imports its packages, and it is their contents that carry the registrations.

    A directory holding modules but no `__init__.py` raises: the walk would skip it in silence,
    which is the one discovery failure that produces no symptom at all.

    Names only, and nothing is imported to find them: `pkgutil.walk_packages` imports every package
    to descend into it, which runs an excluded package's `__init__.py` before any filter can see its
    name. Importing is the caller's job, so a broken package fails there, with its own traceback.
    Keeping the two apart is also what lets the walk be tested against a synthetic tree, which
    matters because a walker with a bug returns an empty list and no test that merely asserts "some
    modules were found" would notice.
    """
    orphans = [d.relative_to(root).as_posix() for d in _orphan_directories(root)]
    if orphans:
        raise ImportError(
            f"These directories under {root} hold modules but no __init__.py, so nothing in them is "
            f"discovered and nothing registers: {', '.join(orphans)}. Add an __init__.py to each, or "
            f"rename it with a leading underscore if it is not meant to be imported."
        )

    return list(_walk(root, prefix, excluded_modules or set()))


def _walk(directory: Path, prefix: str, excluded_modules: Collection[str]) -> Iterator[str]:
    for info in pkgutil.iter_modules([str(directory)]):
        if info.name.startswith("_"):
            continue
        if info.ispkg:
            yield from _walk(directory / info.name, f"{prefix}{info.name}.", excluded_modules)
        else:
            module_name = f"{prefix}{info.name}"
            if module_name not in excluded_modules:
                yield module_name
