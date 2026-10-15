"""The architecture package has an internal direction of dependency; this pins it.

    facet.py <- registry.py <- facets/* <- defs/* <- __init__.py <- the rest of the codebase

Two of these edges are load-bearing rather than tidy. `defs/*` must import `architectures.registry`
directly and never the `architectures` package, because it is that package's own import that brings
the defs modules into being — reaching back for an attribute would find a half-initialized module.
And everything outside must go through the facade, so that importing the registry always means the
registry has been filled.

Modelled on invokeai/frontend/webv2/src/architecture/dependencyPolicy.test.ts: named rules, a single
assertion listing every violation at once, and self-tests proving the checker actually catches
things. The self-tests are the important half — an AST walker with a bug reports no violations and
stays green forever, which is indistinguishable from a codebase that obeys the rules.
"""

import ast
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
ARCH = "invokeai.backend.architectures"
ARCH_DIR = "invokeai/backend/architectures"
TAXONOMY = "invokeai.backend.model_manager.taxonomy"
DISCOVERY = "invokeai.backend.util.module_discovery"
# The conditioning facet names the `*ConditioningInfo` classes themselves, and so do the defs
# that declare them. Added deliberately rather than pre-emptively: each widening of these lists
# is a reviewable line in the change that needs it.
CONDITIONING = "invokeai.backend.stable_diffusion.diffusion.conditioning_data"
# `MainModelDefaultSettings` lives in a leaf module of its own precisely so this edge is safe:
# `configs/main.py` looks the values up through the registry, so a facet importing *that* would
# close a cycle.
DEFAULT_SETTINGS = "invokeai.backend.model_manager.configs.default_settings"

# Vendored third-party trees, mirroring [tool.ruff] exclude, plus the frontend.
EXCLUDED = (
    "invokeai/backend/image_util/mediapipe_face",
    "invokeai/backend/image_util/mlsd",
    "invokeai/backend/image_util/normal_bae",
    "invokeai/backend/image_util/pidi",
    "invokeai/backend/image_util/imwatermark",
    "invokeai/frontend",
)


def _allowed_for(path: str) -> tuple[str, frozenset[str], tuple[str, ...]] | None:
    """The rule governing `path`: its name, the exact imports it may make, and allowed prefixes.

    Returns None for a file no rule covers, which is most of the repository — those are only
    subject to `aggregate-only` below.
    """
    if path == f"{ARCH_DIR}/facet.py":
        return "facet-is-a-leaf", frozenset(), ()
    if path == f"{ARCH_DIR}/registry.py":
        return "registry-is-a-leaf", frozenset({f"{ARCH}.facet", TAXONOMY}), ()
    if path == f"{ARCH_DIR}/capabilities.py":
        # The one module that reads across facets: it renders the table the API serves. It may
        # name every facet, but still not the aggregate — it is imported by it.
        return (
            "capabilities-allowlist",
            frozenset({f"{ARCH}.facet", f"{ARCH}.facets", f"{ARCH}.registry", TAXONOMY, DEFAULT_SETTINGS}),
            (f"{ARCH}.facets.",),
        )
    if path == f"{ARCH_DIR}/__init__.py":
        return "aggregate-is-a-facade", frozenset({ARCH}), (f"{ARCH}.",)
    if path.startswith(f"{ARCH_DIR}/facets/"):
        return (
            "facets-allowlist",
            frozenset({f"{ARCH}.facet", f"{ARCH}.registry", TAXONOMY, DISCOVERY, CONDITIONING, DEFAULT_SETTINGS}),
            (),
        )
    if path.startswith(f"{ARCH_DIR}/defs/"):
        return (
            "defs-allowlist",
            frozenset(
                {
                    f"{ARCH}.facet",
                    f"{ARCH}.facets",
                    f"{ARCH}.registry",
                    TAXONOMY,
                    DISCOVERY,
                    CONDITIONING,
                    DEFAULT_SETTINGS,
                }
            ),
            (f"{ARCH}.facets.",),
        )
    return None


def _imported_modules(tree: ast.AST) -> set[str]:
    """Every `invokeai.*` module the file depends on, plus a marker for any relative import.

    `from a.b import c` yields `a.b`, and also `a.b.c` when that is a module on disk — otherwise
    `from ...defs import wan` would be invisible while `import ...defs.wan` was not. The on-disk
    check is what keeps `from ...registry import require` from being read as a module import.

    Imports guarded by `if TYPE_CHECKING:` count. A type-only edge is still an architecture edge:
    it is the reason a module cannot later be moved or split.
    """
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("invokeai."))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                modules.add("<relative>")
                continue
            if not node.module or not node.module.startswith("invokeai."):
                continue
            modules.add(node.module)
            for alias in node.names:
                candidate = f"{node.module}.{alias.name}"
                relative = candidate.replace(".", "/")
                if (REPO_ROOT / f"{relative}.py").exists() or (REPO_ROOT / relative / "__init__.py").exists():
                    modules.add(candidate)
    return modules


def _violations(path: str, source: str) -> list[str]:
    """Rule violations in one file, as `rule-name: importer -> imported` strings."""
    dotted = path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
    rule = _allowed_for(path)
    found: list[str] = []

    for imported in sorted(_imported_modules(ast.parse(source))):
        if rule is None:
            # No rule covers this file, so it is outside the package: the facade or nothing. The
            # relative-import ban is deliberately not applied here — ruff's TID252 owns the rest of
            # the repository, and duplicating it would make this test fail on pre-existing debt that
            # has nothing to do with the architecture package.
            if imported.startswith(f"{ARCH}."):
                found.append(f"aggregate-only: {dotted} -> {imported}")
        elif imported == "<relative>":
            found.append(f"no-relative-imports: {dotted}")
        else:
            name, exact, prefixes = rule
            if imported not in exact and not imported.startswith(prefixes):
                found.append(f"{name}: {dotted} -> {imported}")
    return found


def _production_files() -> Iterable[tuple[str, str]]:
    for p in sorted((REPO_ROOT / "invokeai").rglob("*.py")):
        path = p.relative_to(REPO_ROOT).as_posix()
        if path.startswith(EXCLUDED):
            continue
        yield path, p.read_text(encoding="utf-8")


def test_the_walk_reaches_the_files_the_rules_are_about() -> None:
    """The self-tests below prove `_violations` catches things. They say nothing about whether it is
    ever handed anything: one more prefix in `EXCLUDED`, or an `rglob` that stops matching, and
    `test_no_layering_violations` iterates an empty sequence and passes forever."""
    scanned = {path for path, _ in _production_files()}

    assert len(scanned) > 100
    assert {
        f"{ARCH_DIR}/registry.py",
        f"{ARCH_DIR}/defs/wan.py",
        f"{ARCH_DIR}/facets/latent_space.py",
        "invokeai/app/util/step_callback.py",
    } <= scanned


def test_no_layering_violations() -> None:
    violations = sorted(v for path, source in _production_files() for v in _violations(path, source))
    assert violations == []


# --- self-tests: does the checker actually catch anything? ----------------------------------------


def test_catches_defs_reaching_into_core() -> None:
    v = _violations(f"{ARCH_DIR}/defs/wan.py", "from invokeai.app.util.step_callback import calc_percentage\n")
    assert v == [f"defs-allowlist: {ARCH}.defs.wan -> invokeai.app.util.step_callback"]


def test_catches_defs_importing_the_aggregate() -> None:
    """The circular-import trap: `defs` is imported *by* the package it would be reaching into."""
    v = _violations(f"{ARCH_DIR}/defs/wan.py", "from invokeai.backend.architectures import register\n")
    assert v == [f"defs-allowlist: {ARCH}.defs.wan -> {ARCH}"]


def test_catches_core_bypassing_the_facade() -> None:
    v = _violations("invokeai/app/util/step_callback.py", f"from {ARCH}.registry import require\n")
    assert v == [f"aggregate-only: invokeai.app.util.step_callback -> {ARCH}.registry"]


def test_catches_a_type_checking_only_edge() -> None:
    source = "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from invokeai.app.invocations import model\n"
    assert _violations(f"{ARCH_DIR}/registry.py", source) == [
        f"registry-is-a-leaf: {ARCH}.registry -> invokeai.app.invocations",
        f"registry-is-a-leaf: {ARCH}.registry -> invokeai.app.invocations.model",
    ]


def test_catches_a_relative_import() -> None:
    assert _violations(f"{ARCH_DIR}/defs/wan.py", "from ..registry import register\n") == [
        f"no-relative-imports: {ARCH}.defs.wan"
    ]


def test_catches_registry_importing_a_facet() -> None:
    """`facet.py` is a separate module precisely so this edge cannot exist.

    Note the reported module is `...facets.latent_space`, not `...facets`: `from a.b.c import X`
    depends on `a.b.c`. The one-character difference between `facet` (allowed) and `facets`
    (forbidden) is the whole point of the rule, so the assertion spells the name out in full.
    """
    v = _violations(f"{ARCH_DIR}/registry.py", f"from {ARCH}.facets.latent_space import LatentSpaceFacet\n")
    assert v == [f"registry-is-a-leaf: {ARCH}.registry -> {ARCH}.facets.latent_space"]


def test_registry_may_import_facet_but_not_facets() -> None:
    """Pins the one-character distinction from both sides."""
    assert _violations(f"{ARCH_DIR}/registry.py", f"from {ARCH}.facet import Facet\n") == []
    assert _violations(f"{ARCH_DIR}/registry.py", f"from {ARCH}.facets import something\n") == [
        f"registry-is-a-leaf: {ARCH}.registry -> {ARCH}.facets"
    ]


def test_allows_the_legitimate_cases() -> None:
    """A checker that flagged everything would also pass every test above."""
    assert _violations(f"{ARCH_DIR}/defs/wan.py", f"from {ARCH}.registry import register\n") == []
    assert _violations(f"{ARCH_DIR}/registry.py", f"from {TAXONOMY} import BaseModelType\n") == []
    assert _violations("invokeai/app/util/step_callback.py", f"from {ARCH} import require\n") == []
    assert _violations("invokeai/app/util/step_callback.py", "import torch\nfrom PIL import Image\n") == []
