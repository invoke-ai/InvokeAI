"""The starter catalogue is a package now; these pin what the split must not change.

`STARTER_MODELS` is curated product data — the order the install dialog shows, decided by someone,
reconstructible from nothing. It stays written out in `__init__.py` rather than assembled from the
per-architecture modules, and the first test here is what stops a later tidy-up from sorting it.
"""

import ast
import importlib
import pkgutil
from pathlib import Path

import invokeai.backend.model_manager.starter_models as starters
from invokeai.backend.model_manager.starter_models import STARTER_BUNDLES, STARTER_MODELS
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import BaseModelType

PACKAGE_DIR = Path(starters.__file__).parent

# One module per architecture, plus the three that are not architectures.
NOT_AN_ARCHITECTURE = {"types", "common", "external"}


def test_the_curated_order_is_not_a_sorted_one() -> None:
    """If this ever passes by accident, the order has been replaced by a derivable one and the
    curation is gone. It is the sequence of the install dialog, and nothing can rebuild it."""
    names = [m.name for m in STARTER_MODELS]
    assert names != sorted(names), "STARTER_MODELS looks sorted — the curated order was lost"


def test_no_model_is_listed_twice() -> None:
    sources = [m.source for m in STARTER_MODELS]
    duplicates = sorted({s for s in sources if sources.count(s) > 1})
    assert duplicates == []


def test_each_architecture_module_holds_one_architecture() -> None:
    """A module that mixes bases means the split has drifted and the file names stop meaning
    anything. `common` and `external` are exempt by definition.

    Read from the source rather than the module namespace: three modules legitimately *import* a
    model from another architecture — Krea-2 shares Qwen-Image's VAE, Z-Image the FLUX one, and the
    refiner SDXL's — and an imported name is not a name this module defines.
    """
    mixed = []
    for info in pkgutil.iter_modules([str(PACKAGE_DIR)]):
        if info.name in NOT_AN_ARCHITECTURE:
            continue
        tree = ast.parse((PACKAGE_DIR / f"{info.name}.py").read_text(encoding="utf-8"))
        declared = set()
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "BaseModelType"
                ):
                    declared.add(sub.attr)
        declared -= {"Any"}
        if len(declared) > 1:
            mixed.append(f"{info.name}: {sorted(declared)}")
    assert mixed == []


def test_every_bundle_is_reachable_from_the_catalogue() -> None:
    """A bundle listing a model that is not in STARTER_MODELS would offer an install the dialog
    cannot show."""
    catalogue = {m.source for m in STARTER_MODELS}
    orphaned = sorted(
        f"{base.value}/{model.name}"
        for base, bundle in STARTER_BUNDLES.items()
        for model in bundle.models
        if model.source not in catalogue
    )
    assert orphaned == []


def test_the_package_splits_along_the_lines_it_claims() -> None:
    """Fifteen architecture modules plus types, common and external. Named so a contributor adding
    an architecture knows which file to open without reading any of them."""
    modules = {info.name for info in pkgutil.iter_modules([str(PACKAGE_DIR)])}
    assert NOT_AN_ARCHITECTURE < modules
    architecture_modules = modules - NOT_AN_ARCHITECTURE
    # Every architecture module is named for a base, using the same convention as
    # `architectures/defs/`: the base value with `-` replaced by `_`.
    known = {b.value.replace("-", "_") for b in BaseModelType}
    assert architecture_modules <= known, sorted(architecture_modules - known)


# Defined but reachable from nothing: not in `STARTER_MODELS`, not in a bundle, and not a dependency
# of anything that is. Both pre-date the package split — they are recorded rather than removed
# because deleting curated product data is a product decision, not a refactor's. The point of the
# list is that it is closed: a *new* orphan fails the test below instead of joining them silently.
KNOWN_UNREACHABLE = {
    "ESRGAN_SRx4_DF2KOST_official",
    "FLUX.1 Kontext dev",
}


def test_every_defined_model_is_reachable() -> None:
    """A `StarterModel` defined in a submodule but never listed is invisible to users.

    Before the split this was one file, where an unreferenced definition was at least visible while
    scrolling past it. Across nineteen modules nothing shows it — the dialog simply never offers the
    model, which looks like it was never added.
    """
    reachable = {id(m) for m in STARTER_MODELS}
    # `bundle.models`, not `bundle`: `StarterModelBundle` is a pydantic model, so iterating it
    # yields ('name', ...) field tuples whose ids belong to nothing. The set stayed correct only
    # because every bundle entry is today the same object as its STARTER_MODELS row.
    bundled = {id(m) for bundle in STARTER_BUNDLES.values() for m in bundle.models}
    assert bundled, "no bundle contributed a model; the reachability set below would be too small"
    reachable |= bundled
    # One pass is enough: dependencies are leaves, and none declares dependencies of its own.
    reachable |= {id(d) for m in STARTER_MODELS for d in (m.dependencies or [])}

    orphans = set()
    for info in pkgutil.iter_modules([str(PACKAGE_DIR)]):
        module = importlib.import_module(f"{starters.__name__}.{info.name}")
        for name, value in vars(module).items():
            if isinstance(value, StarterModel) and not name.startswith("_") and id(value) not in reachable:
                orphans.add(value.name)

    assert sorted(orphans - KNOWN_UNREACHABLE) == []
    assert sorted(KNOWN_UNREACHABLE - orphans) == [], (
        "a known-unreachable model became reachable; drop it from the list"
    )
