"""The registry covers every architecture, and the files on disk agree with it.

These run against the *real* registry, unlike test_registry.py. They are what makes the boot check
meaningful before any facet is required: with no required facets, `validate()` still has to catch a
`BaseModelType` member that nobody declared.
"""

from pathlib import Path

from invokeai.backend.architectures import facets, generative_bases, validate
from invokeai.backend.architectures.registry import _NOT_ARCHITECTURES, defs_module_path
from invokeai.backend.model_manager.taxonomy import BaseModelType

REPO_ROOT = Path(__file__).parents[3]
DEFS_DIR = REPO_ROOT / "invokeai" / "backend" / "architectures" / "defs"


def test_every_architecture_is_registered() -> None:
    """The one place the sentinel subtraction is written down.

    Production code never computes `set(BaseModelType) - sentinels` to decide what is generative —
    being registered is what makes an architecture generative. This asserts the two agree, which is
    what catches a new enum member whose `defs/` module was never written.
    """
    assert generative_bases() == set(BaseModelType) - _NOT_ARCHITECTURES


def test_each_registered_architecture_has_the_file_its_errors_name() -> None:
    """`defs_module_path` is quoted in every error message; a wrong path is worse than no path."""
    missing = [
        defs_module_path(base) for base in generative_bases() if not (REPO_ROOT / defs_module_path(base)).exists()
    ]
    assert missing == []


def test_no_defs_module_is_left_over() -> None:
    """The other direction: a file for a base that no longer exists would never be noticed.

    Discovery imports whatever is in the directory, and a stale module's `register()` call would
    either raise on an unknown enum member or, worse, keep registering a base that was renamed.
    """
    on_disk = {p.stem for p in DEFS_DIR.glob("*.py") if not p.stem.startswith("_")}
    derived = {base.value.replace("-", "_") for base in BaseModelType if base not in _NOT_ARCHITECTURES}
    assert on_disk == derived


def test_every_facet_module_was_imported() -> None:
    """`validate()` only checks requirements it knows about, and it learns them at import time."""
    facets_dir = REPO_ROOT / "invokeai" / "backend" / "architectures" / "facets"
    on_disk = {
        f"invokeai.backend.architectures.facets.{p.stem}"
        for p in facets_dir.rglob("*.py")
        if not any(part.startswith("_") for part in p.relative_to(facets_dir).parts)
    }
    assert on_disk == set(facets._MODULES)


def test_validate_passes() -> None:
    validate()
