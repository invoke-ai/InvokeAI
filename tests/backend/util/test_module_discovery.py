"""The package walker must descend into subpackages, and must not hide a broken one.

A walker that silently finds nothing is green forever, so these run it against a synthetic tree
whose expected result is written out by hand — independent of any real package's layout.
"""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from invokeai.backend.util.module_discovery import discover_modules


@pytest.fixture
def tree(tmp_path: Path) -> Iterator[Path]:
    """A miniature package: one flat module, one in a subpackage, plus things that must be skipped.

    `assets/` holds no modules and so is none of discovery's business; it is here because the
    missing-`__init__.py` check must not mistake an ordinary data directory for a broken package.
    """
    root = tmp_path / "synthetic_pkg"
    (root / "sub").mkdir(parents=True)
    (root / "__pycache__").mkdir()
    (root / "assets").mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "flat.py").write_text("", encoding="utf-8")
    (root / "_private.py").write_text("", encoding="utf-8")
    (root / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (root / "sub" / "nested.py").write_text("", encoding="utf-8")
    (root / "__pycache__" / "stale.py").write_text("", encoding="utf-8")
    (root / "assets" / "palette.json").write_text("{}", encoding="utf-8")

    sys.path.insert(0, str(tmp_path))
    try:
        yield root
    finally:
        sys.path.remove(str(tmp_path))
        for name in [m for m in sys.modules if m.startswith("synthetic_pkg")]:
            del sys.modules[name]


def test_descends_into_subpackages(tree: Path) -> None:
    assert sorted(discover_modules(tree, "synthetic_pkg.")) == [
        "synthetic_pkg.flat",
        "synthetic_pkg.sub.nested",
    ]


def test_excludes_named_modules(tree: Path) -> None:
    assert discover_modules(tree, "synthetic_pkg.", {"synthetic_pkg.sub.nested"}) == ["synthetic_pkg.flat"]


def test_skips_private_modules_and_pycache(tree: Path) -> None:
    found = discover_modules(tree, "synthetic_pkg.")
    assert not [n for n in found if "_private" in n or "__pycache__" in n or "stale" in n]


def test_an_excluded_package_is_neither_imported_nor_descended_into(tree: Path) -> None:
    """`walk_packages` imports a package to descend into it, so a `_disabled` package's initializer
    ran -- and could register nodes or abort startup -- before its name was ever filtered."""
    (tree / "_disabled").mkdir()
    (tree / "_disabled" / "__init__.py").write_text("raise RuntimeError('imported')", encoding="utf-8")
    (tree / "_disabled" / "node.py").write_text("", encoding="utf-8")

    assert sorted(discover_modules(tree, "synthetic_pkg.")) == ["synthetic_pkg.flat", "synthetic_pkg.sub.nested"]
    assert "synthetic_pkg._disabled" not in sys.modules


def test_a_broken_subpackage_fails_when_its_modules_are_imported(tree: Path) -> None:
    """Discovery imports nothing, so a broken initializer surfaces at the caller's import, with its
    own error, rather than as a module that quietly does not exist."""
    (tree / "sub" / "__init__.py").write_text("raise RuntimeError('boom')", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        for name in discover_modules(tree, "synthetic_pkg."):
            importlib.import_module(name)


def test_reports_a_directory_with_no_init(tree: Path) -> None:
    """The walk does not descend into one, and does not complain either.

    The result is a module that is simply never imported and a registration that silently never
    happens — the failure this whole module exists to make impossible.
    """
    (tree / "sub_noinit").mkdir()
    (tree / "sub_noinit" / "orphan.py").write_text("", encoding="utf-8")

    with pytest.raises(ImportError, match="sub_noinit"):
        discover_modules(tree, "synthetic_pkg.")


def test_reports_a_directory_with_no_init_nested_in_a_package(tree: Path) -> None:
    """The same mistake one level down, where it is easier to make and harder to spot."""
    (tree / "sub" / "deeper").mkdir()
    (tree / "sub" / "deeper" / "orphan.py").write_text("", encoding="utf-8")

    with pytest.raises(ImportError, match="sub/deeper"):
        discover_modules(tree, "synthetic_pkg.")


def test_a_directory_holding_no_modules_is_not_reported(tree: Path) -> None:
    """`assets/` and `__pycache__` have no `__init__.py` either, and neither is a mistake.

    Without this, the check above would fire on every data directory in the repository.
    """
    assert discover_modules(tree, "synthetic_pkg.")
