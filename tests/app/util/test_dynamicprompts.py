from __future__ import annotations

from pathlib import Path

import pytest

from invokeai.app.util.dynamicprompts import find_missing_wildcards, get_wildcard_manager


@pytest.fixture
def wildcards_dir(tmp_path: Path) -> Path:
    """A wildcards directory containing a single `animals` collection."""
    (tmp_path / "animals.txt").write_text("cat\ndog\nbird\n", encoding="utf-8")
    return tmp_path


def test_get_wildcard_manager_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "does-not-exist-yet"
    assert not target.exists()
    get_wildcard_manager(target)
    assert target.is_dir()


def test_find_missing_wildcards_detects_unknown_wildcard_in_variant(wildcards_dir: Path) -> None:
    # Regression: `__random__` inside a variant is parsed as a wildcard reference. Left unchecked it
    # sends the combinatorial generator into an infinite loop, so it must be reported up front.
    wm = get_wildcard_manager(wildcards_dir)
    assert find_missing_wildcards("{__random__8chan|fenster|stuff}", wm) == ["random"]


def test_find_missing_wildcards_passes_known_wildcard(wildcards_dir: Path) -> None:
    wm = get_wildcard_manager(wildcards_dir)
    assert find_missing_wildcards("a {__animals__|house}", wm) == []


@pytest.mark.parametrize("prompt", ["plain text", "{a|b|c}", "a {2$$x|y|z}"])
def test_find_missing_wildcards_ignores_prompts_without_wildcards(wildcards_dir: Path, prompt: str) -> None:
    wm = get_wildcard_manager(wildcards_dir)
    assert find_missing_wildcards(prompt, wm) == []


def test_find_missing_wildcards_dedupes_repeated_unknown_wildcards(wildcards_dir: Path) -> None:
    wm = get_wildcard_manager(wildcards_dir)
    assert find_missing_wildcards("__nope__ and __nope__ and __animals__", wm) == ["nope"]
