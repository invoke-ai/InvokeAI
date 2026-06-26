from __future__ import annotations

import pytest

from invokeai.app.util.dynamicprompts import find_missing_wildcards


def test_find_missing_wildcards_detects_unknown_wildcard_in_variant() -> None:
    # Regression: `__random__` inside a variant is parsed as a wildcard reference. Left unchecked it
    # sends the combinatorial generator into an infinite loop, so it must be reported up front.
    assert find_missing_wildcards("{__random__8chan|fenster|stuff}") == ["random"]


@pytest.mark.parametrize("prompt", ["plain text", "{a|b|c}", "a {2$$x|y|z}"])
def test_find_missing_wildcards_ignores_prompts_without_wildcards(prompt: str) -> None:
    assert find_missing_wildcards(prompt) == []


def test_find_missing_wildcards_dedupes_repeated_unknown_wildcards() -> None:
    assert find_missing_wildcards("__nope__ and __nope__ and __other__") == ["nope", "other"]
