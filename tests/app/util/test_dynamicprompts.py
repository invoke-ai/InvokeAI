from __future__ import annotations

import pytest

from invokeai.app.util.dynamicprompts import find_missing_wildcards


def test_find_missing_wildcards_detects_unknown_wildcard_in_variant() -> None:
    # Regression: `__random__` inside a variant is parsed as a wildcard reference. Left unchecked it
    # sends the combinatorial generator into an infinite loop, so it must be reported up front.
    assert find_missing_wildcards("{__random__8chan|fenster|stuff}") == ["random"]


def test_find_missing_wildcards_detects_unknown_wildcard_nested_in_sequence_in_variant() -> None:
    # The wildcard hangs the generator even when wrapped in other text inside the variant value.
    assert find_missing_wildcards("{a __nope__|b}") == ["nope"]


@pytest.mark.parametrize("prompt", ["a __nope__ b", "__nope__", "a photo, __my_style__"])
def test_find_missing_wildcards_ignores_wildcards_outside_variants(prompt: str) -> None:
    # A wildcard used as plain literal text generates fine (no hang), so it must not be reported.
    assert find_missing_wildcards(prompt) == []


@pytest.mark.parametrize("prompt", ["plain text", "{a|b|c}", "a {2$$x|y|z}"])
def test_find_missing_wildcards_ignores_prompts_without_wildcards(prompt: str) -> None:
    assert find_missing_wildcards(prompt) == []


def test_find_missing_wildcards_dedupes_repeated_unknown_wildcards() -> None:
    assert find_missing_wildcards("{__nope__|a} {__nope__|b} {__other__|c}") == ["nope", "other"]
