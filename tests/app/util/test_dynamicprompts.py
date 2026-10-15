from __future__ import annotations

import pytest
from dynamicprompts.generators import CombinatorialPromptGenerator
from dynamicprompts.wildcards import WildcardManager

from invokeai.app.util.dynamicprompts import find_missing_wildcards


@pytest.fixture
def wildcard_manager() -> WildcardManager:
    return WildcardManager(root_map={"": [{"colors": ["red", "green", "blue"]}]})


def test_find_missing_wildcards_detects_unknown_wildcard_in_variant() -> None:
    # `__random__` inside a variant is parsed as a wildcard reference. Left unchecked it sends the
    # combinatorial generator into an infinite loop, so it must be reported up front.
    assert find_missing_wildcards("{__random__8chan|fenster|stuff}") == ["random"]


def test_find_missing_wildcards_detects_unknown_wildcard_nested_in_sequence_in_variant() -> None:
    assert find_missing_wildcards("{a __nope__|b}") == ["nope"]


@pytest.mark.parametrize("prompt", ["a __nope__ b", "__nope__", "a photo, __my_style__"])
def test_find_missing_wildcards_detects_unknown_wildcards_outside_variants(prompt: str) -> None:
    # Regression: these were previously ignored on the theory that only variant-nested wildcards
    # misbehave. They do not hang, but they silently collapse the prompt (see the generator test
    # below), so they must be reported too.
    assert find_missing_wildcards(prompt) != []


def test_unresolvable_wildcard_outside_a_variant_silently_collapses_the_prompt() -> None:
    # The behaviour that makes the case above worth reporting: rather than erroring, the generator
    # emits `max_prompts` copies of one prompt and drops the other variant's values entirely.
    prompt = "a {red|green} __nope__ ball"
    generated = CombinatorialPromptGenerator().generate(prompt, max_prompts=6)

    assert len(generated) == 6
    assert len(set(generated)) == 1
    assert find_missing_wildcards(prompt) == ["nope"]


@pytest.mark.parametrize("prompt", ["plain text", "{a|b|c}", "a {2$$x|y|z}"])
def test_find_missing_wildcards_ignores_prompts_without_wildcards(prompt: str) -> None:
    assert find_missing_wildcards(prompt) == []


def test_find_missing_wildcards_dedupes_repeated_unknown_wildcards() -> None:
    assert find_missing_wildcards("{__nope__|a} {__nope__|b} {__other__|c}") == ["nope", "other"]


def test_find_missing_wildcards_reports_nothing_for_resolvable_wildcards(
    wildcard_manager: WildcardManager,
) -> None:
    assert find_missing_wildcards("a __colors__ ball", wildcard_manager) == []
    assert find_missing_wildcards("{__colors__|x}", wildcard_manager) == []
    assert find_missing_wildcards("a {__colors__|__nope__} ball", wildcard_manager) == ["nope"]


@pytest.mark.timeout(20)
def test_resolvable_wildcard_in_a_variant_generates_instead_of_hanging(
    wildcard_manager: WildcardManager,
) -> None:
    # The hang this module guards against is a symptom of an unresolvable wildcard. Once the manager
    # can resolve it, the same shape generates normally.
    generated = CombinatorialPromptGenerator(wildcard_manager).generate("{__colors__|x}", max_prompts=5)

    assert sorted(generated) == ["blue", "green", "red", "x"]
