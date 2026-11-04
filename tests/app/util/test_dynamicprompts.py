from __future__ import annotations

import pytest
from dynamicprompts.generators import CombinatorialPromptGenerator
from dynamicprompts.wildcards import WildcardManager

from invokeai.app.util.dynamicprompts import ExpandedPrompts, expand_dynamic_prompt, find_missing_wildcards


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


@pytest.mark.parametrize(
    "prompt",
    [
        "${c={__nope__|x}} a ${c}",
        "${c=__nope__} a ${c}",
        "a ${c:{__nope__|x}}",
    ],
)
def test_find_missing_wildcards_detects_unknown_wildcards_in_variables(prompt: str) -> None:
    # A variable's value (or an access's default) is expanded exactly like an inline prompt, so an
    # unresolvable wildcard there hangs or collapses the combinatorial generator the same way.
    assert find_missing_wildcards(prompt) == ["nope"]


def test_find_missing_wildcards_reports_nothing_for_resolvable_wildcards_in_variables(
    wildcard_manager: WildcardManager,
) -> None:
    assert find_missing_wildcards("${c={__colors__|x}} a ${c}", wildcard_manager) == []
    assert find_missing_wildcards("a ${c:__colors__} ${d}", wildcard_manager) == []


@pytest.mark.parametrize(
    "prompt",
    [
        # `c` is set, so the default never runs.
        "${c=red} a ${c:{__nope__|x}}",
        # The lazy value is never read.
        "${c={__nope__|x}} a",
        # The lazy value is replaced before `c` is read.
        "${c={__nope__|x}} ${c=red} a ${c}",
        # An immediate value is sampled once, keeping only its first result.
        "${c=!{red|__nope__}} a ${c}",
        "${c=!{__nope__|x}} a {b|d}",
        # The first result of a bare unknown wildcard is its wrapped name.
        "${c=!__nope__ {a|b}} a ${c}",
        # Allowing zero options makes the first result an empty string.
        "${c=!{0-1$$__nope__|{__nope__}}} a ${c}",
        # Picking several options keeps the first combination, the first options in order, when
        # they are distinct literals.
        "${c=!{2$$red|blue|__nope__}} a ${c}",
        "${c=!{2-3$$red|blue|{__nope__}}} a ${c}",
        # The same when the first option is itself a variant.
        "${c=!{2$${red|green}|blue|__nope__}} a ${c}",
        # Duplicate literals in the first combination are deduped away, but the next combination
        # still produces a result before the `{__nope__}` option is reached.
        "${c=!{2$$red|red|__nope__|{__nope__}}} a ${c}",
    ],
)
def test_find_missing_wildcards_ignores_variable_values_that_never_run(prompt: str) -> None:
    assert find_missing_wildcards(prompt) == []
    result = expand_dynamic_prompt(prompt, max_prompts=10, combinatorial=True)
    assert result.error is None


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "prompt",
    [
        # Picking two distinct options from values that all sample to `__nope__` never finishes,
        # so the immediate assignment hangs before its first result.
        "${c=!{2$$__nope__|{__nope__}|red}} a ${c}",
        # A `{__wildcard__}` variant of an unknown wildcard has no options, so the first result
        # never comes and the sampler raises instead.
        "${c=!{__nope__}} a ${c}",
        # The same, reached through the first option, a variable default and an overwritten assignment.
        "${c=!{{2$$__nope__|{__nope__}}|red}} a ${c}",
        "${c=!x ${d:{__nope__}}} a ${c}",
        "${c=!{__nope__}} ${c=red} a ${c}",
        # Every combination of two distinct options needs `{__nope__}` or a second `__nope__`.
        "${c=!{2$$red|red|{__nope__}}} a ${c}",
    ],
)
def test_find_missing_wildcards_detects_immediate_values_that_cannot_produce_a_result(prompt: str) -> None:
    assert find_missing_wildcards(prompt) == ["nope"]
    result = expand_dynamic_prompt(prompt, max_prompts=1, combinatorial=True)
    assert result.error is not None


@pytest.mark.timeout(5)
@pytest.mark.parametrize(
    "prompt",
    [
        # A cyclical variant repeats `red` forever and the two-pick dedupes every repeat away, so the
        # immediate value never produces a result even though it reads no unknown wildcard.
        "${c=!{2$$${x:{@red}}|red}} __nope__",
    ],
)
def test_find_missing_wildcards_does_not_hang_on_an_immediate_value_that_never_produces_a_result(
    prompt: str,
) -> None:
    assert find_missing_wildcards(prompt) == ["nope"]
    result = expand_dynamic_prompt(prompt, max_prompts=1, combinatorial=True)
    assert result.error is not None


def test_immediate_assignment_keeps_only_the_first_result() -> None:
    result = expand_dynamic_prompt("${c=!{red|__nope__}} a ${c}", max_prompts=10, combinatorial=True)

    assert result == ExpandedPrompts(prompts=[" a red"])


def test_immediate_multi_pick_assignment_keeps_the_first_combination() -> None:
    result = expand_dynamic_prompt("${c=!{2$$red|blue|__nope__}} a ${c}", max_prompts=10, combinatorial=True)

    assert result == ExpandedPrompts(prompts=[" a red,blue"])


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("${c=!{2$${red|green}|blue|__nope__}} a ${c}", " a red,blue"),
        ("${c=!{2$$red|red|__nope__|{__nope__}}} a ${c}", " a red,__nope__"),
    ],
)
def test_immediate_multi_pick_assignment_matches_the_sampler(prompt: str, expected: str) -> None:
    result = expand_dynamic_prompt(prompt, max_prompts=10, combinatorial=True)

    assert result == ExpandedPrompts(prompts=[expected])


@pytest.mark.timeout(5)
def test_find_missing_wildcards_walks_a_shared_variable_expression_once() -> None:
    # Every variable reads the previous one twice, so walking each reference separately would take
    # 2**40 steps.
    chain = "".join(f"${{v{i}=${{v{i - 1}}} ${{v{i - 1}}}}} " for i in range(1, 41))
    prompt = "${v0={a|b}} " + chain + "${v40}"

    assert find_missing_wildcards(prompt) == []
    assert find_missing_wildcards(prompt.replace("{a|b}", "{a|__nope__}")) == ["nope"]
