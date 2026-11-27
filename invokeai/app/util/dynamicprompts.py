from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from dynamicprompts.commands import (
    Command,
    SequenceCommand,
    VariantCommand,
    WildcardCommand,
    WrapCommand,
)
from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from dynamicprompts.parser.parse import parse
from dynamicprompts.wildcards import WildcardManager
from pyparsing import ParseException


def _iter_wildcard_names(command: Command) -> Iterator[str]:
    """Recursively yield the statically-known wildcard names referenced by `command`."""
    if isinstance(command, WildcardCommand):
        # The wildcard name may itself be a dynamic Command (e.g. `__${var}__`). Only plain string
        # names can be validated ahead of time, so the dynamic case is intentionally skipped.
        if isinstance(command.wildcard, str):
            yield command.wildcard
    elif isinstance(command, SequenceCommand):
        for token in command.tokens:
            yield from _iter_wildcard_names(token)
    elif isinstance(command, VariantCommand):
        for value in command.values:
            yield from _iter_wildcard_names(value)
    elif isinstance(command, WrapCommand):
        yield from _iter_wildcard_names(command.wrapper)
        yield from _iter_wildcard_names(command.inner)
    # LiteralCommand and variable commands reference no wildcards we can resolve statically.


def find_missing_wildcards(prompt: str, wildcard_manager: WildcardManager | None = None) -> list[str]:
    """Return the unique wildcard names in `prompt` that `wildcard_manager` cannot resolve.

    An unresolvable wildcard breaks the combinatorial generator two different ways, so callers
    should treat any non-empty result as a hard error rather than generating:

    - *As a variant value* (e.g. `{__nope__|x}`) it loops forever: the not-found fallback
      (`get_wildcard_not_found_fallback`) yields the wrapped wildcard infinitely, and the
      combinatorial variant logic dedupes those duplicates away without ever advancing.
    - *Anywhere else* (e.g. `a {red|green} __nope__ b`) it silently produces `max_prompts` copies of
      a single prompt and collapses the other variants, so a caller that generated anyway would
      queue N identical results.

    Only the combinatorial generator is affected; the random generator leaves an unresolvable
    wildcard as literal text.

    Without a configured `wildcard_manager`, an empty one is used, so every referenced wildcard is
    reported as missing.
    """
    if wildcard_manager is None:
        wildcard_manager = WildcardManager()

    try:
        tree = parse(prompt)
    except ParseException:
        # Malformed prompts are surfaced separately by the generators; nothing to validate here.
        return []

    missing: list[str] = []
    for name in _iter_wildcard_names(tree):
        if name not in missing and not wildcard_manager.get_values(name):
            missing.append(name)
    return missing


@dataclass(frozen=True)
class ExpandedPrompts:
    """The outcome of expanding one prompt.

    `error` is a soft failure: `prompts` always holds something submittable (the original prompt
    when expansion could not proceed), so an HTTP caller can surface the message while a caller that
    must fail hard can raise on it.
    """

    prompts: list[str]
    error: str | None = None


def expand_dynamic_prompt(
    prompt: str,
    *,
    max_prompts: int,
    combinatorial: bool,
    seed: int | None = None,
    wildcard_manager: WildcardManager | None = None,
) -> ExpandedPrompts:
    """Expand `prompt` with adieyal/dynamicprompts, guarding the cases that misbehave.

    Shared by the `/utilities/dynamicprompts` route and `DynamicPromptInvocation` so the two agree
    on guard, generator selection and error text.
    """
    if combinatorial:
        # Unresolvable wildcards either hang the combinatorial generator or silently collapse the
        # prompt, so bail out before generating. The random generator needs no such guard.
        missing_wildcards = find_missing_wildcards(prompt, wildcard_manager)
        if missing_wildcards:
            return ExpandedPrompts(
                prompts=[prompt],
                error=f"No values found for wildcard(s): {', '.join(missing_wildcards)}",
            )

    try:
        if combinatorial:
            prompts = CombinatorialPromptGenerator(wildcard_manager).generate(prompt, max_prompts=max_prompts)
        else:
            prompts = RandomPromptGenerator(wildcard_manager, seed=seed).generate(prompt, num_images=max_prompts)
    except ParseException as e:
        return ExpandedPrompts(prompts=[prompt], error=str(e))

    return ExpandedPrompts(prompts=list(prompts) if prompts else [""])
