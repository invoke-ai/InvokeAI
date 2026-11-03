from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from dynamicprompts.commands import (
    Command,
    LiteralCommand,
    SequenceCommand,
    VariantCommand,
    WildcardCommand,
    WrapCommand,
)
from dynamicprompts.commands.variable_commands import VariableAccessCommand, VariableAssignmentCommand
from dynamicprompts.generators import CombinatorialPromptGenerator, RandomPromptGenerator
from dynamicprompts.parser.parse import parse
from dynamicprompts.wildcards import WildcardManager
from pyparsing import ParseException


def _iter_wildcard_names(
    command: Command,
    variables: Mapping[str, Command] | None = None,
    sampling: frozenset[str] = frozenset(),
) -> Iterator[str]:
    """Recursively yield the statically-known wildcard names that sampling `command` would evaluate.

    Variables are tracked the way `SamplingContext` resolves them, so a wildcard is only reported when
    it can actually run: a lazy assignment's value counts where the variable is read, a variable
    default counts only while the variable is unset, and an immediate assignment counts where it is
    assigned.
    """
    if variables is None:
        variables = {}
    if isinstance(command, WildcardCommand):
        # The wildcard name may itself be a dynamic Command (e.g. `__${var}__`). Only plain string
        # names can be validated ahead of time, so the dynamic case is intentionally skipped.
        if isinstance(command.wildcard, str):
            yield command.wildcard
    elif isinstance(command, SequenceCommand):
        # Mirrors `SamplingContext.process_variable_assignments`: every assignment in the sequence is
        # applied against the enclosing scope before any other token is sampled, the last one wins.
        assigned: dict[str, Command] = {}
        for token in command.tokens:
            if not isinstance(token, VariableAssignmentCommand):
                continue
            if not token.overwrite and token.name in variables:
                assigned[token.name] = variables[token.name]
            elif token.immediate:
                # `${c=!{__nope__|x}}` samples its value right away, whether or not `c` is read.
                yield from _iter_wildcard_names(token.value, variables, sampling)
                assigned[token.name] = LiteralCommand("")
            else:
                assigned[token.name] = token.value
        scope = {**variables, **assigned} if assigned else variables
        for token in command.tokens:
            if not isinstance(token, VariableAssignmentCommand):
                yield from _iter_wildcard_names(token, scope, sampling)
    elif isinstance(command, VariantCommand):
        for value in command.values:
            yield from _iter_wildcard_names(value, variables, sampling)
    elif isinstance(command, WrapCommand):
        yield from _iter_wildcard_names(command.wrapper, variables, sampling)
        yield from _iter_wildcard_names(command.inner, variables, sampling)
    elif isinstance(command, VariableAccessCommand):
        # `${c}` samples the value bound to `c`, and `${c:{__nope__|x}}` falls back to its default
        # only when `c` is unset. A recursive reference is left to the sampler's own RecursionError.
        if command.name not in sampling:
            target = variables.get(command.name, command.default)
            if target is not None:
                yield from _iter_wildcard_names(target, variables, sampling | {command.name})
    # LiteralCommand references no wildcards.


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
