from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from dynamicprompts.commands import (
    Command,
    LiteralCommand,
    SamplingMethod,
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

# Stands in for an immediate `${c=!...}` value, which the sampler reduces to one literal string.
_SAMPLED_VALUE = LiteralCommand("")


def _first_combination_is_distinct_literals(values: list[Command]) -> bool:
    """Whether `values` are plain literals with pairwise different text."""
    texts = [value.literal for value in values if isinstance(value, LiteralCommand)]
    return len(texts) == len(values) and len(set(texts)) == len(texts)


def _collect_wildcard_names(command: Command) -> list[str]:
    """Return the statically-known wildcard names that sampling `command` would evaluate, in order.

    Variables are tracked the way `SamplingContext` resolves them, so a wildcard is only reported when
    it can actually run: a lazy assignment's value counts where the variable is read and a variable
    default counts only while the variable is unset.

    An immediate assignment's value is sampled once, where it is assigned, and only its first result
    is kept (`SamplingContext.process_variable_assignment`), so it is walked in "first result" mode
    (see `first_only` below) and the variable is bound to a literal placeholder.

    A command is walked at most once per variable scope, so an expression that many references
    share is not walked again for each of them.
    """
    names: dict[str, None] = {}
    visited: set[tuple[int, frozenset[tuple[str, int]], bool]] = set()

    def visit(
        command: Command,
        scope: Mapping[str, Command],
        scope_key: frozenset[tuple[str, int]],
        first_only: bool = False,
    ) -> None:
        """Record the wildcards in `command`.

        With `first_only`, only wildcards that can stop the combinatorial sampler from producing its
        *first* result are recorded. A bare unresolvable wildcard is harmless there, because its
        not-found fallback yields the wrapped name straight away. A plain variant whose smallest pick
        is one option first samples only its first option, and one whose smallest pick is zero
        options first yields an empty string. Anything else (a variant that picks several distinct
        options, a `{__wildcard__}` variant, or a non-combinatorial variant) is walked in full,
        because the deduplication there can loop forever, or end without a result, before yielding
        anything.
        """
        key = (id(command), scope_key, first_only)
        if key in visited:
            # Already walked in this scope (or being walked, for a recursive reference that the
            # sampler rejects with its own RecursionError), so it cannot add a new name.
            return
        visited.add(key)
        if isinstance(command, WildcardCommand):
            # The wildcard name may itself be a dynamic Command (e.g. `__${var}__`). Only plain string
            # names can be validated ahead of time, so the dynamic case is intentionally skipped.
            if isinstance(command.wildcard, str) and not first_only:
                names.setdefault(command.wildcard)
        elif isinstance(command, SequenceCommand):
            # Mirrors `SamplingContext.process_variable_assignments`: every assignment in the sequence
            # is applied against the enclosing scope before any other token is sampled, the last one
            # wins.
            assigned: dict[str, Command] = {}
            for token in command.tokens:
                if not isinstance(token, VariableAssignmentCommand):
                    continue
                if not token.overwrite and token.name in scope:
                    assigned[token.name] = scope[token.name]
                elif token.immediate:
                    # Sampled right here against the enclosing scope, even if a later assignment
                    # replaces it, and only the first result is kept.
                    visit(token.value, scope, scope_key, first_only=True)
                    assigned[token.name] = _SAMPLED_VALUE
                else:
                    assigned[token.name] = token.value
            if assigned:
                scope = {**scope, **assigned}
                scope_key = frozenset((name, id(value)) for name, value in scope.items())
            for token in command.tokens:
                if not isinstance(token, VariableAssignmentCommand):
                    visit(token, scope, scope_key, first_only)
        elif isinstance(command, VariantCommand):
            is_wildcard_variant = len(command.values) == 1 and isinstance(command.values[0], WildcardCommand)
            smallest_pick = min(command.min_bound, len(command.values))
            first_only_variant = (
                first_only
                and not is_wildcard_variant
                and command.sampling_method in (None, SamplingMethod.COMBINATORIAL)
            )
            if first_only_variant and smallest_pick <= 1:
                if command.min_bound >= 1 and command.values:
                    visit(command.values[0], scope, scope_key, first_only)
            elif first_only_variant and _first_combination_is_distinct_literals(command.values[:smallest_pick]):
                # The first combination the sampler tries is the first `smallest_pick` options in
                # order. Distinct literals always survive deduplication, so that combination is the
                # first result and no other option is ever sampled.
                pass
            else:
                for value in command.values:
                    visit(value, scope, scope_key)
        elif isinstance(command, WrapCommand):
            visit(command.wrapper, scope, scope_key, first_only)
            visit(command.inner, scope, scope_key, first_only)
        elif isinstance(command, VariableAccessCommand):
            # `${c}` samples the value bound to `c`, and `${c:{__nope__|x}}` falls back to its default
            # only when `c` is unset.
            target = scope.get(command.name, command.default)
            if target is not None:
                visit(target, scope, scope_key, first_only)
        # LiteralCommand references no wildcards.

    visit(command, {}, frozenset())
    return list(names)


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

    return [name for name in _collect_wildcard_names(tree) if not wildcard_manager.get_values(name)]


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
