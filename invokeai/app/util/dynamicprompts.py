from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

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
from dynamicprompts.samplers import Sampler
from dynamicprompts.samplers.combinatorial import CombinatorialSampler
from dynamicprompts.samplers.cycle import CyclicalSampler
from dynamicprompts.samplers.random import RandomSampler
from dynamicprompts.sampling_context import SamplingContext
from dynamicprompts.types import ResultGen
from dynamicprompts.wildcards import WildcardManager
from pyparsing import ParseException

# Stands in for an immediate `${c=!...}` value whose first result could not be worked out.
_SAMPLED_VALUE = LiteralCommand("")

# How many results the dry run of an immediate value may draw, summed over every part of the value,
# before it treats the value as hanging. A value that loops forever (an unknown wildcard's not-found
# fallback, or a cyclical variant whose repeats a multi-pick keeps deduping away) keeps drawing
# without producing a result, while a value that produces a result at all needs only a few draws.
_DRAW_LIMIT = 10_000


class _DrawLimitReached(Exception):
    """Raised when the dry run keeps drawing results without producing a first result."""


class _RecordingWildcardManager:
    """Wraps a `WildcardManager` and records every wildcard it was asked for and could not resolve."""

    def __init__(self, inner: WildcardManager) -> None:
        self._inner = inner
        self.missing: dict[str, None] = {}

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def get_values(self, wildcard: str) -> object:
        values = self._inner.get_values(wildcard)
        if not values:
            self.missing.setdefault(wildcard)
        return values


class _DrawBudget:
    """Counts the results drawn across the whole dry run and stops it at `_DRAW_LIMIT`."""

    def __init__(self) -> None:
        self.draws = 0

    def count(self, results: ResultGen) -> ResultGen:
        for result in results:
            self.draws += 1
            if self.draws > _DRAW_LIMIT:
                raise _DrawLimitReached()
            yield result


class _BudgetedSampler:
    """Wraps a sampler so that every result it hands out is charged to a shared `_DrawBudget`.

    `SamplingContext` looks up the sampler for every sub-command, so wrapping the context's samplers
    bounds all of the dry run's work, however deeply the value nests.
    """

    def __init__(self, inner: Sampler, budget: _DrawBudget) -> None:
        self._inner = inner
        self._budget = budget

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def generator_from_command(self, command: Command, context: SamplingContext) -> ResultGen:
        return self._budget.count(self._inner.generator_from_command(command, context))


def _sample_first_result(
    value: Command, scope: Mapping[str, Command], wildcard_manager: WildcardManager
) -> tuple[str | None, list[str]]:
    """Dry-run an immediate `${c=!...}` value the way `SamplingContext.process_variable_assignment` does.

    The sampler keeps only the first result of an immediate value, so whether an unknown wildcard in it
    matters depends on whether that wildcard has to run before the first result comes out. Rather than
    predicting that from the parse tree, the value is sampled with the real combinatorial sampler,
    with a limit on how many results the whole dry run may draw.

    Returns the first result, or `None` when there is none, together with the unknown wildcards that
    were reached when there is no result (the sampler would hang or fail on them).
    """
    recorder = _RecordingWildcardManager(wildcard_manager)
    budget = _DrawBudget()
    samplers: dict[SamplingMethod, Sampler] = {
        SamplingMethod.COMBINATORIAL: CombinatorialSampler(),
        SamplingMethod.CYCLICAL: CyclicalSampler(),
        SamplingMethod.RANDOM: RandomSampler(),
    }
    context = SamplingContext(
        default_sampling_method=SamplingMethod.COMBINATORIAL,
        wildcard_manager=cast(WildcardManager, recorder),
        samplers={method: cast(Sampler, _BudgetedSampler(sampler, budget)) for method, sampler in samplers.items()},
        variables=dict(scope),
    )
    try:
        return str(next(iter(context.generator_from_command(value)))), []
    except (_DrawLimitReached, StopIteration):
        # No first result: either the value would keep drawing forever, or it has no results.
        return None, list(recorder.missing)
    except RuntimeError as e:
        if isinstance(e.__cause__, StopIteration):
            # A nested part with no results ends a generator early, which Python reports this way.
            return None, list(recorder.missing)
        return None, []
    except Exception:
        # Other failures (an unknown variable, a recursive one) are raised by the generator itself
        # with their own message, so they are not reported as missing wildcards.
        return None, []


def _collect_wildcard_names(command: Command, wildcard_manager: WildcardManager) -> list[str]:
    """Return the statically-known wildcard names that sampling `command` would evaluate, in order.

    Variables are tracked the way `SamplingContext` resolves them, so a wildcard is only reported when
    it can actually run: a lazy assignment's value counts where the variable is read and a variable
    default counts only while the variable is unset.

    An immediate assignment's value is sampled once, where it is assigned, and only its first result
    is kept, so it is dry-run with `_sample_first_result` and the variable is bound to that result.

    A command is walked at most once per variable scope, so an expression that many references
    share is not walked again for each of them.
    """
    names: dict[str, None] = {}
    visited: set[tuple[int, frozenset[tuple[str, int]]]] = set()

    def visit(command: Command, scope: Mapping[str, Command], scope_key: frozenset[tuple[str, int]]) -> None:
        key = (id(command), scope_key)
        if key in visited:
            # Already walked in this scope (or being walked, for a recursive reference that the
            # sampler rejects with its own RecursionError), so it cannot add a new name.
            return
        visited.add(key)
        if isinstance(command, WildcardCommand):
            # The wildcard name may itself be a dynamic Command (e.g. `__${var}__`). Only plain string
            # names can be validated ahead of time, so the dynamic case is intentionally skipped.
            if isinstance(command.wildcard, str):
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
                    first_result, missing = _sample_first_result(token.value, scope, wildcard_manager)
                    for name in missing:
                        names.setdefault(name)
                    assigned[token.name] = _SAMPLED_VALUE if first_result is None else LiteralCommand(first_result)
                else:
                    assigned[token.name] = token.value
            if assigned:
                scope = {**scope, **assigned}
                scope_key = frozenset((name, id(value)) for name, value in scope.items())
            for token in command.tokens:
                if not isinstance(token, VariableAssignmentCommand):
                    visit(token, scope, scope_key)
        elif isinstance(command, VariantCommand):
            for value in command.values:
                visit(value, scope, scope_key)
        elif isinstance(command, WrapCommand):
            visit(command.wrapper, scope, scope_key)
            visit(command.inner, scope, scope_key)
        elif isinstance(command, VariableAccessCommand):
            # `${c}` samples the value bound to `c`, and `${c:{__nope__|x}}` falls back to its default
            # only when `c` is unset.
            target = scope.get(command.name, command.default)
            if target is not None:
                visit(target, scope, scope_key)
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

    return [name for name in _collect_wildcard_names(tree, wildcard_manager) if not wildcard_manager.get_values(name)]


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
