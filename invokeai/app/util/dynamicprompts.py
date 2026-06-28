from __future__ import annotations

from collections.abc import Iterator

from dynamicprompts.commands import (
    Command,
    SequenceCommand,
    VariantCommand,
    WildcardCommand,
    WrapCommand,
)
from dynamicprompts.parser.parse import parse
from dynamicprompts.wildcards import WildcardManager
from pyparsing import ParseException


def _iter_wildcard_names(command: Command, in_variant: bool = False) -> Iterator[str]:
    """Recursively yield the statically-known wildcard names that appear as (part of) a variant value.

    Only wildcards reachable from a `VariantCommand` value are yielded: those are the references that
    hang the combinatorial generator (see `find_missing_wildcards`). The same wildcard used as plain
    literal text (e.g. `a __nope__ b`) generates fine, so it is intentionally not reported.
    """
    if isinstance(command, WildcardCommand):
        # The wildcard name may itself be a dynamic Command (e.g. `__${var}__`). Only plain string
        # names can be validated ahead of time, so the dynamic case is intentionally skipped.
        if in_variant and isinstance(command.wildcard, str):
            yield command.wildcard
    elif isinstance(command, SequenceCommand):
        for token in command.tokens:
            yield from _iter_wildcard_names(token, in_variant)
    elif isinstance(command, VariantCommand):
        # Everything below a variant value is a variant-nested reference, even across sequences.
        for value in command.values:
            yield from _iter_wildcard_names(value, in_variant=True)
    elif isinstance(command, WrapCommand):
        yield from _iter_wildcard_names(command.wrapper, in_variant)
        yield from _iter_wildcard_names(command.inner, in_variant)
    # LiteralCommand and variable commands reference no wildcards we can resolve statically.


def find_missing_wildcards(prompt: str, wildcard_manager: WildcardManager | None = None) -> list[str]:
    """Return the unique unknown wildcard names in `prompt` that hang the combinatorial generator.

    Referencing an unknown wildcard *as a variant value* (e.g. `{__nope__|x}`) makes dynamicprompts'
    combinatorial generator loop forever: its not-found fallback (`get_wildcard_not_found_fallback`)
    yields the wrapped wildcard infinitely, and the combinatorial variant logic dedupes those
    duplicates away without ever advancing. Detecting these names up front lets callers report a clear
    error instead of hanging. Only the combinatorial generator is affected, and only for wildcards
    nested in a variant — a bare `a __nope__ b` generates fine and is not reported.

    Without a configured `wildcard_manager`, an empty one is used so that every referenced wildcard is
    treated as missing (wildcards are not resolved against any files here).
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
