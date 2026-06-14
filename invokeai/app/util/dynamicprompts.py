from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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


def get_wildcard_manager(wildcards_path: Path) -> WildcardManager:
    """Build a `WildcardManager` rooted at the given directory, creating the directory if needed.

    Wildcard collections are plain `.txt` files (one value per line) placed under this directory and
    referenced in prompts as `__name__` (subdirectories map to `__subdir/name__`).
    """
    wildcards_path.mkdir(parents=True, exist_ok=True)
    return WildcardManager(wildcards_path)


def _iter_wildcard_names(command: Command) -> Iterator[str]:
    """Recursively yield the statically-known wildcard names referenced in a parsed prompt."""
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


def find_missing_wildcards(prompt: str, wildcard_manager: WildcardManager) -> list[str]:
    """Return the unique wildcard names referenced in `prompt` that resolve to no values.

    Referencing an unknown wildcard makes dynamicprompts' combinatorial generator loop forever: its
    not-found fallback (`get_wildcard_not_found_fallback`) yields the wrapped wildcard infinitely, and
    the combinatorial variant logic dedupes those duplicates away without ever advancing. Detecting
    the missing names up front lets callers report a clear error instead of hanging.
    """
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
