"""Prompt-weighting helpers for Anima text conditioning."""

import math

import torch
from transformers import PreTrainedTokenizerBase

WeightedTextRange = tuple[int, int, float]


def _find_closing_parenthesis(prompt: str, opening_index: int) -> int | None:
    depth = 1
    index = opening_index + 1
    while index < len(prompt):
        if prompt[index] == "\\" and index + 1 < len(prompt) and prompt[index + 1] in "()\\":
            index += 2
            continue
        if prompt[index] == "(":
            depth += 1
        elif prompt[index] == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _split_explicit_weight(group: str) -> tuple[str, float] | None:
    """Split a trailing top-level ``:weight`` from a parenthesized group."""
    depth = 0
    last_colon: int | None = None
    index = 0
    while index < len(group):
        if group[index] == "\\" and index + 1 < len(group) and group[index + 1] in "()\\":
            index += 2
            continue
        if group[index] == "(":
            depth += 1
        elif group[index] == ")":
            depth -= 1
        elif group[index] == ":" and depth == 0:
            last_colon = index
        index += 1

    if last_colon is None or last_colon == 0:
        return None

    try:
        weight = float(group[last_colon + 1 :])
    except ValueError:
        return None
    if not math.isfinite(weight):
        return None
    return group[:last_colon], weight


def parse_prompt_attention(prompt: str) -> tuple[str, list[WeightedTextRange]]:
    """Remove prompt-weighting markup and return weighted character ranges.

    Parentheses increase weight by 10%, while a trailing ``:number`` sets an
    explicit weight for that group. Escaped parentheses remain literal text.
    """
    cleaned_parts: list[str] = []
    weighted_ranges: list[WeightedTextRange] = []
    cleaned_length = 0

    def append_text(text: str, weight: float) -> None:
        nonlocal cleaned_length
        if not text:
            return
        start = cleaned_length
        cleaned_parts.append(text)
        cleaned_length += len(text)
        if weighted_ranges and weighted_ranges[-1][1] == start and weighted_ranges[-1][2] == weight:
            previous_start, _, _ = weighted_ranges[-1]
            weighted_ranges[-1] = (previous_start, cleaned_length, weight)
        else:
            weighted_ranges.append((start, cleaned_length, weight))

    def parse(text: str, weight: float) -> None:
        index = 0
        plain_text: list[str] = []

        def flush_plain_text() -> None:
            if plain_text:
                append_text("".join(plain_text), weight)
                plain_text.clear()

        while index < len(text):
            char = text[index]
            if char == "\\" and index + 1 < len(text) and text[index + 1] in "()\\":
                plain_text.append(text[index + 1])
                index += 2
                continue
            if char != "(":
                plain_text.append(char)
                index += 1
                continue

            closing_index = _find_closing_parenthesis(text, index)
            if closing_index is None:
                plain_text.append(char)
                index += 1
                continue

            flush_plain_text()
            group = text[index + 1 : closing_index]
            explicit_weight = _split_explicit_weight(group)
            if explicit_weight is None:
                parse(group, weight * 1.1)
            else:
                group_text, group_weight = explicit_weight
                parse(group_text, group_weight)
            index = closing_index + 1

        flush_plain_text()

    parse(prompt, 1.0)
    return "".join(cleaned_parts), weighted_ranges


def tokenize_t5_with_weights(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    weighted_ranges: list[WeightedTextRange],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize cleaned text once, then map character weights to T5 tokens."""
    tokens = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids[0]
    offsets = tokens.offset_mapping[0].tolist()
    special_tokens_mask = tokens.special_tokens_mask[0].tolist()

    token_weights: list[float] = []
    range_index = 0
    for (token_start, token_end), is_special in zip(offsets, special_tokens_mask, strict=True):
        if is_special or token_end <= token_start:
            token_weights.append(1.0)
            continue

        while range_index < len(weighted_ranges) and weighted_ranges[range_index][1] <= token_start:
            range_index += 1

        weighted_length = 0.0
        covered_length = 0
        current_range = range_index
        while current_range < len(weighted_ranges) and weighted_ranges[current_range][0] < token_end:
            range_start, range_end, weight = weighted_ranges[current_range]
            overlap = max(0, min(token_end, range_end) - max(token_start, range_start))
            weighted_length += overlap * weight
            covered_length += overlap
            current_range += 1

        token_weights.append(weighted_length / covered_length if covered_length else 1.0)

    return input_ids, torch.tensor(token_weights, dtype=torch.float32)
