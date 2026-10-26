"""Prompt-weighting helpers for Anima text conditioning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase

WeightedTextRange = tuple[int, int, float]
MAX_WEIGHT = torch.finfo(torch.float16).max
NUMERIC_SUFFIX = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")


@dataclass
class _Group:
    parts: list[str | "_Group"]
    weight: float = 1.0
    closed: bool = True


@dataclass
class _Frame:
    parts: list[str | _Group] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    word_start: int = 0
    word_has_unsupported_syntax: bool = False

    def append_text(self, value: str, *, blocked: bool = False, escaped: bool = False) -> None:
        for char in value:
            self.text.append(char)
            if char.isspace() or char == ",":
                self.word_start = len(self.text)
                self.word_has_unsupported_syntax = False
            elif char in '.=")' and not escaped:
                self.word_has_unsupported_syntax = True
        if blocked:
            self.word_has_unsupported_syntax = True

    def flush_text(self) -> None:
        if self.text:
            self.parts.append("".join(self.text))
            self.text.clear()
        self.word_start = 0
        self.word_has_unsupported_syntax = False


def _validate_weight(weight: float) -> float:
    if not math.isfinite(weight) or abs(weight) > MAX_WEIGHT:
        raise ValueError("Anima prompt weight must be finite and within float16 magnitude.")
    return weight


def _suffix_weight(prompt: str, index: int) -> tuple[float, int]:
    """Read a Compel-style numeric or repeated +/- suffix after a group."""
    if index >= len(prompt):
        return 1.0, index
    if prompt[index] in "+-":
        marker = prompt[index]
        end = index + 1
        while end < len(prompt) and prompt[end] == marker:
            end += 1
        try:
            weight = (1.1 if marker == "+" else 0.9) ** (end - index)
        except OverflowError as error:
            raise ValueError("Anima prompt weight must be finite and within float16 magnitude.") from error
        return _validate_weight(weight), end
    match = NUMERIC_SUFFIX.match(prompt, index)
    if match is not None:
        return _validate_weight(float(match.group())), match.end()
    return 1.0, index


def _finish_group(frame: _Frame, closed: bool, weight: float = 1.0) -> _Group:
    frame.flush_text()
    return _Group(frame.parts, weight=weight, closed=closed)


def _unescape_text(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value) and value[index + 1] in "()\\":
            result.append(value[index + 1])
            index += 2
        else:
            result.append(value[index])
            index += 1
    return "".join(result)


def parse_prompt_attention(prompt: str) -> tuple[str, list[WeightedTextRange]]:
    """Parse Compel-style attention weights and return weighted character ranges.

    Supports (text)1.5, (text)++, and word+/- without changing the Anima
    adapter's weight application. Escaped parentheses stay literal. Parsing and
    rendering use explicit stacks so deeply nested input cannot recurse.
    """
    frames = [_Frame()]
    index = 0
    has_markup = False
    has_escapes = False

    while index < len(prompt):
        char = prompt[index]
        frame = frames[-1]
        if char == "\\" and index + 1 < len(prompt) and prompt[index + 1] in "()\\":
            has_escapes = True
            frame.append_text(prompt[index : index + 2], escaped=True)
            index += 2
        elif char == "(":
            frame.flush_text()
            if frame.parts and isinstance(frame.parts[-1], str) and frame.parts[-1][-1].isalnum():
                frame.parts.append(" ")
            frames.append(_Frame())
            index += 1
        elif char == ")" and len(frames) > 1:
            weight, next_index = _suffix_weight(prompt, index + 1)
            group = _finish_group(frames.pop(), closed=True, weight=weight)
            frames[-1].parts.append(group)
            if next_index < len(prompt) and (prompt[next_index].isalnum() or prompt[next_index] == "("):
                frames[-1].parts.append(" ")
            has_markup = True
            index = next_index
        elif char in "+-":
            end = index + 1
            while end < len(prompt) and prompt[end] == char:
                end += 1
            # A lone sign followed by another word character belongs to that word in Compel.
            next_continues_word = end < len(prompt) and not (prompt[end].isspace() or prompt[end] in '(),.+=-"')
            if end == index + 1 and next_continues_word:
                frame.append_text(char)
                index = end
                continue

            # Comma separates fragments; a dot, equals sign, or quote makes a bare word
            # fall back to literal text. Keep that decision as the word grows so malformed
            # runs cannot repeatedly scan the entire prefix before token truncation.
            if frame.word_start < len(frame.text) and not frame.word_has_unsupported_syntax:
                weight, _ = _suffix_weight(prompt, index)
                word = "".join(frame.text[frame.word_start :])
                del frame.text[frame.word_start :]
                frame.flush_text()
                frame.parts.append(_Group([word], weight=weight))
                if next_continues_word:
                    frame.parts.append(" ")
                has_markup = True
                index = end
            else:
                frame.append_text(prompt[index:end], blocked=True)
                index = end
        else:
            frame.append_text(char)
            index += 1

    while len(frames) > 1:
        group = _finish_group(frames.pop(), closed=False)
        frames[-1].parts.append(group)
    frames[0].flush_text()

    if not has_markup and not has_escapes:
        return prompt, [(0, len(prompt), 1.0)] if prompt else []

    cleaned_parts: list[str] = []
    weighted_ranges: list[WeightedTextRange] = []
    cleaned_length = 0

    def append_text(value: str, weight: float) -> None:
        nonlocal cleaned_length
        if not value:
            return
        start = cleaned_length
        cleaned_parts.append(value)
        cleaned_length += len(value)
        if weighted_ranges and weighted_ranges[-1][1] == start and weighted_ranges[-1][2] == weight:
            weighted_ranges[-1] = (weighted_ranges[-1][0], cleaned_length, weight)
        else:
            weighted_ranges.append((start, cleaned_length, weight))

    tasks: list[tuple[str | _Group, float]] = [(part, 1.0) for part in reversed(frames[0].parts)]
    while tasks:
        part, inherited_weight = tasks.pop()
        if isinstance(part, str):
            append_text(_unescape_text(part), inherited_weight)
            continue

        if part.closed:
            weight = _validate_weight(inherited_weight * part.weight)
            tasks.extend((child, weight) for child in reversed(part.parts))
        else:
            tasks.extend((child, inherited_weight) for child in reversed(part.parts))
            tasks.append(("(", inherited_weight))

    return "".join(cleaned_parts), weighted_ranges


def tokenize_t5_with_weights(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    weighted_ranges: list[WeightedTextRange],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Tokenize cleaned text once, then map character weights to T5 tokens."""
    needs_weights = any(weight != 1.0 for _, _, weight in weighted_ranges)
    if not needs_weights:
        tokens = tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tokens.input_ids[0], None

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

    if all(weight == 1.0 for weight in token_weights):
        return input_ids, None
    return input_ids, torch.tensor(token_weights, dtype=torch.float32)
