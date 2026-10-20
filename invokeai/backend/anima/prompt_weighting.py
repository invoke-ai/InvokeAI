"""Prompt-weighting helpers for Anima text conditioning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase

WeightedTextRange = tuple[int, int, float]
MAX_FLOAT32_WEIGHT = torch.finfo(torch.float32).max


@dataclass
class _Group:
    parts: list[str | "_Group"]
    weight: float | None = None
    closed: bool = True


@dataclass
class _Frame:
    parts: list[str | _Group] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    last_colon: int | None = None

    def flush_text(self) -> None:
        if self.text:
            self.parts.append("".join(self.text))
            self.text.clear()


def _finish_group(frame: _Frame, closed: bool) -> _Group:
    weight = None
    if closed and frame.last_colon is not None:
        raw_text = "".join(frame.text)
        candidate = raw_text[frame.last_colon + 1 :]
        if frame.parts or frame.last_colon > 0:
            try:
                parsed_weight = float(candidate)
            except ValueError:
                pass  # A nonnumeric colon is ordinary prompt text.
            else:
                if not math.isfinite(parsed_weight) or abs(parsed_weight) > MAX_FLOAT32_WEIGHT:
                    raise ValueError("Anima prompt weight must be finite and within float32 magnitude.")
                weight = parsed_weight
                frame.text = list(raw_text[: frame.last_colon])
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
    """Parse explicit (text:weight) groups and return weighted character ranges.

    Bare parentheses stay literal, so existing Anima prompts retain their meaning.
    Escaped parentheses are unescaped without triggering weighting. Parsing and rendering
    use explicit stacks so deeply nested or unbalanced input cannot recurse.
    """
    frames = [_Frame()]
    index = 0
    has_explicit_weight = False
    has_escapes = False

    while index < len(prompt):
        char = prompt[index]
        frame = frames[-1]
        if char == "\\" and index + 1 < len(prompt) and prompt[index + 1] in "()\\":
            has_escapes = True
            frame.text.extend((char, prompt[index + 1]))
            index += 2
        elif char == "(":
            frame.flush_text()
            frame.last_colon = None
            frames.append(_Frame())
            index += 1
        elif char == ")" and len(frames) > 1:
            group = _finish_group(frames.pop(), closed=True)
            has_explicit_weight |= group.weight is not None
            frames[-1].parts.append(group)
            frames[-1].last_colon = None
            index += 1
        else:
            if char == ":" and len(frames) > 1:
                frame.last_colon = len(frame.text)
            frame.text.append(char)
            index += 1

    while len(frames) > 1:
        group = _finish_group(frames.pop(), closed=False)
        frames[-1].parts.append(group)
    frames[0].flush_text()

    if not has_explicit_weight and not has_escapes:
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

    # Only explicit groups consume their surrounding markup.
    tasks: list[tuple[str | _Group, float]] = [(part, 1.0) for part in reversed(frames[0].parts)]
    while tasks:
        part, inherited_weight = tasks.pop()
        if isinstance(part, str):
            append_text(_unescape_text(part), inherited_weight)
            continue

        if part.weight is None:
            if part.closed:
                tasks.append((")", inherited_weight))
            tasks.extend((child, inherited_weight) for child in reversed(part.parts))
            tasks.append(("(", inherited_weight))
        else:
            tasks.extend((child, part.weight) for child in reversed(part.parts))

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
