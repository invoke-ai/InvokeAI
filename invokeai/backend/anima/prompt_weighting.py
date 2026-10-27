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
METHOD = re.compile(r"\.[A-Za-z_][A-Za-z0-9_]*[ \t\r\n\v\f]*\(")
SPACE = " \t\r\n\v\f"
SYNTAX = '(),.+=-"'
ESCAPES = SYNTAX
QUOTED_WHITESPACE = {"n": "\n", "t": "\t", "r": "\r", "f": "\f"}


@dataclass
class _Part:
    start: int
    end: int
    text: str | None = None
    # Defer copying invalid source until enclosing-group rollback has finished.
    literal_source: bool = False
    children: list["_Part"] = field(default_factory=list)
    weight: float = 1.0
    markup: bool = False
    # Literal call punctuation and authored whitespace keep their source boundaries.
    glue: bool = False


def _validate_weight(weight: float) -> float:
    if not math.isfinite(weight) or abs(weight) > MAX_WEIGHT:
        raise ValueError("Anima prompt weight must be finite and within float16 magnitude.")
    return weight


def _suffix(prompt: str, start: int, numeric: bool) -> tuple[float, int]:
    if start < len(prompt) and prompt[start] in "+-":
        marker = prompt[start]
        end = start + 1
        while end < len(prompt) and prompt[end] == marker:
            end += 1
        try:
            return (1.1 if marker == "+" else 0.9) ** (end - start), end
        except OverflowError:
            return math.inf, end
    match = NUMERIC_SUFFIX.match(prompt, start) if numeric else None
    if match is None:
        return 1.0, start
    end = match.end()
    if prompt[end - 1] == "." and METHOD.match(prompt, end - 1):
        end -= 1
    return float(prompt[start:end]), end


def _attention(prompt: str, start: int, stop: int, numeric: bool) -> tuple[float, int]:
    implicit = _suffix(prompt, start, numeric)
    explicit = start
    while explicit < stop and prompt[explicit] in SPACE:
        explicit += 1
    method = METHOD.match(prompt, explicit)
    if method is None or method.group().split("(")[0].strip() != ".attend":
        return implicit
    value = method.end()
    while value < stop and prompt[value] in SPACE:
        value += 1
    weight, end = _suffix(prompt, value, True)
    close = end
    while close < stop and prompt[close] in SPACE:
        close += 1
    return (weight, close + 1) if end > value and close < stop and prompt[close] == ")" else implicit


def _structures(prompt: str) -> tuple[dict[int, int], dict[int, str], dict[int, int]]:
    """Pair delimiters once; an unclosed enclosing delimiter cannot expose inner syntax."""
    pairs: dict[int, int] = {}
    rejected_quotes: dict[int, int] = {}
    stack: list[int] = []
    quote: int | None = None
    quote_invalid = False
    quote_multiline = False
    quote_escapes: list[tuple[int, str]] = []
    whitespace_escapes: dict[int, str] = {}
    index = 0
    while index < len(prompt):
        char = prompt[index]
        if quote is not None and char in "\v\f":
            quote_invalid = True
        if char == "\\" and quote is not None and index + 1 < len(prompt):
            escaped = prompt[index + 1]
            if escaped == "f":
                # Compel accepts a blank quoted fragment, but rejects form feed among words.
                quote_invalid = True
            if escaped in QUOTED_WHITESPACE:
                quote_escapes.append((index, QUOTED_WHITESPACE[escaped]))
                index += 2
                continue
        if char == "\\" and index + 1 < len(prompt) and prompt[index + 1] in ESCAPES:
            index += 2
            continue
        if char == '"':
            if quote is None:
                quote = index
                quote_invalid = False
                quote_multiline = False
                quote_escapes = []
                stack.append(index)
            else:
                if (
                    stack
                    and stack[-1] == quote
                    and not quote_multiline
                    and (not quote_invalid or not _decoded_text(prompt, quote + 1, index, dict(quote_escapes)).strip())
                ):
                    pairs[stack.pop()] = index
                    whitespace_escapes.update(quote_escapes)
                else:
                    rejected_quotes[quote] = index
                    while stack and stack[-1] != quote:
                        stack.pop()
                    if stack:
                        stack.pop()
                quote = None
        elif char == "(":
            stack.append(index)
        elif char == ")" and stack and prompt[stack[-1]] == "(":
            pairs[stack.pop()] = index
        elif char in "\r\n" and quote is not None:
            # Keep the physical closing boundary even when the quoted target is rejected.
            quote_multiline = True
        index += 1
    return pairs, whitespace_escapes, rejected_quotes


def _clean_literal(text: str) -> str:
    # Compel preserves the backslash on the other syntactic escapes.
    return text.replace("\\(", "(").replace("\\)", ")").replace('\\"', '"')


def _syntax_source(prompt: str, whitespace_escapes: dict[int, str]) -> str:
    """Use padded whitespace for grammar matching while retaining authored source offsets."""
    if not whitespace_escapes:
        return prompt
    parts: list[str] = []
    cursor = 0
    for start, value in whitespace_escapes.items():
        parts.extend((prompt[cursor:start], value, " "))
        cursor = start + 2
    parts.append(prompt[cursor:])
    return "".join(parts)


def _decoded_text(prompt: str, start: int, end: int, whitespace_escapes: dict[int, str]) -> str:
    if not whitespace_escapes:
        return prompt[start:end]
    parts: list[str] = []
    while start < end:
        value = whitespace_escapes.get(start)
        parts.append(value if value is not None else prompt[start])
        start += 2 if value is not None else 1
    return "".join(parts)


def _fallback_end(prompt: str, start: int, index: int, stop: int) -> int:
    """Consume one literal fragment, retaining comma and literal-method boundaries."""
    if index > start and prompt[index - 1] == ",":
        return index
    while index < stop:
        char = prompt[index]
        if char in SPACE or (index > start and METHOD.match(prompt, index)):
            break
        index += 1
        if char == ",":
            break
    return max(start + 1, index)


def parse_prompt_attention(prompt: str) -> tuple[str, list[WeightedTextRange]]:
    """Parse Anima's attention subset of Compel with bounded, iterative stacks.

    Bare fallback text is source-exact. Parsed fragments receive one space at
    otherwise adjacent boundaries; authored ASCII whitespace is retained. Calls
    other than .attend remain literal operations, with attention-active arguments.
    """
    pairs, whitespace_escapes, rejected_quotes = _structures(prompt)
    decoded_syntax: str | None = None
    root = _Part(0, len(prompt))
    # (container, content end, grouped grammar, preserve call delimiters, quoted target context)
    frames: list[tuple[_Part, int, bool, bool, bool]] = [(root, len(prompt), False, False, False)]
    index = 0
    while frames:
        parent, stop, grouped, call, quoted = frames[-1]
        if quoted:
            if decoded_syntax is None:
                decoded_syntax = _syntax_source(prompt, whitespace_escapes)
            syntax = decoded_syntax
        else:
            syntax = prompt
        if index >= stop:
            frames.pop()
            if not frames:
                break
            if call:
                if index < len(prompt):
                    parent.children.append(_Part(index, index + 1, prompt[index], glue=True))
                index = parent.end
                continue
            # Invalid group grammar rolls back the entire source span, including nested attention.
            if parent.literal_source:
                parent.children.clear()
                parent.weight = 1.0
                parent.markup = False
                enclosing, enclosing_stop, enclosing_group, enclosing_call, _ = frames[-1]
                if enclosing_group and not enclosing_call:
                    enclosing.literal_source = True
                    index = enclosing_stop
                    continue
                if prompt[parent.start] == '"':
                    parent.end = _fallback_end(prompt, parent.start, stop + 1, enclosing_stop)
            index = parent.end
            continue

        char = syntax[index]
        if char in SPACE:
            end = index + 1
            while end < stop and syntax[end] in SPACE:
                end += 1
            parent.children.append(_Part(index, end, _decoded_text(prompt, index, end, whitespace_escapes), glue=True))
            index = end
            continue

        method = METHOD.match(syntax, index)
        if method:
            opening = method.end() - 1
            closing = pairs.get(opening, stop)
            node = _Part(index, min(closing + 1, stop))
            node.children.append(
                _Part(index, method.end(), _decoded_text(prompt, index, method.end(), whitespace_escapes), glue=True)
            )
            parent.children.append(node)
            frames.append((node, closing, True, True, quoted))
            index = method.end()
            continue

        if char in '("':
            closing = pairs.get(index)
            if closing is None or closing >= stop:
                if grouped and not call:
                    parent.literal_source = True
                    index = stop
                    continue
                rejected_close = rejected_quotes.get(index)
                if rejected_close is not None and rejected_close < stop:
                    end = _fallback_end(prompt, index, rejected_close + 1, stop)
                    parent.children.append(_Part(index, end, literal_source=True))
                    index = end
                else:
                    parent.children.append(_Part(index, stop, prompt[index:stop]))
                    index = stop
                continue
            weight, end = _attention(syntax, closing + 1, stop, True)
            node = _Part(index, end, weight=weight, markup=True)
            parent.children.append(node)
            frames.append((node, closing, True, False, quoted or char == '"'))
            index += 1
            continue

        if grouped and char in ",.":
            parent.children.append(_Part(index, index + 1, char))
            index += 1
            continue
        # The target scanner stops at syntax. A single sign followed by a bare
        # character stays in the word; repeated signs are a suffix, never rescanned.
        start = index
        while index < stop:
            char = syntax[index]
            if char == "\\" and index + 1 < stop and syntax[index + 1] in ESCAPES:
                index += 2
                if syntax[index - 1] == ",":
                    break
            elif char in "+-":
                if index + 1 < stop and syntax[index + 1] not in SPACE + SYNTAX:
                    index += 1
                else:
                    break
            elif char in SPACE + SYNTAX:
                break
            else:
                index += 1
        target_end = index
        weight, end = _attention(syntax, index, stop, False) if target_end > start else (1.0, index)
        if end > target_end:
            parent.children.append(
                _Part(start, end, _clean_literal(prompt[start:target_end]), weight=weight, markup=True)
            )
            index = end
            continue
        if grouped and target_end > start:
            parent.children.append(_Part(start, target_end, _clean_literal(prompt[start:target_end])))
            continue
        if grouped and not call:
            parent.literal_source = True
            index = stop
            continue
        # Top-level free-word fallback (including a(b)2). Commas have the same
        # boundary as Compel's comma preprocessing, without rewriting plain input.
        index = _fallback_end(syntax, start, max(start, target_end), stop)
        parent.children.append(_Part(start, index, _clean_literal(prompt[start:index])))

    parts: list[str] = []
    ranges: list[WeightedTextRange] = []
    length = 0
    has_markup = False
    previous_glue = True
    tasks = [(part, 1.0) for part in reversed(root.children)]
    while tasks:
        part, inherited = tasks.pop()
        _validate_weight(part.weight)
        weight = _validate_weight(inherited * part.weight)
        has_markup |= part.markup
        if part.text is None and not part.literal_source:
            tasks.extend((child, weight) for child in reversed(part.children))
            continue
        value = prompt[part.start : part.end] if part.literal_source else part.text
        if not value:
            continue
        if parts and not previous_glue and not part.glue and parts[-1][-1] not in SPACE and value[0] not in SPACE:
            parts.append(" ")
            ranges.append((length, length + 1, 1.0))
            length += 1
        start = length
        parts.append(value)
        length += len(value)
        ranges.append((start, length, weight))
        previous_glue = part.glue
    cleaned = "".join(parts)
    if not has_markup and cleaned == prompt:
        return prompt, [(0, len(prompt), 1.0)] if prompt else []
    if not has_markup and not any(c in prompt for c in ("\\(", "\\)", '\\"')):
        return prompt, [(0, len(prompt), 1.0)] if prompt else []
    merged: list[WeightedTextRange] = []
    for start, end, weight in ranges:
        if merged and merged[-1][1] == start and merged[-1][2] == weight:
            merged[-1] = (merged[-1][0], end, weight)
        else:
            merged.append((start, end, weight))
    return cleaned, merged


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
