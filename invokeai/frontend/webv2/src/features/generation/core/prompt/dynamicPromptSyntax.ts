/**
 * Scanner for adieyal/dynamicprompts syntax (`{a|b}`, `{2::a|b}`, `{2-3$$a|b}`,
 * `__wildcard__`, `${variable}`).
 *
 * This is intentionally a standalone character scanner rather than an extension
 * of `ast.ts`. That tokenizer already emits every character we care about as its
 * own whole token — `{`/`}` fall through to its single-char catch-all, `|`/`$`/`:`
 * are single-char punctuation, and `__wildcard__` reads as one word because `_`
 * is a word character. So the ranges produced here always cover whole tokens and
 * compose with the existing highlight annotations, leaving the attention and
 * embedding parser untouched.
 */

import type { PromptRange } from './ast';

export type DynamicPromptSyntaxKind =
  | 'variantBrace'
  | 'variantSeparator'
  | 'variantWeight'
  | 'variantRange'
  | 'wildcard'
  | 'promptVariable'
  | 'error';

export interface DynamicPromptSyntaxAnnotation {
  kind: DynamicPromptSyntaxKind;
  range: PromptRange;
}

const WILDCARD_PATTERN = /__(.+?)__/g;
/** `2::`, `1.5::` — a variant value's relative weight. */
const WEIGHT_PATTERN = /^\s*\d+(?:\.\d+)?::/;
/** `2$$`, `2-3$$`, `-3$$`, and the custom-separator form `2$$, $$`. */
const RANGE_PATTERN = /^\s*(?:\d+)?(?:-\d+)?\$\$(?:[^$|{}]*\$\$)?/;

const isEscaped = (prompt: string, index: number): boolean => {
  let backslashes = 0;

  for (let cursor = index - 1; cursor >= 0 && prompt[cursor] === '\\'; cursor--) {
    backslashes++;
  }

  return backslashes % 2 === 1;
};

const findVariableEnd = (prompt: string, openBraceIndex: number): number => {
  let depth = 0;

  for (let cursor = openBraceIndex; cursor < prompt.length; cursor++) {
    if (prompt[cursor] === '{' && !isEscaped(prompt, cursor)) {
      depth++;
    } else if (prompt[cursor] === '}' && !isEscaped(prompt, cursor)) {
      depth--;

      if (depth === 0) {
        return cursor + 1;
      }
    }
  }

  return -1;
};

const covers = (outer: PromptRange, inner: PromptRange): boolean =>
  outer.start <= inner.start && outer.end >= inner.end;

/**
 * Annotates a prompt's dynamic syntax. Unmatched braces in either direction are
 * reported as `error` so the textarea can underline them the same way it already
 * underlines unbalanced attention parentheses.
 */
export const scanDynamicPromptSyntax = (
  prompt: string,
  /**
   * Wildcard names the backend can resolve. Omitted means "unknown", so callers
   * without a catalog get the neutral wildcard colour rather than a false error.
   */
  knownWildcards?: ReadonlySet<string>
): DynamicPromptSyntaxAnnotation[] => {
  const annotations: DynamicPromptSyntaxAnnotation[] = [];
  const variableRanges: PromptRange[] = [];
  // Open braces awaiting a `}`, paired with the annotation to rewrite if none arrives.
  const openBraces: { annotationIndex: number }[] = [];
  let index = 0;

  const annotateValueStart = (start: number): void => {
    const weight = prompt.slice(start).match(WEIGHT_PATTERN)?.[0];

    if (weight) {
      annotations.push({ kind: 'variantWeight', range: { end: start + weight.length, start } });
    }
  };

  while (index < prompt.length) {
    const char = prompt[index];

    if (char === '\\') {
      index += 2;
      continue;
    }

    if (char === '$' && prompt[index + 1] === '{') {
      const end = findVariableEnd(prompt, index + 1);

      if (end > 0) {
        annotations.push({ kind: 'promptVariable', range: { end, start: index } });
        variableRanges.push({ end, start: index });
        index = end;
        continue;
      }
    }

    if (char === '{') {
      annotations.push({ kind: 'variantBrace', range: { end: index + 1, start: index } });
      openBraces.push({ annotationIndex: annotations.length - 1 });

      const rangePrefix = prompt.slice(index + 1).match(RANGE_PATTERN)?.[0];

      if (rangePrefix) {
        annotations.push({
          kind: 'variantRange',
          range: { end: index + 1 + rangePrefix.length, start: index + 1 },
        });
        index += 1 + rangePrefix.length;
      } else {
        index++;
      }

      annotateValueStart(index);
      continue;
    }

    if (char === '}') {
      annotations.push({
        kind: openBraces.length > 0 ? 'variantBrace' : 'error',
        range: { end: index + 1, start: index },
      });
      openBraces.pop();
      index++;
      continue;
    }

    if (char === '|' && openBraces.length > 0) {
      annotations.push({ kind: 'variantSeparator', range: { end: index + 1, start: index } });
      index++;
      annotateValueStart(index);
      continue;
    }

    index++;
  }

  for (const openBrace of openBraces) {
    annotations[openBrace.annotationIndex].kind = 'error';
  }

  WILDCARD_PATTERN.lastIndex = 0;

  for (let match = WILDCARD_PATTERN.exec(prompt); match; match = WILDCARD_PATTERN.exec(prompt)) {
    const wildcardRange = { end: match.index + match[0].length, start: match.index };

    if (!variableRanges.some((variableRange) => covers(variableRange, wildcardRange))) {
      // An unresolvable wildcard makes the whole prompt fail to expand, so it earns the same
      // treatment as an unbalanced brace rather than looking like ordinary recognised syntax.
      const isUnknown = knownWildcards !== undefined && !knownWildcards.has(match[1]);

      annotations.push({ kind: isUnknown ? 'error' : 'wildcard', range: wildcardRange });
    }
  }

  return annotations;
};
