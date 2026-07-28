/**
 * Scanner for adieyal/dynamicprompts syntax.
 *
 * ## The syntax, in full
 *
 * The backend runs the real `dynamicprompts` library, so everything below works
 * at generate time whether or not this scanner colours it. Kept here, beside the
 * code that recognizes each form, so the two cannot drift apart.
 *
 * | Form | Meaning | Example |
 * | --- | --- | --- |
 * | `{a\|b}` | Pick one alternative | `a {red\|green} ball` |
 * | `{2$$a\|b\|c}` | Pick exactly two, comma-joined | `{2$$red\|green\|blue}` |
 * | `{1-3$$a\|b\|c}` | Pick between one and three | `{1-3$$red\|green\|blue}` |
 * | `{2$$ and $$a\|b}` | Pick two, joined by a custom separator | `{2$$ and $$red\|green}` |
 * | `{2::a\|b}` | Weight an alternative — `a` twice as likely | `{2::red\|green}` |
 * | `{~a\|b}` | Sample randomly, ignoring combinatorial mode | `{~red\|green}` |
 * | `{@a\|b}` | Cycle through the alternatives in order | `{@red\|green}` |
 * | `__name__` | Substitute one value from a wildcard | `a __colours__ ball` |
 * | `__a/b__` | Nested wildcard names | `__animals/dogs__` |
 * | `__art/*__` | Glob — draws from every matching wildcard | `__artists/*__` |
 * | `__~name__` | Sampler override on a wildcard | `__@colours__` |
 * | `__name(k=v)__` | Bind `${k}` inside that wildcard's values | `__outfit(mood=warm)__` |
 * | `${v=a}` | Assign a variable; expands to nothing itself | `${lens=85mm} shot on ${lens}` |
 * | `${v=!{a\|b}}` | Assign once, immediately — every use is the same value | `${v=!{a\|b}} ${v} ${v}` |
 * | `${v}` | Use a variable | `${lens}` |
 * | `${v:fallback}` | Use a variable, or this if it is unset | `${lens:50mm}` |
 * | `# …` | Comment to end of line; stripped before generating | `a cat # for now` |
 *
 * Written to be lifted wholesale into the app-wide help system when that lands.
 *
 * ## Why a separate scanner
 *
 * This is intentionally a standalone character scanner rather than an extension
 * of `ast.ts`. That tokenizer already emits every character we care about as its
 * own whole token — `{`/`}` fall through to its single-char catch-all, `|`/`$`/`:`
 * are single-char punctuation, and `__wildcard__` reads as one word because `_`
 * is a word character. So the ranges produced here always cover whole tokens and
 * compose with the existing highlight annotations, leaving the attention and
 * embedding parser untouched.
 */

import { matchesKnownWildcard, WILDCARD_REFERENCE_RE } from '@features/generation/core/dynamicPrompts';

import type { PromptRange } from './ast';

export type DynamicPromptSyntaxKind =
  | 'variantBrace'
  | 'variantSeparator'
  | 'variantWeight'
  | 'variantRange'
  | 'variantSampler'
  | 'wildcard'
  | 'promptVariable'
  | 'promptVariableOperator'
  | 'comment'
  | 'error';

export interface DynamicPromptSyntaxAnnotation {
  kind: DynamicPromptSyntaxKind;
  range: PromptRange;
}

/** `2::`, `1.5::` — a variant value's relative weight. */
const WEIGHT_PATTERN = /^\s*\d+(?:\.\d+)?::/;
/** `2$$`, `2-3$$`, `-3$$`, and the custom-separator form `2$$, $$`. */
const RANGE_PATTERN = /^\s*(?:\d+)?(?:-\d+)?\$\$(?:[^$|{}]*\$\$)?/;
/**
 * The operator that makes a `${…}` an assignment rather than a use: `=`, the
 * immediate `=!`, or the `:` introducing a fallback. Muted like a variant's `|`,
 * which is enough to tell a definition from a reference at a glance without
 * spending another colour on it.
 */
const VARIABLE_OPERATOR_PATTERN = /=!?|:/;

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
  // Spans the wildcard sweep below must not look inside: a `__name__` written in
  // either is not a reference, and reporting one would be a mistake the renderer
  // then has to out-prioritize.
  const variableRanges: PromptRange[] = [];
  const commentRanges: PromptRange[] = [];
  // Open braces awaiting a `}`, paired with the annotation to rewrite if none arrives.
  const openBraces: { annotationIndex: number }[] = [];
  let index = 0;

  const annotateValueStart = (start: number): void => {
    const weight = prompt.slice(start).match(WEIGHT_PATTERN)?.[0];

    if (weight) {
      annotations.push({ kind: 'variantWeight', range: { end: start + weight.length, start } });
    }
  };

  /**
   * The first operator inside a `${…}`, over the name and nothing after it — a
   * `:` or `=` further in belongs to the assigned value, which may itself be a
   * whole nested prompt.
   */
  const annotateVariableOperator = (contentStart: number, contentEnd: number): void => {
    const name = prompt.slice(contentStart, contentEnd);
    const operator = name.match(VARIABLE_OPERATOR_PATTERN);

    if (operator?.index !== undefined) {
      const start = contentStart + operator.index;

      annotations.push({ kind: 'promptVariableOperator', range: { end: start + operator[0].length, start } });
    }
  };

  while (index < prompt.length) {
    const char = prompt[index];

    // Comments outrank everything: the parser strips them before it sees any of
    // the syntax below, so an unbalanced brace or an unknown wildcard inside one
    // is inert text rather than a mistake.
    if (char === '#') {
      const lineEnd = prompt.indexOf('\n', index);
      const end = lineEnd === -1 ? prompt.length : lineEnd;

      annotations.push({ kind: 'comment', range: { end, start: index } });
      commentRanges.push({ end, start: index });
      index = end;
      continue;
    }

    if (char === '\\') {
      // `#` is the one character a backslash does not escape: upstream still
      // comments from it and leaves the backslash in the prompt. Stepping over
      // just the backslash lets the branch above see it on the next pass.
      index += prompt[index + 1] === '#' ? 1 : 2;
      continue;
    }

    if (char === '$' && prompt[index + 1] === '{') {
      const end = findVariableEnd(prompt, index + 1);

      if (end > 0) {
        annotations.push({ kind: 'promptVariable', range: { end, start: index } });
        variableRanges.push({ end, start: index });
        annotateVariableOperator(index + 2, end - 1);
        index = end;
        continue;
      }
    }

    if (char === '{') {
      annotations.push({ kind: 'variantBrace', range: { end: index + 1, start: index } });
      openBraces.push({ annotationIndex: annotations.length - 1 });

      // The sampler override binds tightest — `{~2$$a|b}` samples randomly, while
      // `{2$$~a|b}` has `~a` as an ordinary alternative.
      if (prompt[index + 1] === '~' || prompt[index + 1] === '@') {
        annotations.push({ kind: 'variantSampler', range: { end: index + 2, start: index + 1 } });
        index++;
      }

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

  const wildcardPattern = new RegExp(WILDCARD_REFERENCE_RE, 'g');
  // Concatenated once. Built inside the loop, this allocated a fresh array of
  // every variable and comment in the prompt for every wildcard reference in it.
  const enclosingRanges = [...variableRanges, ...commentRanges];

  for (let match = wildcardPattern.exec(prompt); match; match = wildcardPattern.exec(prompt)) {
    const wildcardRange = { end: match.index + match[0].length, start: match.index };

    if (!enclosingRanges.some((outer) => covers(outer, wildcardRange))) {
      // An unresolvable wildcard survives expansion as its own literal text, so it earns the
      // same treatment as an unbalanced brace rather than looking like ordinary recognised
      // syntax. `matchesKnownWildcard` resolves globs, which is why a valid `__artists/*__`
      // is no longer flagged for the sin of not being a name.
      const isUnknown = knownWildcards !== undefined && !matchesKnownWildcard(match[1], knownWildcards);

      annotations.push({ kind: isUnknown ? 'error' : 'wildcard', range: wildcardRange });
    }
  }

  return annotations;
};
