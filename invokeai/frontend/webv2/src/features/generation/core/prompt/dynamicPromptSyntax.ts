import { matchesKnownWildcard, scanWildcardReferences } from '@features/generation/core/dynamicPrompts';

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

const WEIGHT_PATTERN = /^\s*\d+(?:\.\d+)?::/;
const RANGE_PATTERN = /^\s*(?:\d+)?(?:-\d+)?\$\$(?:[^$|{}]*\$\$)?/;
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

export const scanDynamicPromptSyntax = (
  prompt: string,
  knownWildcards?: ReadonlySet<string>
): DynamicPromptSyntaxAnnotation[] => {
  const annotations: DynamicPromptSyntaxAnnotation[] = [];
  const variableRanges: PromptRange[] = [];
  const commentRanges: PromptRange[] = [];
  const openBraces: { annotationIndex: number }[] = [];
  let index = 0;

  const annotateValueStart = (start: number): void => {
    const weight = prompt.slice(start).match(WEIGHT_PATTERN)?.[0];

    if (weight) {
      annotations.push({ kind: 'variantWeight', range: { end: start + weight.length, start } });
    }
  };

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

    if (char === '#') {
      const lineEnd = prompt.indexOf('\n', index);
      const end = lineEnd === -1 ? prompt.length : lineEnd;

      annotations.push({ kind: 'comment', range: { end, start: index } });
      commentRanges.push({ end, start: index });
      index = end;
      continue;
    }

    if (char === '\\') {
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

  const enclosingRanges = [...variableRanges, ...commentRanges];

  for (const reference of scanWildcardReferences(prompt)) {
    const wildcardRange = reference.range;

    if (!enclosingRanges.some((outer) => covers(outer, wildcardRange))) {
      const isUnknown = knownWildcards !== undefined && !matchesKnownWildcard(reference.lookupPath, knownWildcards);

      annotations.push({ kind: isUnknown ? 'error' : 'wildcard', range: wildcardRange });
    }
  }

  return annotations;
};
