import type { PromptAstNode, PromptRange, PromptToken } from './ast';

import { parsePromptTokens, tokenizePrompt } from './ast';
import { scanDynamicPromptSyntax } from './dynamicPromptSyntax';

export type PromptHighlightKind =
  | 'text'
  | 'attention'
  | 'attentionNumeric'
  | 'group'
  | 'embedding'
  | 'escapedParen'
  | 'promptFunctionMethod'
  | 'promptFunctionArg'
  | 'punctuation'
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

export interface PromptHighlightOptions {
  /**
   * Annotate dynamic prompting syntax. Only surfaces whose prompt is actually
   * batch-expanded enable this — a `{a|b}` in a regional guidance or negative
   * prompt is literal text, and colouring it would promise an expansion that
   * never happens.
   */
  dynamicPrompts?: boolean;
  /** Resolvable wildcard names; omitted leaves every `__name__` neutral. */
  knownWildcards?: ReadonlySet<string>;
}

export interface PromptHighlightSegment {
  kind: PromptHighlightKind;
  range: PromptRange;
  text: string;
}

interface HighlightAnnotation {
  kind: PromptHighlightKind;
  priority: number;
  range: PromptRange;
}

const ANNOTATION_PRIORITY = {
  // A comment outranks even an error: the parser strips it before it sees the
  // syntax inside, so nothing in there can be wrong.
  comment: 110,
  error: 100,
  promptVariableOperator: 46,
  wildcard: 45,
  promptVariable: 45,
  variantRange: 42,
  variantSampler: 42,
  variantWeight: 42,
  variantBrace: 41,
  variantSeparator: 41,
  embedding: 40,
  promptFunctionMethod: 35,
  promptFunctionArg: 5,
} as const;

const BASE_PRIORITY: Record<PromptHighlightKind, number> = {
  attention: 30,
  attentionNumeric: 30,
  comment: 110,
  embedding: 30,
  error: 100,
  escapedParen: 20,
  group: 20,
  promptFunctionArg: 5,
  promptFunctionMethod: 35,
  promptVariable: 45,
  promptVariableOperator: 46,
  punctuation: 10,
  text: 0,
  variantBrace: 41,
  variantRange: 42,
  variantSampler: 42,
  variantSeparator: 41,
  variantWeight: 42,
  wildcard: 45,
};

const isNumericAttention = (value: unknown): boolean => typeof value === 'number' || !Number.isNaN(Number(value));

const tokenKind = (token: PromptToken): PromptHighlightKind => {
  switch (token.type) {
    case 'weight':
      return isNumericAttention(token.value) ? 'attentionNumeric' : 'attention';
    case 'lparen':
    case 'rparen':
      return 'group';
    case 'lembed':
    case 'rembed':
      return 'embedding';
    case 'escaped_paren':
      return 'escapedParen';
    case 'punct':
      return 'punctuation';
    default:
      return 'text';
  }
};

const covers = (outer: PromptRange, inner: PromptRange): boolean =>
  outer.start <= inner.start && outer.end >= inner.end;

/**
 * Walks the tokens and the annotations together, keeping only the annotations
 * that could still cover the token in hand.
 *
 * This replaced a `filter().sort()` over every annotation for every token. That
 * is quadratic, and dynamic-prompt highlighting made it bite: a prompt full of
 * `{a|b|c}` now produces an annotation per brace, separator, weight, range,
 * sampler, variable, operator, comment and wildcard, so the annotation count
 * grew to be proportional to the token count. At the 20 000-character ceiling
 * the highlighter allows, that was tens of milliseconds of blocked main thread
 * on every keystroke.
 *
 * Both inputs are in document order, and an annotation that ends before the
 * current token cannot reach any later one either, so it is dropped for good.
 * What is left is the nesting depth at that point, which is small.
 */
const forEachTokenAnnotation = (
  tokens: readonly PromptToken[],
  annotations: readonly HighlightAnnotation[],
  visit: (token: PromptToken, annotation: HighlightAnnotation | undefined) => void
): void => {
  // Ties are broken by the order the annotations were collected in, which the
  // stable sort this replaced preserved by accident and callers may rely on.
  const ordered = annotations
    .map((annotation, index) => ({ annotation, index }))
    .sort((left, right) => left.annotation.range.start - right.annotation.range.start || left.index - right.index);
  const active: { annotation: HighlightAnnotation; index: number }[] = [];
  let next = 0;

  for (const token of tokens) {
    while (next < ordered.length && (ordered[next]?.annotation.range.start ?? Infinity) <= token.range.start) {
      const entry = ordered[next];

      if (entry) {
        active.push(entry);
      }

      next++;
    }

    let kept = 0;

    for (const entry of active) {
      if (entry.annotation.range.end >= token.range.end) {
        active[kept] = entry;
        kept++;
      }
    }

    active.length = kept;

    let best: { annotation: HighlightAnnotation; index: number } | undefined;

    for (const entry of active) {
      if (!covers(entry.annotation.range, token.range)) {
        continue;
      }

      if (!best || entry.annotation.priority > best.annotation.priority) {
        best = entry;
      }
    }

    visit(token, best?.annotation);
  }
};

const addPromptFunctionAnnotations = (
  prompt: string,
  node: Extract<PromptAstNode, { type: 'prompt_function' }>,
  annotations: HighlightAnnotation[]
): void => {
  for (const arg of node.promptArgs) {
    annotations.push({
      kind: 'promptFunctionArg',
      priority: ANNOTATION_PRIORITY.promptFunctionArg,
      range: arg.contentRange,
    });
  }

  const lastArg = node.promptArgs.at(-1);
  const methodStart = lastArg ? prompt.indexOf('.', lastArg.contentRange.end) : -1;

  if (methodStart >= 0 && methodStart < node.range.end) {
    annotations.push({
      kind: 'promptFunctionMethod',
      priority: ANNOTATION_PRIORITY.promptFunctionMethod,
      range: { start: methodStart, end: node.range.end },
    });
  }
};

const collectAnnotations = (
  prompt: string,
  nodes: PromptAstNode[],
  annotations: HighlightAnnotation[] = []
): HighlightAnnotation[] => {
  for (const node of nodes) {
    if (node.type === 'embedding') {
      annotations.push({ kind: 'embedding', priority: ANNOTATION_PRIORITY.embedding, range: node.range });
      continue;
    }

    if (node.type === 'group') {
      collectAnnotations(prompt, node.children, annotations);
      continue;
    }

    if (node.type === 'prompt_function') {
      addPromptFunctionAnnotations(prompt, node, annotations);
      collectAnnotations(
        prompt,
        node.promptArgs.flatMap((arg) => arg.nodes),
        annotations
      );
    }
  }

  return annotations;
};

const collectParenthesisErrors = (tokens: PromptToken[]): HighlightAnnotation[] => {
  const annotations: HighlightAnnotation[] = [];
  const stack: PromptToken[] = [];

  for (const token of tokens) {
    if (token.type === 'lparen') {
      stack.push(token);
    } else if (token.type === 'rparen') {
      if (stack.length > 0) {
        stack.pop();
      } else {
        annotations.push({ kind: 'error', priority: ANNOTATION_PRIORITY.error, range: token.range });
      }
    }
  }

  for (const token of stack) {
    annotations.push({ kind: 'error', priority: ANNOTATION_PRIORITY.error, range: token.range });
  }

  return annotations;
};

const appendSegment = (segments: PromptHighlightSegment[], segment: PromptHighlightSegment): void => {
  const previous = segments.at(-1);

  if (previous && previous.kind === segment.kind && previous.range.end === segment.range.start) {
    previous.text += segment.text;
    previous.range.end = segment.range.end;
    return;
  }

  segments.push(segment);
};

const collectDynamicPromptAnnotations = (prompt: string, knownWildcards?: ReadonlySet<string>): HighlightAnnotation[] =>
  scanDynamicPromptSyntax(prompt, knownWildcards).map(({ kind, range }) => ({
    kind,
    priority: ANNOTATION_PRIORITY[kind],
    range,
  }));

export const buildPromptHighlightSegments = (
  prompt: string,
  options: PromptHighlightOptions = {}
): PromptHighlightSegment[] => {
  if (!prompt) {
    return [];
  }

  try {
    const tokens = tokenizePrompt(prompt);
    const annotations = [
      ...collectAnnotations(prompt, parsePromptTokens(tokens)),
      ...collectParenthesisErrors(tokens),
      ...(options.dynamicPrompts ? collectDynamicPromptAnnotations(prompt, options.knownWildcards) : []),
    ];
    const segments: PromptHighlightSegment[] = [];

    forEachTokenAnnotation(tokens, annotations, (token, annotation) => {
      const baseKind = tokenKind(token);
      const kind = annotation && annotation.priority > BASE_PRIORITY[baseKind] ? annotation.kind : baseKind;

      appendSegment(segments, {
        kind,
        range: { ...token.range },
        text: prompt.slice(token.range.start, token.range.end),
      });
    });

    return segments;
  } catch {
    return [{ kind: 'text', range: { start: 0, end: prompt.length }, text: prompt }];
  }
};
