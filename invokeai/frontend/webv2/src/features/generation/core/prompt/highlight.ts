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
  | 'wildcard'
  | 'promptVariable'
  | 'error';

export interface PromptHighlightOptions {
  /**
   * Annotate dynamic prompting syntax. Only surfaces whose prompt is actually
   * batch-expanded enable this — a `{a|b}` in a regional guidance or negative
   * prompt is literal text, and colouring it would promise an expansion that
   * never happens.
   */
  dynamicPrompts?: boolean;
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
  error: 100,
  wildcard: 45,
  promptVariable: 45,
  variantRange: 42,
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
  embedding: 30,
  error: 100,
  escapedParen: 20,
  group: 20,
  promptFunctionArg: 5,
  promptFunctionMethod: 35,
  promptVariable: 45,
  punctuation: 10,
  text: 0,
  variantBrace: 41,
  variantRange: 42,
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

const getBestAnnotation = (
  annotations: HighlightAnnotation[],
  tokenRange: PromptRange
): HighlightAnnotation | undefined =>
  annotations
    .filter((annotation) => covers(annotation.range, tokenRange))
    .sort((left, right) => right.priority - left.priority)[0];

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

const collectDynamicPromptAnnotations = (prompt: string): HighlightAnnotation[] =>
  scanDynamicPromptSyntax(prompt).map(({ kind, range }) => ({
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
      ...(options.dynamicPrompts ? collectDynamicPromptAnnotations(prompt) : []),
    ];
    const segments: PromptHighlightSegment[] = [];

    for (const token of tokens) {
      const annotation = getBestAnnotation(annotations, token.range);
      const baseKind = tokenKind(token);
      const kind = annotation && annotation.priority > BASE_PRIORITY[baseKind] ? annotation.kind : baseKind;

      appendSegment(segments, {
        kind,
        range: { ...token.range },
        text: prompt.slice(token.range.start, token.range.end),
      });
    }

    return segments;
  } catch {
    return [{ kind: 'text', range: { start: 0, end: prompt.length }, text: prompt }];
  }
};
