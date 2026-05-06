import { type ASTNode, type Attention,parseTokens, tokenize } from 'common/util/promptAST';
import type { WildcardIndexItem } from 'services/api/endpoints/utilities';

import { normalizeWildcardReference } from './wildcards';

const WILDCARD_OCCURRENCE_REGEX = /__([^\r\n]+?)__/g;

export type PromptRange = {
  start: number;
  end: number;
};

export type PromptWildcardBehavior = 'random' | 'cycle' | 'all' | 'missing' | 'unavailable';

export type PromptWildcardOccurrence = {
  id: string;
  type: 'wildcard';
  token: string;
  path: string;
  rawReference: string;
  range: PromptRange;
  behavior: PromptWildcardBehavior;
  wildcard: WildcardIndexItem | null;
  valueCount: number | null;
};

export type PromptWeightOccurrence = {
  id: string;
  type: 'weight';
  text: string;
  attention: Attention;
  range: PromptRange;
  isSupported: boolean;
};

export type PromptWorkbenchOccurrence = PromptWildcardOccurrence | PromptWeightOccurrence;

type GetPromptWorkbenchOccurrencesArg = {
  prompt: string;
  wildcards: WildcardIndexItem[];
  wildcardIndexUnavailable: boolean;
  dynamicPromptMode: 'random' | 'combinatorial';
  supportsAttentionWeights: boolean;
};

type PromptReplacementResult = {
  prompt: string;
  caret: number;
};

export const getPromptWorkbenchOccurrences = ({
  prompt,
  wildcards,
  wildcardIndexUnavailable,
  dynamicPromptMode,
  supportsAttentionWeights,
}: GetPromptWorkbenchOccurrencesArg): PromptWorkbenchOccurrence[] => {
  return [
    ...getPromptWildcardOccurrences({ prompt, wildcards, wildcardIndexUnavailable, dynamicPromptMode }),
    ...getPromptWeightOccurrences({ prompt, supportsAttentionWeights }),
  ].sort((a, b) => a.range.start - b.range.start || a.range.end - b.range.end);
};

export const getPromptWildcardOccurrences = ({
  prompt,
  wildcards,
  wildcardIndexUnavailable,
  dynamicPromptMode,
}: Omit<GetPromptWorkbenchOccurrencesArg, 'supportsAttentionWeights'>): PromptWildcardOccurrence[] => {
  const occurrences: PromptWildcardOccurrence[] = [];

  for (const match of prompt.matchAll(WILDCARD_OCCURRENCE_REGEX)) {
    const token = match[0];
    const rawReference = match[1] ?? '';
    const start = match.index ?? 0;
    const end = start + token.length;
    const path = normalizeWildcardReference(rawReference);
    const matchingWildcards = findMatchingWildcards(path, wildcards);
    const exactWildcard = wildcards.find((wildcard) => wildcard.path === path) ?? null;
    const isCycle = rawReference.trim().startsWith('@');
    const isMissing = !wildcardIndexUnavailable && matchingWildcards.length === 0;

    occurrences.push({
      id: `wildcard:${start}:${end}`,
      type: 'wildcard',
      token,
      path,
      rawReference,
      range: { start, end },
      behavior: getWildcardBehavior({
        dynamicPromptMode,
        isCycle,
        isMissing,
        wildcardIndexUnavailable,
      }),
      wildcard: exactWildcard,
      valueCount:
        matchingWildcards.length > 0
          ? matchingWildcards.reduce((total, wildcard) => total + wildcard.value_count, 0)
          : null,
    });
  }

  return occurrences;
};

export const getWildcardBehaviorLabel = (
  occurrence: PromptWildcardOccurrence,
  randomRefreshMode: 'manual' | 'per_enqueue'
): string => {
  switch (occurrence.behavior) {
    case 'random':
      return randomRefreshMode === 'per_enqueue' ? 'Random every Invoke' : 'Random preview';
    case 'cycle':
      return 'Cycle';
    case 'all':
      return 'All combinations';
    case 'missing':
      return 'Missing';
    case 'unavailable':
      return 'Unavailable';
  }
};

export const getPromptWeightOccurrences = ({
  prompt,
  supportsAttentionWeights,
}: {
  prompt: string;
  supportsAttentionWeights: boolean;
}): PromptWeightOccurrence[] => {
  try {
    const ast = parseTokens(tokenize(prompt));
    const occurrences: PromptWeightOccurrence[] = [];
    collectPromptWeightOccurrences(ast, prompt, supportsAttentionWeights, occurrences);
    return occurrences;
  } catch {
    return [];
  }
};

export const getWeightBehaviorLabel = (occurrence: PromptWeightOccurrence): string =>
  occurrence.isSupported ? 'Weight supported' : 'Weight may be literal';

export const replacePromptRange = (
  prompt: string,
  range: PromptRange,
  replacement: string
): PromptReplacementResult => {
  return {
    prompt: `${prompt.slice(0, range.start)}${replacement}${prompt.slice(range.end)}`,
    caret: range.start + replacement.length,
  };
};

export const removePromptRange = (prompt: string, range: PromptRange): PromptReplacementResult => {
  const nextPrompt = cleanPromptAfterRemoval(`${prompt.slice(0, range.start)}${prompt.slice(range.end)}`);
  return {
    prompt: nextPrompt,
    caret: Math.min(range.start, nextPrompt.length),
  };
};

const getWildcardBehavior = ({
  dynamicPromptMode,
  isCycle,
  isMissing,
  wildcardIndexUnavailable,
}: {
  dynamicPromptMode: 'random' | 'combinatorial';
  isCycle: boolean;
  isMissing: boolean;
  wildcardIndexUnavailable: boolean;
}): PromptWildcardBehavior => {
  if (wildcardIndexUnavailable) {
    return 'unavailable';
  }

  if (isMissing) {
    return 'missing';
  }

  if (isCycle) {
    return 'cycle';
  }

  if (dynamicPromptMode === 'combinatorial') {
    return 'all';
  }

  return 'random';
};

const findMatchingWildcards = (reference: string, wildcards: WildcardIndexItem[]): WildcardIndexItem[] => {
  if (!reference.includes('*')) {
    return wildcards.filter((wildcard) => wildcard.path === reference);
  }

  const regex = new RegExp(`^${reference.split('*').map(escapeRegExp).join('.*')}$`);
  return wildcards.filter((wildcard) => regex.test(wildcard.path));
};

const collectPromptWeightOccurrences = (
  nodes: ASTNode[],
  prompt: string,
  supportsAttentionWeights: boolean,
  occurrences: PromptWeightOccurrence[]
) => {
  for (const node of nodes) {
    if (node.type === 'word' && node.attention !== undefined) {
      occurrences.push({
        id: `weight:${node.range.start}:${node.range.end}`,
        type: 'weight',
        text: prompt.slice(node.range.start, node.range.end),
        attention: node.attention,
        range: node.range,
        isSupported: supportsAttentionWeights,
      });
    }

    if (node.type === 'group') {
      if (node.attention !== undefined) {
        occurrences.push({
          id: `weight:${node.range.start}:${node.range.end}`,
          type: 'weight',
          text: prompt.slice(node.range.start, node.range.end),
          attention: node.attention,
          range: node.range,
          isSupported: supportsAttentionWeights,
        });
      }
      collectPromptWeightOccurrences(node.children, prompt, supportsAttentionWeights, occurrences);
    }

    if (node.type === 'prompt_function') {
      for (const arg of node.promptArgs) {
        collectPromptWeightOccurrences(arg.nodes, prompt, supportsAttentionWeights, occurrences);
      }
    }
  }
};

const cleanPromptAfterRemoval = (prompt: string): string => {
  return prompt
    .replace(/[ \t]*,[ \t]*,[ \t]*/g, ', ')
    .replace(/[ \t]+,[ \t]*/g, ', ')
    .replace(/^[ \t]*,[ \t]*/g, '')
    .replace(/[ \t]*,[ \t]*$/g, '')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
};

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
