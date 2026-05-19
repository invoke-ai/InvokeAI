import { type ASTNode, type Attention, parseTokens, tokenize } from 'common/util/promptAST';
import type { DynamicPromptRandomRefreshMode } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import type { WildcardIndexItem } from 'services/api/endpoints/utilities';

import { type PromptWorkbenchTranslation, tx } from './i18n';
import { getCyclicWildcardToken, normalizeWildcardReference } from './wildcards';

const WILDCARD_OCCURRENCE_REGEX = /__([^\r\n]+?)__/g;
const WRAPPED_WILDCARD_WEIGHT_SUFFIX_REGEX = /^[ \t]*\)(?:[+-]+|[+-]?\d+(?:\.\d+)?)$/;

export type PromptRange = {
  start: number;
  end: number;
};

export type PromptWildcardBehavior = 'random' | 'cycle' | 'all' | 'missing' | 'unavailable';
export type WildcardBehaviorIconType = 'random' | 'cycle' | 'all' | 'warning';
export type WildcardBehaviorAction = 'random' | 'cycle' | 'fixed' | 'remove';

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
  weight: PromptWeightOccurrence | null;
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

type WildcardBehaviorActionIntent = {
  replacement?: string;
  opensFixedValues?: boolean;
  removesPrompt?: boolean;
};

export const getPromptWorkbenchOccurrences = ({
  prompt,
  wildcards,
  wildcardIndexUnavailable,
  dynamicPromptMode,
  supportsAttentionWeights,
}: GetPromptWorkbenchOccurrencesArg): PromptWorkbenchOccurrence[] => {
  const wildcardOccurrences = getPromptWildcardOccurrences({
    prompt,
    wildcards,
    wildcardIndexUnavailable,
    dynamicPromptMode,
  });
  const weightOccurrences = getPromptWeightOccurrences({ prompt, supportsAttentionWeights });
  const attachedWeightIds = new Set<string>();
  const weightedWildcardOccurrences = wildcardOccurrences.map((occurrence) => {
    const weight = getWrappingWeightOccurrence(prompt, occurrence, weightOccurrences);
    if (weight) {
      attachedWeightIds.add(weight.id);
    }
    return { ...occurrence, weight };
  });

  return [
    ...weightedWildcardOccurrences,
    ...weightOccurrences.filter((occurrence) => !attachedWeightIds.has(occurrence.id)),
  ].sort((a, b) => getOccurrenceSortRange(a).start - getOccurrenceSortRange(b).start || a.range.end - b.range.end);
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
      weight: null,
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
  randomRefreshMode: DynamicPromptRandomRefreshMode
): PromptWorkbenchTranslation => {
  switch (occurrence.behavior) {
    case 'random':
      if (randomRefreshMode === 'per_image') {
        return tx('promptWorkbench.behavior.randomImageLabel');
      }
      if (randomRefreshMode === 'per_enqueue') {
        return tx('promptWorkbench.behavior.randomInvokeLabel');
      }
      return tx('promptWorkbench.behavior.randomPreviewLabel');
    case 'cycle':
      return tx('promptWorkbench.behavior.cycleLabel');
    case 'all':
      return tx('promptWorkbench.behavior.allCombinationsLabel');
    case 'missing':
      return tx('promptWorkbench.behavior.missingLabel');
    case 'unavailable':
      return tx('promptWorkbench.behavior.unavailableLabel');
  }
};

export const getWildcardBehaviorShortLabel = (
  occurrence: PromptWildcardOccurrence,
  randomRefreshMode: DynamicPromptRandomRefreshMode
): PromptWorkbenchTranslation => {
  switch (occurrence.behavior) {
    case 'random':
      if (randomRefreshMode === 'per_image') {
        return tx('promptWorkbench.behavior.randomImageShort');
      }
      if (randomRefreshMode === 'per_enqueue') {
        return tx('promptWorkbench.behavior.randomInvokeShort');
      }
      return tx('promptWorkbench.behavior.previewShort');
    case 'cycle':
      return tx('promptWorkbench.behavior.cycleShort');
    case 'all':
      return tx('promptWorkbench.behavior.allShort');
    case 'missing':
      return tx('promptWorkbench.behavior.missingLabel');
    case 'unavailable':
      return tx('promptWorkbench.behavior.unavailableLabel');
  }
};

export const getWildcardBehaviorIconType = (occurrence: PromptWildcardOccurrence): WildcardBehaviorIconType => {
  switch (occurrence.behavior) {
    case 'random':
      return 'random';
    case 'cycle':
      return 'cycle';
    case 'all':
      return 'all';
    case 'missing':
    case 'unavailable':
      return 'warning';
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

export const getWeightBehaviorLabel = (occurrence: PromptWeightOccurrence): PromptWorkbenchTranslation =>
  occurrence.isSupported ? tx('promptWorkbench.weight.supportedLabel') : tx('promptWorkbench.weight.literalLabel');

export const getWeightShortLabel = (occurrence: PromptWeightOccurrence): PromptWorkbenchTranslation =>
  occurrence.isSupported
    ? tx('promptWorkbench.weight.valueLabel', { value: String(occurrence.attention) })
    : tx('promptWorkbench.weight.literalShort');

export const getWildcardBehaviorActionIntent = (
  action: WildcardBehaviorAction,
  wildcardPath: string
): WildcardBehaviorActionIntent => {
  switch (action) {
    case 'random':
      return { replacement: `__${wildcardPath}__` };
    case 'cycle':
      return { replacement: getCyclicWildcardToken(wildcardPath) };
    case 'fixed':
      return { opensFixedValues: true };
    case 'remove':
      return { removesPrompt: true };
  }
};

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

const getWrappingWeightOccurrence = (
  prompt: string,
  wildcardOccurrence: PromptWildcardOccurrence,
  weightOccurrences: PromptWeightOccurrence[]
): PromptWeightOccurrence | null => {
  const candidates = weightOccurrences
    .filter((weightOccurrence) => getDoesWeightCleanlyWrapWildcard(prompt, weightOccurrence, wildcardOccurrence))
    .sort((a, b) => a.range.end - a.range.start - (b.range.end - b.range.start));

  return candidates[0] ?? null;
};

const getDoesWeightCleanlyWrapWildcard = (
  prompt: string,
  weightOccurrence: PromptWeightOccurrence,
  wildcardOccurrence: PromptWildcardOccurrence
): boolean => {
  if (
    weightOccurrence.range.start >= wildcardOccurrence.range.start ||
    weightOccurrence.range.end <= wildcardOccurrence.range.end
  ) {
    return false;
  }

  return (
    /^[ \t]*\([ \t]*$/.test(prompt.slice(weightOccurrence.range.start, wildcardOccurrence.range.start)) &&
    WRAPPED_WILDCARD_WEIGHT_SUFFIX_REGEX.test(prompt.slice(wildcardOccurrence.range.end, weightOccurrence.range.end))
  );
};

const getOccurrenceSortRange = (occurrence: PromptWorkbenchOccurrence): PromptRange =>
  occurrence.type === 'wildcard' && occurrence.weight ? occurrence.weight.range : occurrence.range;

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
