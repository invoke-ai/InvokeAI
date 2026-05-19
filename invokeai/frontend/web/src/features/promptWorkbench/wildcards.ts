import type { WildcardIndexItem } from 'services/api/endpoints/utilities';

import { type PromptWorkbenchTranslation, tx } from './i18n';

const WILDCARD_REFERENCE_REGEX = /__([^\r\n]+?)__/g;
const WILDCARD_COMPLETION_STOP_CHARS = new Set([',', ';', '{', '}', '[', ']', '(', ')']);

export type WildcardCompletionContext = {
  start: number;
  end: number;
  query: string;
};

export type WildcardCompletionResult = {
  prompt: string;
  caret: number;
};

type GetWildcardAutocompleteStatusMessageArg = {
  isLoading: boolean;
  isUnavailable: boolean;
  optionCount: number;
  query: string;
  wildcardCount: number;
};

export const getWildcardCompletionContext = (prompt: string, caret: number): WildcardCompletionContext | null => {
  const beforeCaret = prompt.slice(0, caret);
  const delimiterCount = beforeCaret.match(/__/g)?.length ?? 0;
  if (delimiterCount % 2 === 0) {
    return null;
  }

  const start = beforeCaret.lastIndexOf('__');

  if (start < 0) {
    return null;
  }

  const query = beforeCaret.slice(start + 2);
  if (query.includes('__') || [...query].some((char) => WILDCARD_COMPLETION_STOP_CHARS.has(char) || /\s/.test(char))) {
    return null;
  }

  return { start, end: caret, query };
};

export const filterWildcardOptions = (wildcards: WildcardIndexItem[], query: string): WildcardIndexItem[] => {
  const normalizedQuery = query.toLowerCase();
  return wildcards
    .filter((wildcard) => {
      if (!normalizedQuery) {
        return true;
      }
      return (
        wildcard.path.toLowerCase().includes(normalizedQuery) ||
        wildcard.label.toLowerCase().includes(normalizedQuery) ||
        wildcard.samples.some((sample) => sample.toLowerCase().includes(normalizedQuery))
      );
    })
    .slice(0, 30);
};

export const applyWildcardCompletion = (
  prompt: string,
  context: WildcardCompletionContext,
  replacement: string
): WildcardCompletionResult => {
  return {
    prompt: `${prompt.slice(0, context.start)}${replacement}${prompt.slice(context.end)}`,
    caret: context.start + replacement.length,
  };
};

export const getCyclicWildcardToken = (wildcardPath: string): string => `__@${wildcardPath}__`;

export const getWildcardDisplayPath = (wildcard: Pick<WildcardIndexItem, 'path'>): string => wildcard.path;

export const getWildcardAutocompleteStatusMessage = ({
  isLoading,
  isUnavailable,
  optionCount,
  query,
  wildcardCount,
}: GetWildcardAutocompleteStatusMessageArg): PromptWorkbenchTranslation | null => {
  if (optionCount > 0) {
    return null;
  }

  if (isLoading) {
    return tx('promptWorkbench.autocomplete.loading');
  }

  if (isUnavailable) {
    return tx('promptWorkbench.autocomplete.unavailable');
  }

  if (wildcardCount === 0) {
    return tx('promptWorkbench.autocomplete.empty');
  }

  if (query) {
    return tx('promptWorkbench.autocomplete.noMatches', { query });
  }

  return tx('promptWorkbench.autocomplete.noWildcardMatches');
};

export const getWildcardReferences = (prompt: string): string[] => {
  const refs: string[] = [];
  const seen = new Set<string>();
  for (const match of prompt.matchAll(WILDCARD_REFERENCE_REGEX)) {
    const ref = normalizeWildcardReference(match[1] ?? '');
    if (!ref || seen.has(ref)) {
      continue;
    }
    seen.add(ref);
    refs.push(ref);
  }
  return refs;
};

export const getMissingWildcardReferences = (prompt: string, wildcards: WildcardIndexItem[]): string[] => {
  const available = new Set(wildcards.map((wildcard) => wildcard.path));
  return getWildcardReferences(prompt).filter((ref) => !wildcardReferenceExists(ref, available));
};

export const normalizeWildcardReference = (reference: string): string => {
  let path = reference.trim();
  if (path.startsWith('~') || path.startsWith('@')) {
    path = path.slice(1);
  }
  if (path.includes('(')) {
    path = path.split('(', 1)[0] ?? '';
  }
  return path.replaceAll('\\', '/').replace(/^\/+|\/+$/g, '');
};

const wildcardReferenceExists = (reference: string, available: Set<string>): boolean => {
  if (!reference.includes('*')) {
    return available.has(reference);
  }

  const regex = new RegExp(`^${reference.split('*').map(escapeRegExp).join('.*')}$`);
  return [...available].some((path) => regex.test(path));
};

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
