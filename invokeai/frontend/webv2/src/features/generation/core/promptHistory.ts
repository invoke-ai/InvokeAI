import type { PromptHistoryItem } from '@features/generation/core/contracts';

import type { GenerateSettings } from './types';

import { searchProse } from './catalogSearch';

export const MAX_PROMPT_HISTORY = 100;

export const normalizePromptHistoryItem = (item: PromptHistoryItem): PromptHistoryItem | null => {
  const positivePrompt = item.positivePrompt.trim();
  const negativePrompt = item.negativePrompt?.trim() || null;

  if (positivePrompt.length === 0 && !negativePrompt) {
    return null;
  }

  return { negativePrompt, positivePrompt };
};

const isSamePromptHistoryItem = (a: PromptHistoryItem, b: PromptHistoryItem): boolean =>
  a.positivePrompt === b.positivePrompt && (a.negativePrompt ?? null) === (b.negativePrompt ?? null);

export const addPromptHistoryItem = (
  history: readonly PromptHistoryItem[],
  item: PromptHistoryItem
): PromptHistoryItem[] => {
  const prompt = normalizePromptHistoryItem(item);

  if (!prompt) {
    return [...history];
  }

  return [prompt, ...history.filter((entry) => !isSamePromptHistoryItem(entry, prompt))].slice(0, MAX_PROMPT_HISTORY);
};

export const removePromptHistoryItem = (
  history: readonly PromptHistoryItem[],
  item: PromptHistoryItem
): PromptHistoryItem[] => history.filter((entry) => !isSamePromptHistoryItem(entry, item));

/** A history entry has no name — it is its two prompts. */
const getPromptHistoryProse = (item: PromptHistoryItem): readonly string[] => [
  item.positivePrompt,
  item.negativePrompt ?? '',
];

export const filterPromptHistory = (history: readonly PromptHistoryItem[], searchTerm: string): PromptHistoryItem[] =>
  searchProse(history, searchTerm, getPromptHistoryProse);

export const getPromptHistoryItemFromGenerateSettings = (settings: GenerateSettings): PromptHistoryItem => ({
  negativePrompt: settings.negativePromptEnabled ? settings.negativePrompt : null,
  positivePrompt: settings.positivePrompt,
});
