import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';

import { hasDynamicPromptSyntax } from '@features/generation/core/dynamicPrompts';
import { dynamicPromptsQueryOptions } from '@features/generation/data/dynamicPromptsQueries';
import { useDebouncedValue } from '@platform/react/useDebouncedValue';
import { useQuery } from '@tanstack/react-query';

/** This debounce follows the form commit debounce. */
const DYNAMIC_PROMPTS_DEBOUNCE_MS = 500;

export interface DynamicPromptsExpansion {
  /** The expanded prompts, or the prompt itself when there is nothing to expand. */
  prompts: string[];
  /** Generations one iteration will produce. */
  count: number;
  /** A backend parse notice; the prompts alongside it are still usable. */
  error: string | null;
  /** The request failed outright, so the literal prompt is what would generate. */
  isError: boolean;
  isLoading: boolean;
  /** The prompt contains `{…}`, so it is subject to expansion. */
  isDynamic: boolean;
}

/** Preview, tooltip, and submit share query identity. */
export const useDynamicPrompts = (prompt: string, config: DynamicPromptsConfig | null): DynamicPromptsExpansion => {
  const isDynamic = Boolean(config) && hasDynamicPromptSyntax(prompt);
  const debouncedPrompt = useDebouncedValue(prompt, DYNAMIC_PROMPTS_DEBOUNCE_MS);
  // Remain loading while the current prompt differs from the debounced expansion.
  const isSettled = debouncedPrompt === prompt;
  const query = useQuery({
    ...dynamicPromptsQueryOptions({
      combinatorial: config?.combinatorial !== false,
      max_prompts: config?.maxPrompts,
      prompt: debouncedPrompt,
      seed: config?.combinatorial === false ? config.sampleSeed : null,
    }),
    enabled: isDynamic && isSettled,
  });

  if (!isDynamic) {
    return { count: 1, error: null, isDynamic: false, isError: false, isLoading: false, prompts: [prompt] };
  }

  const prompts = query.data?.prompts ?? [prompt];

  return {
    count: prompts.length,
    error: query.data?.error ?? null,
    isDynamic: true,
    isError: query.isError,
    isLoading: !isSettled || query.isPending,
    prompts,
  };
};
