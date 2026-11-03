import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';

import { expandsDynamicPrompts } from '@features/generation/core/dynamicPrompts';
import { dynamicPromptsQueryOptions } from '@features/generation/data/dynamicPromptsQueries';
import { useDebouncedValue } from '@platform/react/useDebouncedValue';
import { useQuery } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

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
  /** A retry of a failed request is on its way. */
  isRetrying: boolean;
  /** Dynamic prompts are on and the prompt has their syntax, so it is subject to expansion. */
  isDynamic: boolean;
  /** Asks again after a failed request; a failure is kept until the prompt changes or this runs. */
  retry: () => void;
}

const NO_RETRY = () => {};

/** Preview, tooltip, and submit share query identity. */
export const useDynamicPrompts = (prompt: string, config: DynamicPromptsConfig | null): DynamicPromptsExpansion => {
  const isDynamic = config !== null && expandsDynamicPrompts(prompt, config.enabled);
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
  const { refetch } = query;
  const retry = useCallback(() => void refetch(), [refetch]);

  const data = query.data;
  const isError = query.isError;
  const isLoading = !isSettled || query.isPending;
  const isRetrying = query.isError && query.isFetching;

  // One object per distinct state, so the prompt field's memoized actions hold between keystrokes.
  return useMemo((): DynamicPromptsExpansion => {
    if (!isDynamic) {
      return {
        count: 1,
        error: null,
        isDynamic: false,
        isError: false,
        isLoading: false,
        isRetrying: false,
        prompts: [prompt],
        retry: NO_RETRY,
      };
    }

    const prompts = data?.prompts ?? [prompt];

    return {
      count: prompts.length,
      error: data?.error ?? null,
      isDynamic: true,
      isError,
      isLoading,
      isRetrying,
      prompts,
      retry,
    };
  }, [data, isDynamic, isError, isLoading, isRetrying, prompt, retry]);
};
