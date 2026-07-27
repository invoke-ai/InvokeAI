import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';

import { hasDynamicPromptSyntax } from '@features/generation/core/dynamicPrompts';
import { dynamicPromptsQueryOptions } from '@features/generation/data/dynamicPromptsQueries';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';

/**
 * On top of the Generate form's 250ms commit debounce, so a burst of typing
 * settles before it costs a round trip.
 */
const DYNAMIC_PROMPTS_DEBOUNCE_MS = 500;

const useDebouncedValue = <Value>(value: Value, delayMs: number): Value => {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timeout = window.setTimeout(() => setDebouncedValue(value), delayMs);

    return () => window.clearTimeout(timeout);
  }, [delayMs, value]);

  return debouncedValue;
};

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

/**
 * Reads the shared expansion cache. The Generate preview, the Invoke tooltip,
 * and the submit path all address the same query key, so observing here neither
 * duplicates a request nor goes stale relative to what is submitted.
 */
export const useDynamicPrompts = (prompt: string, config: DynamicPromptsConfig | null): DynamicPromptsExpansion => {
  const isDynamic = Boolean(config) && hasDynamicPromptSyntax(prompt);
  const debouncedPrompt = useDebouncedValue(prompt, DYNAMIC_PROMPTS_DEBOUNCE_MS);
  // A prompt that changed since the last debounce tick has a stale expansion, so
  // report loading rather than a count the user is about to see change.
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
