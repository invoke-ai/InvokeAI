/**
 * The one cache for prompt expansion. The preview popover and the Invoke tooltip
 * observe it through `useQuery`; the enqueue path reads it through
 * `resolveDynamicPrompts`. Because all three address the same key, invoking
 * normally costs no extra round trip, and invoking before the preview has
 * settled still enqueues the prompts the backend actually produces rather than a
 * stale list.
 *
 * Expansion is deterministic for a given request — combinatorial by
 * construction, and seeded when random — so entries never go stale on their own.
 */

import type {
  ParseDynamicPromptsRequest,
  ParseDynamicPromptsResponse,
} from '@features/generation/data/promptUtilities';
import type { QueryClient } from '@tanstack/react-query';

import { parseDynamicPrompts } from '@features/generation/data/promptUtilities';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { queryOptions } from '@tanstack/react-query';

const DYNAMIC_PROMPTS_GC_TIME = 30 * 60 * 1000;

export const dynamicPromptsKeys = {
  all: ['generation', 'dynamic-prompts'] as const,
  expansion: (request: ParseDynamicPromptsRequest) =>
    [
      ...dynamicPromptsKeys.all,
      request.prompt,
      request.combinatorial !== false,
      request.max_prompts ?? null,
      request.combinatorial === false ? (request.seed ?? null) : null,
    ] as const,
};

export const dynamicPromptsQueryOptions = (request: ParseDynamicPromptsRequest) =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      gcTime: DYNAMIC_PROMPTS_GC_TIME,
      queryFn: async ({ signal }): Promise<ParseDynamicPromptsResponse> => {
        const requestSignal = AbortSignal.any([signal, owner.signal]);
        const response = await parseDynamicPrompts(request, requestSignal);

        assertAccountScopeCurrent(owner);
        return response;
      },
      queryKey: dynamicPromptsKeys.expansion(request),
      retry: false,
      staleTime: Infinity,
    });
  })();

/** Cache-first expansion for the enqueue path, which cannot render a loading state. */
export const resolveDynamicPrompts = (
  queryClient: QueryClient,
  request: ParseDynamicPromptsRequest
): Promise<ParseDynamicPromptsResponse> => queryClient.ensureQueryData(dynamicPromptsQueryOptions(request));
