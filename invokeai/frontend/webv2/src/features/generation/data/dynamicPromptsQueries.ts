/** Share deterministic expansion queries and prevent stale results across consumers. */

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

/** Use fetchQuery: ensureQueryData can return invalidated entries despite infinite staleTime. */
export const resolveDynamicPrompts = (
  queryClient: QueryClient,
  request: ParseDynamicPromptsRequest
): Promise<ParseDynamicPromptsResponse> => queryClient.fetchQuery(dynamicPromptsQueryOptions(request));
