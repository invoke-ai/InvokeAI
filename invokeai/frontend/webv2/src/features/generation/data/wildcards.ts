/**
 * The current user's wildcards: named value lists that `__name__` expands to.
 *
 * The catalog feeds three consumers — the `__` autocomplete, the highlighter's
 * known/unknown distinction, and the wildcards editor — so it is fetched once
 * and shared. Mutations must invalidate the dynamic prompts cache as well; see
 * `invalidateWildcardDependents`.
 */

import type { QueryClient } from '@tanstack/react-query';

import { dynamicPromptsKeys } from '@features/generation/data/dynamicPromptsQueries';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { apiFetch, apiFetchJson } from '@platform/transport/http';
import { queryOptions } from '@tanstack/react-query';

export interface WildcardRecord {
  id: string;
  name: string;
  values: string[];
}

export const wildcardKeys = {
  all: ['generation', 'wildcards'] as const,
};

export const wildcardsQueryOptions = () =>
  (() => {
    const owner = captureAccountScope();

    return queryOptions({
      queryFn: async ({ signal }): Promise<WildcardRecord[]> => {
        const requestSignal = AbortSignal.any([signal, owner.signal]);
        const wildcards = await apiFetchJson<WildcardRecord[]>('/api/v1/wildcards/', { signal: requestSignal });

        assertAccountScopeCurrent(owner);
        return wildcards;
      },
      queryKey: wildcardKeys.all,
      staleTime: 30_000,
    });
  })();

export const createWildcard = (wildcard: { name: string; values: string[] }): Promise<WildcardRecord> =>
  apiFetchJson('/api/v1/wildcards/', { body: JSON.stringify(wildcard), method: 'POST' });

export const updateWildcard = (id: string, changes: { name?: string; values?: string[] }): Promise<WildcardRecord> =>
  apiFetchJson(`/api/v1/wildcards/${encodeURIComponent(id)}`, {
    body: JSON.stringify(changes),
    method: 'PATCH',
  });

export const deleteWildcard = async (id: string): Promise<void> => {
  await apiFetch(`/api/v1/wildcards/${encodeURIComponent(id)}`, { method: 'DELETE' });
};

/**
 * Editing a wildcard changes what a prompt expands to without changing the
 * prompt, and the expansion cache is keyed only on the request (with
 * `staleTime: Infinity`). Both caches must therefore be dropped together, or the
 * preview and the submitted batch would both keep serving the old expansion.
 */
export const invalidateWildcardDependents = async (queryClient: QueryClient): Promise<void> => {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: wildcardKeys.all }),
    queryClient.invalidateQueries({ queryKey: dynamicPromptsKeys.all }),
  ]);
};
