/** Catalog edits invalidate shared expansion queries too. */

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

/** The most ids one batch delete takes (`MAX_WILDCARD_BATCH_SIZE` on the server). */
export const WILDCARD_DELETE_BATCH_SIZE = 1000;

/**
 * Deletes those of the ids the caller owns; the result names them. Ids the server does not know as the caller's are
 * absent from it, so a wildcard deleted meanwhile reads as not deleted here.
 */
export const deleteWildcards = async (ids: readonly string[], signal?: AbortSignal): Promise<string[]> => {
  const result = await apiFetchJson<{ deleted_ids: string[] }>('/api/v1/wildcards/delete', {
    body: JSON.stringify({ wildcard_ids: ids }),
    method: 'POST',
    signal,
  });

  return result.deleted_ids;
};

/** Infinite staleTime and request-only keys require dependent invalidation and staleness-aware readers. */
export const invalidateWildcardDependents = async (queryClient: QueryClient): Promise<void> => {
  await Promise.all([
    queryClient.invalidateQueries({ queryKey: wildcardKeys.all }),
    queryClient.invalidateQueries({ queryKey: dynamicPromptsKeys.all }),
  ]);
};
