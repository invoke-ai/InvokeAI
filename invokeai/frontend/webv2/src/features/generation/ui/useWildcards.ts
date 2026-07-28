import type { WildcardRecord } from '@features/generation/data/wildcards';

import {
  createWildcard,
  deleteWildcard,
  invalidateWildcardDependents,
  updateWildcard,
  wildcardsQueryOptions,
} from '@features/generation/data/wildcards';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';

export interface WildcardCatalog {
  wildcards: WildcardRecord[];
  /** Names the backend can resolve, for the highlighter's known/unknown split. */
  knownNames: ReadonlySet<string>;
  isLoading: boolean;
  create: (wildcard: { name: string; values: string[] }) => Promise<void>;
  update: (id: string, changes: { name?: string; values?: string[] }) => Promise<void>;
  remove: (id: string) => Promise<void>;
}

/**
 * The shared wildcard catalog. Every mutation invalidates the dynamic prompts
 * cache too, because editing a wildcard changes what an unchanged prompt expands
 * to and that cache never goes stale on its own.
 */
export const useWildcards = (): WildcardCatalog => {
  const queryClient = useQueryClient();
  const query = useQuery(wildcardsQueryOptions());
  const wildcards = useMemo(() => query.data ?? [], [query.data]);
  // Only a wildcard with values resolves; the backend omits empty ones from its
  // manager, so an empty one must read as unknown here too.
  const knownNames = useMemo(
    () => new Set(wildcards.filter((wildcard) => wildcard.values.length > 0).map((wildcard) => wildcard.name)),
    [wildcards]
  );

  // Each mutation captures the identity that started it, so a sign-out mid-flight
  // does not invalidate the next account's caches on the way back. Stated once,
  // as `usePromptTemplates` does — three copies of it was three chances for the
  // next mutation added here to forget a step.
  const runAndInvalidate = useCallback(
    async (run: () => Promise<unknown>): Promise<void> => {
      const owner = captureAccountScope();

      await run();
      assertAccountScopeCurrent(owner);
      await invalidateWildcardDependents(queryClient);
    },
    [queryClient]
  );

  const create = useCallback(
    (wildcard: { name: string; values: string[] }) => runAndInvalidate(() => createWildcard(wildcard)),
    [runAndInvalidate]
  );

  const update = useCallback(
    (id: string, changes: { name?: string; values?: string[] }) => runAndInvalidate(() => updateWildcard(id, changes)),
    [runAndInvalidate]
  );

  const remove = useCallback((id: string) => runAndInvalidate(() => deleteWildcard(id)), [runAndInvalidate]);

  return { create, isLoading: query.isPending, knownNames, remove, update, wildcards };
};
