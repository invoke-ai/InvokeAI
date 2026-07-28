import type { WildcardRecord } from '@features/generation/data/wildcards';

import {
  createWildcard,
  deleteWildcard,
  invalidateWildcardDependents,
  updateWildcard,
  wildcardsQueryOptions,
} from '@features/generation/data/wildcards';
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

  const create = useCallback(
    async (wildcard: { name: string; values: string[] }) => {
      await createWildcard(wildcard);
      await invalidateWildcardDependents(queryClient);
    },
    [queryClient]
  );

  const update = useCallback(
    async (id: string, changes: { name?: string; values?: string[] }) => {
      await updateWildcard(id, changes);
      await invalidateWildcardDependents(queryClient);
    },
    [queryClient]
  );

  const remove = useCallback(
    async (id: string) => {
      await deleteWildcard(id);
      await invalidateWildcardDependents(queryClient);
    },
    [queryClient]
  );

  return { create, isLoading: query.isPending, knownNames, remove, update, wildcards };
};
