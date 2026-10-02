import type { WildcardRecord } from '@features/generation/data/wildcards';
import type { AccountScope } from '@platform/state/accountLifecycle';

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

/** One write, as `applyWrites` will perform it. */
export interface WildcardWrite {
  name: string;
  values: string[];
  /** Present to update that wildcard; absent to create a new one. */
  id?: string;
}

/** Report completed writes on partial success so retries do not duplicate them. */
export class WildcardWriteError extends Error {
  constructor(
    readonly done: number,
    override readonly cause: unknown
  ) {
    super(cause instanceof Error ? cause.message : String(cause));
    this.name = 'WildcardWriteError';
  }
}

export interface WildcardCatalog {
  wildcards: WildcardRecord[];
  /** Names the backend can resolve, for the highlighter's known/unknown split. */
  knownNames: ReadonlySet<string>;
  isLoading: boolean;
  create: (wildcard: { name: string; values: string[] }) => Promise<void>;
  update: (id: string, changes: { name?: string; values?: string[] }) => Promise<void>;
  remove: (id: string) => Promise<void>;
  /** Invalidate once after the batch; failures include the completed-write count. */
  applyWrites: (writes: readonly WildcardWrite[], owner: AccountScope) => Promise<number>;
}

/** Wildcard edits invalidate otherwise permanent expansion entries. */
export const useWildcards = (): WildcardCatalog => {
  const queryClient = useQueryClient();
  const query = useQuery(wildcardsQueryOptions());
  const wildcards = useMemo(() => query.data ?? [], [query.data]);
  // The backend omits empty wildcards, so they must remain unknown.
  const knownNames = useMemo(
    () => new Set(wildcards.filter((wildcard) => wildcard.values.length > 0).map((wildcard) => wildcard.name)),
    [wildcards]
  );

  // Capture identity separately for every mutation.
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

  const applyWrites = useCallback(
    async (writes: readonly WildcardWrite[], owner: AccountScope): Promise<number> => {
      let done = 0;
      let failure: unknown;

      try {
        for (const write of writes) {
          assertAccountScopeCurrent(owner);

          if (write.id === undefined) {
            await createWildcard({ name: write.name, values: write.values });
          } else {
            await updateWildcard(write.id, { name: write.name, values: write.values });
          }

          assertAccountScopeCurrent(owner);
          done++;
        }
      } catch (caught) {
        failure = caught;
      }

      // Invalidate after partial success, fenced to the originating account.
      if (done > 0) {
        assertAccountScopeCurrent(owner);
        await invalidateWildcardDependents(queryClient);
      }

      if (failure) {
        throw new WildcardWriteError(done, failure);
      }

      return done;
    },
    [queryClient]
  );

  return { applyWrites, create, isLoading: query.isPending, knownNames, remove, update, wildcards };
};
