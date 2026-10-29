import type { WildcardRecord } from '@features/generation/data/wildcards';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';

import {
  createWildcard,
  deleteWildcard,
  deleteWildcards,
  invalidateWildcardDependents,
  updateWildcard,
  WILDCARD_DELETE_BATCH_SIZE,
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

/** Report the ids a failed batch delete removed before it failed, so the selection keeps only the rest. */
export class WildcardDeleteError extends Error {
  constructor(
    readonly deletedIds: readonly string[],
    override readonly cause: unknown
  ) {
    super(cause instanceof Error ? cause.message : String(cause));
    this.name = 'WildcardDeleteError';
  }
}

export interface WildcardCatalog {
  wildcards: WildcardRecord[];
  /** Names the backend can resolve, for the highlighter's known/unknown split. */
  knownNames: ReadonlySet<string>;
  isLoading: boolean;
  /** The last read failed; any wildcards listed are from an earlier read. */
  isError: boolean;
  /** A retry after a failed read is on its way. */
  isRetrying: boolean;
  retry: () => void;
  create: (wildcard: { name: string; values: string[] }) => Promise<void>;
  update: (id: string, changes: { name?: string; values?: string[] }) => Promise<void>;
  remove: (id: string) => Promise<void>;
  /**
   * Deletes the wildcards in batches the server accepts, one request at a time, and invalidates once at the end;
   * resolves with the deleted ids. A failure rejects with a `WildcardDeleteError` naming those deleted before it.
   */
  removeMany: (ids: readonly string[]) => Promise<string[]>;
  /** Invalidate once after the batch; failures include the completed-write count. */
  applyWrites: (writes: readonly WildcardWrite[], owner: AccountScope) => Promise<number>;
}

/**
 * Invalidate after an attempted write, fenced to the account it started under: a write whose answer was lost may still
 * have committed. A success waits for the refreshed catalog; a failure reports at once while the refresh, which may
 * retry against the same outage, runs behind it.
 */
const invalidateAfterAttempt = async (
  queryClient: QueryClient,
  owner: AccountScope,
  failed: boolean
): Promise<void> => {
  assertAccountScopeCurrent(owner);
  const invalidation = invalidateWildcardDependents(queryClient);

  if (failed) {
    // The write's own failure is what gets reported; the refresh failing too adds nothing to it.
    invalidation.catch(() => undefined);
    return;
  }

  await invalidation;
};

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

      try {
        await run();
      } catch (error) {
        await invalidateAfterAttempt(queryClient, owner, true);
        throw error;
      }

      await invalidateAfterAttempt(queryClient, owner, false);
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

  const removeMany = useCallback(
    async (ids: readonly string[]): Promise<string[]> => {
      const owner = captureAccountScope();
      const deleted: string[] = [];
      let attempted = false;
      let failure: unknown;

      try {
        for (let start = 0; start < ids.length; start += WILDCARD_DELETE_BATCH_SIZE) {
          assertAccountScopeCurrent(owner);
          attempted = true;
          const batch = await deleteWildcards(ids.slice(start, start + WILDCARD_DELETE_BATCH_SIZE), owner.signal);

          assertAccountScopeCurrent(owner);
          deleted.push(...batch);
        }
      } catch (caught) {
        failure = caught;
      }

      // Any attempt can change the catalog: a batch whose answer was lost may still have committed, and ids an answer
      // leaves out may have gone in another tab.
      if (attempted) {
        await invalidateAfterAttempt(queryClient, owner, Boolean(failure));
      }

      if (failure) {
        throw new WildcardDeleteError(deleted, failure);
      }

      return deleted;
    },
    [queryClient]
  );

  const applyWrites = useCallback(
    async (writes: readonly WildcardWrite[], owner: AccountScope): Promise<number> => {
      let done = 0;
      let attempted = false;
      let failure: unknown;

      try {
        for (const write of writes) {
          assertAccountScopeCurrent(owner);
          attempted = true;

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

      if (attempted) {
        await invalidateAfterAttempt(queryClient, owner, Boolean(failure));
      }

      if (failure) {
        throw new WildcardWriteError(done, failure);
      }

      return done;
    },
    [queryClient]
  );

  return {
    applyWrites,
    create,
    isError: query.isError,
    isLoading: query.isPending,
    isRetrying: query.isError && query.isFetching,
    knownNames,
    remove,
    removeMany,
    retry: () => void query.refetch(),
    update,
    wildcards,
  };
};
