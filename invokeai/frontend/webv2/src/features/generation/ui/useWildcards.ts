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

/**
 * A run of writes that stopped part-way, carrying how many had landed.
 *
 * The count is the point: the ones already written are real, and a caller that
 * reports only "it failed" sends the user back for a second run that then
 * clashes with everything the first one made.
 */
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
  /**
   * A run of writes with a single invalidation at the end.
   *
   * Resolves with how many landed, and rethrows whatever stopped it — the caller
   * needs both to say "stopped after 6 of 40". Ordinary edits go through the
   * single-write methods above; this exists for import, where doing it one call
   * at a time meant re-fetching the whole catalog, plus every dynamic-prompt
   * expansion, between each write and the next.
   */
  applyWrites: (writes: readonly WildcardWrite[], owner: AccountScope) => Promise<number>;
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

      // Whatever landed before the failure is real, so the caches are stale
      // either way — invalidate on the strength of that rather than on the run
      // having finished. The scope check stays where it is for every other
      // mutation: a sign-out mid-run must not drop the next account's caches.
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
