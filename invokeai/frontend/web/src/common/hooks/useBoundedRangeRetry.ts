import { useCallback, useEffect, useRef } from 'react';
import type { ListRange } from 'react-virtuoso';

const RETRY_INITIAL_DELAY_MS = 1_000;
const RETRY_MAX_DELAY_MS = 16_000;
const RETRY_MAX_ATTEMPTS = 5;

/**
 * Merge overlapping or adjacent ranges into a minimal, sorted, disjoint set.
 *
 * This is what bounds the retry state: failed ranges are accumulated as a coalesced union, so
 * repeated failures over the same viewport collapse into one entry instead of growing by a
 * duplicate range per retry cycle.
 */
export const coalesceRanges = (ranges: ListRange[]): ListRange[] => {
  if (ranges.length <= 1) {
    return ranges;
  }
  const sorted = [...ranges].sort((a, b) => a.startIndex - b.startIndex);
  const first = sorted[0]!;
  const coalesced: ListRange[] = [{ startIndex: first.startIndex, endIndex: first.endIndex }];
  for (let i = 1; i < sorted.length; i++) {
    const range = sorted[i]!;
    const last = coalesced[coalesced.length - 1]!;
    if (range.startIndex <= last.endIndex + 1) {
      last.endIndex = Math.max(last.endIndex, range.endIndex);
    } else {
      coalesced.push({ startIndex: range.startIndex, endIndex: range.endIndex });
    }
  }
  return coalesced;
};

interface UseBoundedRangeRetryReturn {
  /**
   * Report a failed bulk fetch, with the ranges it was fetching. Schedules a single retry with
   * exponential backoff (1s, 2s, ... capped at 16s); while one is already scheduled, additional
   * failures only merge their ranges into it. After RETRY_MAX_ATTEMPTS consecutive failures the
   * hook gives up until the budget is reset.
   */
  onFetchFailure: (ranges: ListRange[]) => void;
  /**
   * End the current failure streak. Call when a fetch succeeds (the backend is answering again)
   * and on new user input (a fresh range report), so a list that gave up resumes retrying as the
   * user scrolls.
   */
  resetRetryBudget: () => void;
}

/**
 * Bounded, backoff-driven retry of failed range fetches.
 *
 * The range-based fetching hooks are the ONLY fetcher for their rows (the row components consume
 * the cache with `skip: isUninitialized`), so a failed bulk fetch must be retried or the rows stay
 * placeholders until the user happens to scroll. But an unbounded retry is a fixed-rate request
 * storm from every open tab against a backend that is trying to come back up. This hook bounds it:
 * exponential backoff between attempts, a cap on consecutive failures, and coalesced accumulation
 * of the failed ranges.
 *
 * `restoreRanges` is invoked when a retry fires, with the coalesced union of every range that
 * failed since the last retry. It must be referentially stable (wrap it in `useCallback`).
 */
export const useBoundedRangeRetry = (
  restoreRanges: (failedRanges: ListRange[]) => void
): UseBoundedRangeRetryReturn => {
  const stateRef = useRef<{
    attempts: number;
    failedRanges: ListRange[];
    timeoutId: ReturnType<typeof setTimeout> | null;
  }>({ attempts: 0, failedRanges: [], timeoutId: null });

  useEffect(() => {
    const state = stateRef.current;
    return () => {
      if (state.timeoutId !== null) {
        clearTimeout(state.timeoutId);
        // Null the sentinel too: effect cleanup can run while the instance (and this ref)
        // survives — Fast Refresh, or a re-suspending Suspense/Activity boundary. A stale
        // non-null timeoutId would make every future onFetchFailure early-return, silently
        // disabling retry for the lifetime of the instance.
        state.timeoutId = null;
      }
    };
  }, []);

  const onFetchFailure = useCallback(
    (ranges: ListRange[]) => {
      const state = stateRef.current;
      state.failedRanges = coalesceRanges([...state.failedRanges, ...ranges]);
      if (state.timeoutId !== null) {
        // A retry is already scheduled; it will pick up the merged ranges when it fires.
        return;
      }
      if (state.attempts >= RETRY_MAX_ATTEMPTS) {
        // Budget exhausted — abandon these ranges rather than letting them accumulate. The rows
        // still in view are re-reported by the next range change, which also resets the budget.
        state.failedRanges = [];
        return;
      }
      state.attempts += 1;
      const delay = Math.min(RETRY_INITIAL_DELAY_MS * 2 ** (state.attempts - 1), RETRY_MAX_DELAY_MS);
      state.timeoutId = setTimeout(() => {
        state.timeoutId = null;
        const failedRanges = state.failedRanges;
        state.failedRanges = [];
        if (failedRanges.length > 0) {
          restoreRanges(failedRanges);
        }
      }, delay);
    },
    [restoreRanges]
  );

  const resetRetryBudget = useCallback(() => {
    stateRef.current.attempts = 0;
  }, []);

  return { onFetchFailure, resetRetryBudget };
};
