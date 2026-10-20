import { useCallback, useEffect, useLayoutEffect, useRef } from 'react';
import type { ListRange } from 'react-virtuoso';
import { $isConnected } from 'services/events/stores';

const RETRY_INITIAL_DELAY_MS = 1_000;
const RETRY_MAX_DELAY_MS = 16_000;
const RETRY_MAX_ATTEMPTS = 5;
/**
 * Floor on how often a socket reconnect may re-arm an exhausted budget. A reconnect is evidence
 * the backend is answering, but the socket and the REST API can disagree: a proxy can route the
 * websocket to a healthy replica while REST hits a sick one, and a crash-looping container
 * completes a handshake on every restart. Without this floor the budget would be per-reconnect
 * rather than per-outage, and a flapping socket would turn the bounded retry back into a stream.
 */
const RETRY_REARM_COOLDOWN_MS = 60_000;

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
   * hook stops scheduling and parks the ranges until the budget is reset.
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
 * Giving up is not the same as dying. Ranges abandoned when the budget runs out are parked (as a
 * coalesced union, so parking them is bounded too) and restored on the next event that says the
 * backend is answering again: a successful fetch, a fresh range report from the user, or a socket
 * reconnect. The reconnect signal is what covers the case the retry budget cannot — a restart that
 * takes longer than the ~31s schedule, where an idle user is watching a gallery whose `imageNames`
 * never change and so has no other reason to re-run the fetch effect. Success and user input are
 * self-limiting signals; a reconnect is not, so it re-arms at most once per
 * RETRY_REARM_COOLDOWN_MS and only when there is something parked to heal.
 *
 * `restoreRanges` is invoked with the coalesced union of every range that failed since the last
 * retry. It is read through a ref, so an unstable callback cannot churn `onFetchFailure`; the ref
 * is updated in a layout effect, so a restore firing between render and commit still sees the
 * previous render's closure.
 */
export const useBoundedRangeRetry = (
  restoreRanges: (failedRanges: ListRange[]) => void
): UseBoundedRangeRetryReturn => {
  const stateRef = useRef<{
    attempts: number;
    failedRanges: ListRange[];
    abandonedRanges: ListRange[];
    timeoutId: ReturnType<typeof setTimeout> | null;
    isMounted: boolean;
    lastRearmAt: number;
  }>({
    attempts: 0,
    failedRanges: [],
    abandonedRanges: [],
    timeoutId: null,
    isMounted: true,
    lastRearmAt: 0,
  });

  // Read `restoreRanges` through a ref so an unstable callback cannot churn `onFetchFailure` (and
  // through it the caller's fetch callback, its throttle, and the effect that drives it) on every
  // render. The hook's contract shouldn't depend on the caller remembering to useCallback.
  const restoreRangesRef = useRef(restoreRanges);
  useLayoutEffect(() => {
    restoreRangesRef.current = restoreRanges;
  }, [restoreRanges]);

  useEffect(() => {
    const state = stateRef.current;
    state.isMounted = true;
    return () => {
      // A bulk fetch may still be in flight and reject after unmount; without this flag its
      // `onFetchFailure` would schedule a fresh backoff timer that no cleanup will ever reach.
      state.isMounted = false;
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

  const takeAbandonedRanges = useCallback((): ListRange[] | null => {
    const state = stateRef.current;
    if (!state.isMounted || state.abandonedRanges.length === 0) {
      return null;
    }
    const abandonedRanges = state.abandonedRanges;
    state.abandonedRanges = [];
    return abandonedRanges;
  }, []);

  const resetRetryBudget = useCallback(() => {
    const state = stateRef.current;
    if (!state.isMounted) {
      return;
    }
    state.attempts = 0;
    const abandonedRanges = takeAbandonedRanges();
    if (abandonedRanges) {
      restoreRangesRef.current(abandonedRanges);
    }
  }, [takeAbandonedRanges]);

  useEffect(() => {
    // A reconnect means the backend is answering again. Nothing else re-arms an exhausted budget
    // for an idle user: in production `socketConnected` only invalidates `FetchOnReconnect` when
    // the queue status changed, and even then RTK Query's structural sharing hands the gallery
    // back the same `imageNames` reference, so no dependency of the fetch effect changes.
    // `listen` fires on transitions only, so this runs on reconnect, not on the initial connect.
    return $isConnected.listen((isConnected) => {
      if (!isConnected) {
        return;
      }
      const state = stateRef.current;
      // Unlike a success or a scroll, reconnects are not self-limiting — see the cooldown's note.
      // Both guards matter: re-arming with nothing parked would zero `attempts` mid-streak, so a
      // socket flapping faster than the backoff would pin the delay at 1s indefinitely.
      const now = Date.now();
      if (now - state.lastRearmAt < RETRY_REARM_COOLDOWN_MS) {
        return;
      }
      const abandonedRanges = takeAbandonedRanges();
      if (!abandonedRanges) {
        return;
      }
      state.lastRearmAt = now;
      state.attempts = 0;
      restoreRangesRef.current(abandonedRanges);
    });
  }, [takeAbandonedRanges]);

  const onFetchFailure = useCallback((ranges: ListRange[]) => {
    const state = stateRef.current;
    if (!state.isMounted) {
      return;
    }
    state.failedRanges = coalesceRanges([...state.failedRanges, ...ranges]);
    if (state.timeoutId !== null) {
      // A retry is already scheduled; it will pick up the merged ranges when it fires.
      return;
    }
    if (state.attempts >= RETRY_MAX_ATTEMPTS) {
      // Budget exhausted — stop scheduling, but park the ranges (coalesced, so parking is bounded)
      // rather than dropping them, so a reconnect, a later success, or a scroll can heal the rows.
      state.abandonedRanges = coalesceRanges([...state.abandonedRanges, ...state.failedRanges]);
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
        restoreRangesRef.current(failedRanges);
      }
    }, delay);
  }, []);

  return { onFetchFailure, resetRetryBudget };
};
