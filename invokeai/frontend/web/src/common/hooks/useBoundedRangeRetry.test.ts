// @vitest-environment happy-dom
import { act, createElement, type FC } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { ListRange } from 'react-virtuoso';
import { $isConnected } from 'services/events/stores';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { coalesceRanges, useBoundedRangeRetry } from './useBoundedRangeRetry';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('coalesceRanges', () => {
  it('returns empty and single-range inputs as-is', () => {
    expect(coalesceRanges([])).toEqual([]);
    expect(coalesceRanges([{ startIndex: 3, endIndex: 7 }])).toEqual([{ startIndex: 3, endIndex: 7 }]);
  });

  it('merges overlapping ranges', () => {
    expect(
      coalesceRanges([
        { startIndex: 0, endIndex: 5 },
        { startIndex: 3, endIndex: 8 },
      ])
    ).toEqual([{ startIndex: 0, endIndex: 8 }]);
  });

  it('merges adjacent ranges', () => {
    expect(
      coalesceRanges([
        { startIndex: 0, endIndex: 2 },
        { startIndex: 3, endIndex: 5 },
      ])
    ).toEqual([{ startIndex: 0, endIndex: 5 }]);
  });

  it('collapses duplicates — the per-retry-cycle growth case', () => {
    // Pre-change, each retry cycle appended the viewport range again, so the pending state grew
    // by a duplicate entry per cycle for as long as the failure persisted.
    const range = { startIndex: 10, endIndex: 30 };
    expect(coalesceRanges([range, range, range, range])).toEqual([range]);
  });

  it('absorbs contained ranges', () => {
    expect(
      coalesceRanges([
        { startIndex: 0, endIndex: 10 },
        { startIndex: 2, endIndex: 4 },
      ])
    ).toEqual([{ startIndex: 0, endIndex: 10 }]);
  });

  it('keeps disjoint ranges separate and sorts them', () => {
    expect(
      coalesceRanges([
        { startIndex: 6, endIndex: 8 },
        { startIndex: 0, endIndex: 2 },
      ])
    ).toEqual([
      { startIndex: 0, endIndex: 2 },
      { startIndex: 6, endIndex: 8 },
    ]);
  });

  it('does not mutate its input', () => {
    const input = [
      { startIndex: 0, endIndex: 5 },
      { startIndex: 3, endIndex: 8 },
    ];
    coalesceRanges(input);
    expect(input).toEqual([
      { startIndex: 0, endIndex: 5 },
      { startIndex: 3, endIndex: 8 },
    ]);
  });
});

// Review finding (coverage): coalescing was pinned only by the unit tests above — replacing either
// accumulation site inside the hook with a plain concat left every suite green, so the bounded-state
// property rested on the helper being *correct*, not on it being *used*. These tests observe what
// the hook actually hands to `restoreRanges`, which is where an uncoalesced set would leak out.
describe('useBoundedRangeRetry', () => {
  let root: Root | null = null;
  let hookReturn: ReturnType<typeof useBoundedRangeRetry>;
  let restoredRanges: ListRange[][] = [];

  const Harness: FC = () => {
    hookReturn = useBoundedRangeRetry((failedRanges) => {
      restoredRanges.push(failedRanges);
    });
    return null;
  };

  beforeEach(() => {
    vi.useFakeTimers();
    restoredRanges = [];
    $isConnected.set(false);
    root = createRoot(document.createElement('div'));
    act(() => {
      root!.render(createElement(Harness));
    });
  });

  afterEach(() => {
    if (root) {
      act(() => {
        root!.unmount();
      });
      root = null;
    }
    $isConnected.set(false);
    vi.useRealTimers();
  });

  it('hands the retry one coalesced union of the failures merged while it was scheduled', async () => {
    // The first failure schedules the backoff retry; failures reported while it is pending only
    // merge their ranges into it. The merge must go through coalesceRanges — accumulated raw, the
    // set handed back grows by one entry per failure for as long as the failure persists, which is
    // exactly the unbounded-state pathology the accumulation exists to prevent.
    hookReturn.onFetchFailure([{ startIndex: 0, endIndex: 2 }]);
    hookReturn.onFetchFailure([{ startIndex: 0, endIndex: 2 }]);
    hookReturn.onFetchFailure([{ startIndex: 1, endIndex: 4 }]);
    hookReturn.onFetchFailure([{ startIndex: 3, endIndex: 6 }]);

    await vi.advanceTimersByTimeAsync(1_000);

    expect(restoredRanges).toEqual([[{ startIndex: 0, endIndex: 6 }]]);
  });

  it('parks repeated post-exhaustion failures as one coalesced union', async () => {
    // Exhaust the budget: five failure/retry cycles over the same viewport.
    for (let i = 0; i < 5; i++) {
      hookReturn.onFetchFailure([{ startIndex: 0, endIndex: 2 }]);
      await vi.advanceTimersByTimeAsync(16_000);
    }
    expect(restoredRanges).toHaveLength(5);
    restoredRanges = [];

    // Every further failure lands on the parked set. Merged raw instead of coalesced, the parked
    // set grows by an entry per failure — unbounded for as long as the outage lasts — and the
    // eventual heal restores (and re-fetches) that whole pile instead of the union.
    hookReturn.onFetchFailure([{ startIndex: 0, endIndex: 2 }]);
    hookReturn.onFetchFailure([{ startIndex: 0, endIndex: 2 }]);
    hookReturn.onFetchFailure([{ startIndex: 1, endIndex: 4 }]);
    expect(vi.getTimerCount()).toBe(0);

    hookReturn.resetRetryBudget();
    expect(restoredRanges).toEqual([[{ startIndex: 0, endIndex: 4 }]]);
  });
});
