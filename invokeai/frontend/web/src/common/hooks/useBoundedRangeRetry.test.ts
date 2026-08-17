import { describe, expect, it } from 'vitest';

import { coalesceRanges } from './useBoundedRangeRetry';

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
