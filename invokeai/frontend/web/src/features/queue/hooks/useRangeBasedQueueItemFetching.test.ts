import { describe, expect, it } from 'vitest';

import { getItemIdBatches, getUncachedItemIds } from './useRangeBasedQueueItemFetching';

describe('queue item summary batching', () => {
  it('sends nothing when there is nothing to fetch', () => {
    expect(getItemIdBatches([])).toEqual([]);
  });

  it('keeps a request that is exactly at the backend limit in one batch', () => {
    const batches = getItemIdBatches(Array.from({ length: 1000 }, (_, i) => i));
    expect(batches).toHaveLength(1);
    expect(batches[0]).toHaveLength(1000);
  });

  it('splits past the backend limit, which rejects larger batches with a 422', () => {
    const itemIds = Array.from({ length: 2001 }, (_, i) => i);

    const batches = getItemIdBatches(itemIds);

    expect(batches.map((batch) => batch.length)).toEqual([1000, 1000, 1]);
    // Every id is sent exactly once, in order — a dropped id means a row stuck on its placeholder.
    expect(batches.flat()).toEqual(itemIds);
  });

  it('does not re-request ids while their bulk request is still in flight', () => {
    const ranges = [{ startIndex: 0, endIndex: 2 }];

    expect(getUncachedItemIds([11, 12, 13], [], ranges, new Set([12]))).toEqual([11, 13]);
  });
});
