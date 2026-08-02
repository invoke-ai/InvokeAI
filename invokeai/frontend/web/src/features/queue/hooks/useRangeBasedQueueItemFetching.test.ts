// @vitest-environment happy-dom
import { act, createElement, type FC } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { ListRange } from 'react-virtuoso';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useRangeBasedQueueItemFetching } from './useRangeBasedQueueItemFetching';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mocks = vi.hoisted(() => ({
  // Args of every getQueueItemDTOsByItemIds call, in order.
  queueFetches: [] as number[][],
  // Item ids with a getQueueItem cache entry, as reported by selectCachedArgsForQuery.
  cachedItemIds: [] as number[],
  // When true, a successful fetch upserts the requested ids into the cache, like the mutation's
  // onQueryStarted does. When false, requested ids never land in the cache.
  cacheLands: true,
  // When true, the mutation rejects, like a backend restart or a 502 from a reverse proxy.
  failFetches: false,
}));

vi.mock('app/store/storeHooks', () => {
  const store = { getState: () => ({}), dispatch: () => undefined };
  return { useAppStore: () => store };
});

vi.mock('services/api/endpoints/queue', () => {
  const trigger = (arg: { item_ids: number[] }) => {
    mocks.queueFetches.push(arg.item_ids);
    // Like the real mutation: onQueryStarted upserts when the request fulfills, whether or not
    // the caller unwraps, and only the promise returned by unwrap() surfaces the rejection.
    const settled = mocks.failFetches
      ? Promise.reject(new Error('fetch failed'))
      : Promise.resolve().then(() => {
          if (mocks.cacheLands) {
            mocks.cachedItemIds.push(...arg.item_ids);
          }
          return [];
        });
    settled.catch(() => undefined);
    return { unwrap: () => settled.then((r) => r) };
  };
  // RTK Query's mutation trigger is referentially stable across renders; the hook's
  // fetchQueueItems callback (and therefore its throttle and effect) depend on that.
  const result = [trigger];
  return {
    queueApi: { util: { selectCachedArgsForQuery: () => mocks.cachedItemIds } },
    useGetQueueItemDTOsByItemIdsMutation: () => result,
  };
});

const ITEM_IDS = [1, 2, 3];
const THROTTLE_MS = 500;

describe('useRangeBasedQueueItemFetching', () => {
  let root: Root | null = null;
  let renderCount = 0;
  let hookReturn: ReturnType<typeof useRangeBasedQueueItemFetching>;

  const renderHook = (itemIds: number[], enabled: boolean) => {
    const Harness: FC = () => {
      renderCount++;
      hookReturn = useRangeBasedQueueItemFetching({ itemIds, enabled });
      return null;
    };
    root = createRoot(document.createElement('div'));
    act(() => {
      root!.render(createElement(Harness));
    });
  };

  const scrollTo = (range: ListRange) => {
    act(() => {
      hookReturn.onRangeChanged(range);
    });
  };

  // Advance fake time in small steps, flushing React work (renders + effects) between steps. A
  // single long advance would defer all effect re-runs to the end of the act scope, which breaks
  // the feedback cycle this suite exists to detect: state update -> effect -> throttle -> fetch ->
  // state update. Stepping mimics real event-loop turns, letting a loop sustain itself if the
  // code allows one.
  const advance = async (ms: number) => {
    const step = 250;
    for (let elapsed = 0; elapsed < ms; elapsed += step) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(step);
      });
    }
  };

  beforeEach(() => {
    vi.useFakeTimers();
    mocks.queueFetches = [];
    mocks.cachedItemIds = [];
    mocks.cacheLands = true;
    mocks.failFetches = false;
    renderCount = 0;
  });

  afterEach(() => {
    if (root) {
      act(() => {
        root!.unmount();
      });
      root = null;
    }
    vi.useRealTimers();
  });

  it('fetches uncached items for a reported range, then goes quiet', async () => {
    renderHook(ITEM_IDS, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 2);

    expect(mocks.queueFetches).toEqual([ITEM_IDS]);

    // Regression: clearing pendingRanges with a fresh `[]` (a new identity every time) re-ran the
    // effect, re-armed the throttle, and cleared again — a self-sustaining render loop. Once the
    // range has been handled and the throttle has drained, both renders and fetches must stop.
    const settledRenders = renderCount;
    await advance(THROTTLE_MS * 10);
    expect(renderCount).toBe(settledRenders);
    expect(mocks.queueFetches).toEqual([ITEM_IDS]);
  });

  it('stops re-requesting items that never land in the cache', async () => {
    // A requested id the server does not return never gets a getQueueItem cache entry, so it is
    // uncached on every pass. Pre-fix, that sustained the loop: the list re-requested such ids
    // every ~500ms for as long as it was mounted.
    mocks.cacheLands = false;
    renderHook(ITEM_IDS, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 10);

    // The range-change pass fetches once, and clearing pendingRanges ([range] -> EMPTY_ARRAY) is a
    // real state change, so one follow-up pass may re-check the cache and re-request the
    // still-missing ids. After that the state is stable and the stream must stop — pre-fix it
    // continued at one request per throttle window, forever.
    expect(mocks.queueFetches.length).toBeGreaterThanOrEqual(1);
    expect(mocks.queueFetches.length).toBeLessThanOrEqual(2);
    const settledFetches = mocks.queueFetches.length;
    await advance(THROTTLE_MS * 10);
    expect(mocks.queueFetches.length).toBe(settledFetches);
  });

  it('retries a failed fetch until it succeeds, then goes quiet', async () => {
    // This bulk fetch is the only fetcher for these rows (QueueItemAtPosition subscribes with
    // `skip: isUninitialized`), so a transient failure must be retried or the placeholders stay
    // empty until the user happens to scroll.
    mocks.failFetches = true;
    renderHook(ITEM_IDS, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 4);

    // The catch-driven retry produces a fetch per throttle window. Without it, clearing
    // pendingRanges after the failed fetch still re-runs the effect once, so the count caps at
    // two — three or more requires the catch handler restoring the ranges.
    expect(mocks.queueFetches.length).toBeGreaterThanOrEqual(3);
    expect(mocks.cachedItemIds).toEqual([]);

    mocks.failFetches = false;
    await advance(THROTTLE_MS * 4);
    expect(mocks.cachedItemIds).toEqual(ITEM_IDS);

    const fetchesAfterRecovery = mocks.queueFetches.length;
    const settledRenders = renderCount;
    await advance(THROTTLE_MS * 10);
    expect(mocks.queueFetches.length).toBe(fetchesAfterRecovery);
    expect(renderCount).toBe(settledRenders);
  });

  it('still fetches for new ranges after settling', async () => {
    const itemIds = [1, 2, 3, 4, 5, 6];
    renderHook(itemIds, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 10);
    expect(mocks.queueFetches).toEqual([[1, 2, 3]]);

    scrollTo({ startIndex: 3, endIndex: 5 });
    await advance(THROTTLE_MS * 2);
    expect(mocks.queueFetches).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });

  it('fetches every range reported within a throttle window, not just the last', async () => {
    // onRangeChanged accumulates ranges into pendingRanges precisely so that ranges reported
    // mid-window are not dropped when the trailing invocation only sees the latest call's args.
    const itemIds = [1, 2, 3, 4, 5, 6];
    renderHook(itemIds, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    scrollTo({ startIndex: 3, endIndex: 5 });
    await advance(THROTTLE_MS * 2);
    expect(mocks.queueFetches).toEqual([[1, 2, 3, 4, 5, 6]]);
  });

  it('drops handled ranges instead of accumulating them', async () => {
    // A handled range must not be re-scanned by later passes. Pre-fix, this hook returned early
    // without clearing when everything was cached, so ranges accumulated for the lifetime of the
    // list and a later pass would re-request an item evicted from a range handled long ago.
    const itemIds = [1, 2, 3, 4, 5, 6];
    mocks.cachedItemIds = [1, 2, 3];
    renderHook(itemIds, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 10);
    expect(mocks.queueFetches).toEqual([]);

    mocks.cachedItemIds = [1, 3];
    scrollTo({ startIndex: 3, endIndex: 5 });
    await advance(THROTTLE_MS * 2);
    expect(mocks.queueFetches).toEqual([[4, 5, 6]]);
  });

  it('does not fetch when disabled', async () => {
    renderHook(ITEM_IDS, false);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 4);
    expect(mocks.queueFetches).toEqual([]);
  });
});
