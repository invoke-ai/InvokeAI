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

    // The initial failure produces a fetch at the leading and trailing edges of the throttle
    // window, and the first backoff retry (1s) restores the ranges for at least one more pass.
    // Without the retry, clearing pendingRanges after the failed fetch still re-runs the effect
    // once, so the count caps at two — three or more requires the retry restoring the ranges.
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

  it('stops retrying when failure is sustained, instead of storming', async () => {
    // Review finding on the original retry: restoring the ranges immediately meant a sustained
    // backend outage produced a request every throttle window, forever — a fixed-rate storm from
    // every open tab against a backend trying to come back up. The bounded retry backs off
    // (1s, 2s, 4s, 8s, 16s) and gives up after five consecutive scheduled retries, so the request
    // stream must terminate. Each retry pass produces at most a leading and a trailing fetch,
    // bounding the total at 12; six requires every backoff retry to have actually fired.
    mocks.failFetches = true;
    renderHook(ITEM_IDS, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(35_000);

    expect(mocks.queueFetches.length).toBeGreaterThanOrEqual(6);
    expect(mocks.queueFetches.length).toBeLessThanOrEqual(12);

    const settledFetches = mocks.queueFetches.length;
    const settledRenders = renderCount;
    await advance(30_000);
    expect(mocks.queueFetches.length).toBe(settledFetches);
    expect(renderCount).toBe(settledRenders);
  });

  it('resumes retrying after giving up when the user scrolls', async () => {
    mocks.failFetches = true;
    renderHook(ITEM_IDS, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(35_000);
    const fetchesAfterGiveUp = mocks.queueFetches.length;

    // A new range report is fresh user input: it restarts the retry budget, so the list does not
    // stay dead until reload. With the budget still exhausted, only the scroll-triggered fetch and
    // its trailing companion would fire — three or more new fetches requires the backoff schedule
    // to have restarted.
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(2_000);
    expect(mocks.queueFetches.length).toBeGreaterThanOrEqual(fetchesAfterGiveUp + 3);
  });

  it('recovers a range that failed while the user was scrolling elsewhere', async () => {
    // Review finding on the original retry: the catch (`prev.length > 0 ? prev : ranges`) dropped
    // the failed range whenever another range had been reported in the meantime — rows the user
    // had scrolled past stayed blank placeholders. The retry now merges the failed ranges with
    // whatever is pending instead of choosing one side, so both ranges end up fetched with no
    // further user input.
    const itemIds = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    mocks.failFetches = true;
    renderHook(itemIds, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    // The fetches for the failed range land at t=500 (throttle edges), scheduling the 1s backoff
    // retry for t=1500.
    await advance(1_250);

    // The backend recovers, and the user scrolls to a disjoint range. The first report fires on
    // the throttle's leading edge (t=1250); the second lands in pendingRanges and stays there
    // until the trailing edge (t=1750) — so the backoff retry at t=1500 finds a non-empty
    // pendingRanges and must merge into it rather than pick a side.
    mocks.failFetches = false;
    scrollTo({ startIndex: 6, endIndex: 8 });
    scrollTo({ startIndex: 6, endIndex: 8 });
    await advance(3_000);

    // Both the failed range (1-3) and the new one (7-9) land, with no user input beyond the one
    // scroll — and nothing outside the reported ranges is fetched.
    expect([...mocks.cachedItemIds].sort((a, b) => a - b)).toEqual([1, 2, 3, 7, 8, 9]);
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
