// @vitest-environment happy-dom
import { act, createElement, type FC } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { ListRange } from 'react-virtuoso';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getVideoPrefetchOptions, hasCachedVideoDTO, useRangeBasedImageFetching } from './useRangeBasedImageFetching';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mocks = vi.hoisted(() => ({
  // Args of every getImageDTOsByNames call, in order.
  imageFetches: [] as string[][],
  // Names with a getImageDTO cache entry, as reported by selectCachedArgsForQuery.
  cachedImageNames: [] as string[],
  // When true, a successful fetch upserts the requested names into the cache, like
  // getImageDTOsByNames.onQueryStarted does. When false, requested names never land in the
  // cache — the deleted-image / multiuser-filtered case that drove the pre-fix request stream.
  cacheLands: true,
  // When true, the mutation rejects, like a backend restart or a 502 from a reverse proxy.
  failFetches: false,
}));

vi.mock('app/store/storeHooks', () => {
  const store = { getState: () => ({}), dispatch: () => undefined };
  return { useAppStore: () => store };
});

vi.mock('features/gallery/store/types', () => ({
  isVideoName: (name: string) => name.endsWith('.mp4'),
}));

vi.mock('services/api/endpoints/images', () => {
  const trigger = (arg: { image_names: string[] }) => {
    mocks.imageFetches.push(arg.image_names);
    // Like the real mutation: onQueryStarted upserts when the request fulfills, whether or not
    // the caller unwraps, and only the promise returned by unwrap() surfaces the rejection.
    const settled = mocks.failFetches
      ? Promise.reject(new Error('fetch failed'))
      : Promise.resolve().then(() => {
          if (mocks.cacheLands) {
            mocks.cachedImageNames.push(...arg.image_names);
          }
          return [];
        });
    settled.catch(() => undefined);
    return { unwrap: () => settled.then((r) => r) };
  };
  // RTK Query's mutation trigger is referentially stable across renders; the hook's fetchItems
  // callback (and therefore its throttle and effect) depend on that.
  const result = [trigger];
  return {
    imagesApi: { util: { selectCachedArgsForQuery: () => mocks.cachedImageNames } },
    useGetImageDTOsByNamesMutation: () => result,
  };
});

vi.mock('services/api/endpoints/videos', () => ({
  videosApi: {
    util: { selectCachedArgsForQuery: () => [] },
    endpoints: { getVideoDTO: { select: () => () => ({ data: undefined }), initiate: () => ({ type: 'noop' }) } },
  },
}));

const IMAGE_NAMES = ['a.png', 'b.png', 'c.png'];
const THROTTLE_MS = 500;

describe('useRangeBasedImageFetching', () => {
  let root: Root | null = null;
  let renderCount = 0;
  let hookReturn: ReturnType<typeof useRangeBasedImageFetching>;

  const renderHook = (imageNames: string[], enabled: boolean) => {
    const Harness: FC = () => {
      renderCount++;
      hookReturn = useRangeBasedImageFetching({ imageNames, enabled });
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
    mocks.imageFetches = [];
    mocks.cachedImageNames = [];
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

  it('fetches uncached names for a reported range, then goes quiet', async () => {
    renderHook(IMAGE_NAMES, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 2);

    expect(mocks.imageFetches).toEqual([IMAGE_NAMES]);

    // Regression: clearing pendingRanges with a fresh `[]` (a new identity every time) re-ran the
    // effect, re-armed the throttle, and cleared again — a self-sustaining render loop that
    // re-rendered every ~500ms for as long as the grid was mounted, with no user input. Once the
    // range has been handled and the throttle has drained, both renders and fetches must stop.
    const settledRenders = renderCount;
    await advance(THROTTLE_MS * 10);
    expect(renderCount).toBe(settledRenders);
    expect(mocks.imageFetches).toEqual([IMAGE_NAMES]);
  });

  it('does not loop even while the grid is mounted with nothing to fetch', async () => {
    // Pre-fix, the loop ran from mount even with no ranges reported, because the clear was
    // unconditional and every pass installed a new [] identity.
    renderHook(IMAGE_NAMES, true);
    const settledRenders = renderCount;
    await advance(THROTTLE_MS * 10);
    expect(renderCount).toBe(settledRenders);
    expect(mocks.imageFetches).toEqual([]);
  });

  it('stops re-requesting names that never land in the cache', async () => {
    // onQueryStarted upserts only the DTOs the server actually returned, so a requested name that
    // comes back missing (deleted image, multiuser ownership filter) never gets a cache entry.
    // Pre-fix, the render loop re-requested such names every ~500ms, forever.
    mocks.cacheLands = false;
    renderHook(IMAGE_NAMES, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 10);

    // The range-change pass fetches once, and clearing pendingRanges ([range] -> EMPTY_ARRAY) is a
    // real state change, so one follow-up pass may re-check the cache and re-request the
    // still-missing names. After that the state is stable and the stream must stop — pre-fix it
    // continued at one request per throttle window, forever.
    expect(mocks.imageFetches.length).toBeGreaterThanOrEqual(1);
    expect(mocks.imageFetches.length).toBeLessThanOrEqual(2);
    const settledFetches = mocks.imageFetches.length;
    await advance(THROTTLE_MS * 10);
    expect(mocks.imageFetches.length).toBe(settledFetches);
  });

  it('retries a failed fetch until it succeeds, then goes quiet', async () => {
    // The pre-fix loop was also an accidental retry, and this bulk fetch is the only fetcher for
    // these rows (ImageAtPosition subscribes with `skip: isUninitialized`). Without an explicit
    // retry, a transient failure would leave grey placeholders until the user happens to scroll.
    mocks.failFetches = true;
    renderHook(IMAGE_NAMES, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 4);

    // The initial failure produces a fetch at the leading and trailing edges of the throttle
    // window, and the first backoff retry (1s) restores the ranges for at least one more pass.
    // Without the retry, clearing pendingRanges after the failed fetch still re-runs the effect
    // once, so the count caps at two — three or more requires the retry restoring the ranges.
    expect(mocks.imageFetches.length).toBeGreaterThanOrEqual(3);
    expect(mocks.cachedImageNames).toEqual([]);

    mocks.failFetches = false;
    await advance(THROTTLE_MS * 4);
    expect(mocks.cachedImageNames).toEqual(IMAGE_NAMES);

    const fetchesAfterRecovery = mocks.imageFetches.length;
    const settledRenders = renderCount;
    await advance(THROTTLE_MS * 10);
    expect(mocks.imageFetches.length).toBe(fetchesAfterRecovery);
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
    renderHook(IMAGE_NAMES, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(35_000);

    expect(mocks.imageFetches.length).toBeGreaterThanOrEqual(6);
    expect(mocks.imageFetches.length).toBeLessThanOrEqual(12);

    const settledFetches = mocks.imageFetches.length;
    const settledRenders = renderCount;
    await advance(30_000);
    expect(mocks.imageFetches.length).toBe(settledFetches);
    expect(renderCount).toBe(settledRenders);
  });

  it('resumes retrying after giving up when the user scrolls', async () => {
    mocks.failFetches = true;
    renderHook(IMAGE_NAMES, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(35_000);
    const fetchesAfterGiveUp = mocks.imageFetches.length;

    // A new range report is fresh user input: it restarts the retry budget, so the grid does not
    // stay dead until reload. With the budget still exhausted, only the scroll-triggered fetch and
    // its trailing companion would fire — three or more new fetches requires the backoff schedule
    // to have restarted.
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(2_000);
    expect(mocks.imageFetches.length).toBeGreaterThanOrEqual(fetchesAfterGiveUp + 3);
  });

  it('recovers a range that failed while the user was scrolling elsewhere', async () => {
    // Review finding on the original retry: the catch (`prev.length > 0 ? prev : ranges`) dropped
    // the failed range whenever another range had been reported in the meantime — rows the user
    // had scrolled past stayed grey placeholders. The retry now merges the failed ranges with
    // whatever is pending instead of choosing one side, so both ranges end up fetched with no
    // further user input.
    const names = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png', 'g.png', 'h.png', 'i.png'];
    mocks.failFetches = true;
    renderHook(names, true);
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

    // Both the failed range (a-c) and the new one (g-i) land, with no user input beyond the one
    // scroll — and nothing outside the reported ranges is fetched.
    expect([...mocks.cachedImageNames].sort()).toEqual(['a.png', 'b.png', 'c.png', 'g.png', 'h.png', 'i.png']);
  });

  it('still fetches for new ranges after settling', async () => {
    const names = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png'];
    renderHook(names, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 10);
    expect(mocks.imageFetches).toEqual([['a.png', 'b.png', 'c.png']]);

    scrollTo({ startIndex: 3, endIndex: 5 });
    await advance(THROTTLE_MS * 2);
    expect(mocks.imageFetches).toEqual([
      ['a.png', 'b.png', 'c.png'],
      ['d.png', 'e.png', 'f.png'],
    ]);
  });

  it('fetches every range reported within a throttle window, not just the last', async () => {
    // onRangeChanged accumulates ranges into pendingRanges precisely so that ranges reported
    // mid-window are not dropped when the trailing invocation only sees the latest call's args.
    const names = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png'];
    renderHook(names, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    scrollTo({ startIndex: 3, endIndex: 5 });
    await advance(THROTTLE_MS * 2);
    expect(mocks.imageFetches).toEqual([['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png']]);
  });

  it('drops handled ranges instead of accumulating them', async () => {
    // A handled range must not be re-scanned by later passes. Pre-fix, the queue variant of this
    // hook returned early without clearing when everything was cached, so ranges accumulated for
    // the lifetime of the list and a later pass would re-request an item evicted from a range
    // handled long ago.
    const names = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png'];
    mocks.cachedImageNames = ['a.png', 'b.png', 'c.png'];
    renderHook(names, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 10);
    expect(mocks.imageFetches).toEqual([]);

    mocks.cachedImageNames = ['a.png', 'c.png'];
    scrollTo({ startIndex: 3, endIndex: 5 });
    await advance(THROTTLE_MS * 2);
    expect(mocks.imageFetches).toEqual([['d.png', 'e.png', 'f.png']]);
  });

  it('does not fetch when disabled', async () => {
    renderHook(IMAGE_NAMES, false);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 4);
    expect(mocks.imageFetches).toEqual([]);
  });
});

describe('video range prefetch', () => {
  it('does not retain an RTK Query subscription', () => {
    expect(getVideoPrefetchOptions()).toEqual({ subscribe: false, forceRefetch: true });
  });

  it('only treats fulfilled DTO queries as cached', () => {
    expect(hasCachedVideoDTO({ data: { video_name: 'video.mp4' } })).toBe(true);
    expect(hasCachedVideoDTO({ isError: true })).toBe(false);
  });
});
