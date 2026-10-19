// @vitest-environment happy-dom
import { act, createElement, type FC } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import type { ListRange } from 'react-virtuoso';
import { $isConnected } from 'services/events/stores';
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
  // When true, the mutation returns a promise the test rejects by hand, so a rejection can be
  // delivered at a chosen moment (e.g. after unmount) rather than on the next microtask.
  manualFailure: false,
  rejectPending: [] as (() => void)[],
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
    if (mocks.manualFailure) {
      let reject!: () => void;
      const pending = new Promise<never>((_, rej) => {
        reject = () => rej(new Error('fetch failed'));
      });
      pending.catch(() => undefined);
      mocks.rejectPending.push(reject);
      return { unwrap: () => pending.then((r) => r) };
    }
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

  // One stable component type, so re-rendering with new props updates the existing instance
  // instead of remounting it — a remount would silently reset the state under test.
  const Harness: FC<{ imageNames: string[]; enabled: boolean }> = ({ imageNames, enabled }) => {
    renderCount++;
    hookReturn = useRangeBasedImageFetching({ imageNames, enabled });
    return null;
  };

  const renderHook = (imageNames: string[], enabled: boolean) => {
    root = createRoot(document.createElement('div'));
    act(() => {
      root!.render(createElement(Harness, { imageNames, enabled }));
    });
  };

  const rerenderHook = (imageNames: string[], enabled: boolean) => {
    act(() => {
      root!.render(createElement(Harness, { imageNames, enabled }));
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
    mocks.manualFailure = false;
    mocks.rejectPending = [];
    renderCount = 0;
    $isConnected.set(false);
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

  it('restores parked ranges on the next scroll when the socket never dropped', async () => {
    // Review finding (coverage): parked ranges are restored by three signals — a socket reconnect,
    // a later success, and a fresh range report — but only the reconnect was pinned. `resumes
    // retrying after giving up when the user scrolls` re-reports the *same* range, which
    // `lastRange` re-fetches whether or not the parked set was restored, so deleting the restore
    // from `resetRetryBudget` left the suite green. The distinction only shows for a range that is
    // parked but no longer on screen, after a recovery the socket never observed — a transient 502
    // from a reverse proxy, say, where the websocket stays up throughout and the reconnect path
    // never fires.
    const imageNames = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png', 'g.png', 'h.png', 'i.png'];
    $isConnected.set(true);
    mocks.failFetches = true;
    renderHook(imageNames, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(35_000);
    expect(mocks.cachedImageNames).toEqual([]);

    // REST answers again with no socket transition, so the scroll is the only signal that can heal
    // the parked range.
    mocks.failFetches = false;
    scrollTo({ startIndex: 6, endIndex: 8 });
    await advance(THROTTLE_MS * 4);

    // The rows the user scrolled past during the outage land along with the new viewport. Without
    // the restore only g-i would be fetched and a-c would stay grey placeholders permanently.
    expect([...mocks.cachedImageNames].sort()).toEqual(['a.png', 'b.png', 'c.png', 'g.png', 'h.png', 'i.png']);
  });

  it('heals a grid that gave up when the socket reconnects, with no user input', async () => {
    // Review finding: the retry budget ends ~31s after the first failure, but an InvokeAI restart
    // (config load, DB migrations, model scan) routinely takes longer. For an idle user nothing
    // else re-arms it — in production `socketConnected` only invalidates `FetchOnReconnect` when
    // the queue status changed, and RTK Query's structural sharing hands back the same
    // `imageNames` reference either way, so no dependency of the fetch effect changes and
    // `enabled` (`!isLoading`) does not toggle on a refetch. Ranges abandoned by the exhausted
    // budget are parked, not dropped, and the socket reconnect restores them.
    $isConnected.set(true);
    mocks.failFetches = true;
    renderHook(IMAGE_NAMES, true);
    scrollTo({ startIndex: 0, endIndex: 2 });

    // Backend goes down: the socket drops and the retry budget runs out while it is down.
    $isConnected.set(false);
    await advance(35_000);
    const fetchesAfterGiveUp = mocks.imageFetches.length;
    await advance(30_000);
    expect(mocks.imageFetches.length).toBe(fetchesAfterGiveUp);
    expect(mocks.cachedImageNames).toEqual([]);

    // Backend comes back, well past the retry budget. No scroll, no change to imageNames.
    mocks.failFetches = false;
    act(() => {
      $isConnected.set(true);
    });
    await advance(THROTTLE_MS * 4);

    expect(mocks.cachedImageNames).toEqual(IMAGE_NAMES);

    // And the heal must settle. Restoring the parked ranges without emptying the parked set would
    // make every later success restore them again — success -> restore -> fetch -> success — a
    // loop that the cache assertion alone cannot see.
    const fetchesAfterHeal = mocks.imageFetches.length;
    await advance(30_000);
    expect(mocks.imageFetches.length).toBe(fetchesAfterHeal);
  });

  it('empties the parked set when it heals, even if the rows never reach the cache', async () => {
    // The parked set is handed to the restore and cleared in one step. Restoring without clearing
    // it looks harmless while the rows do land in the cache — the follow-up pass finds nothing to
    // request — but a name the server never returns is uncached on every pass, so a parked set
    // that outlived its restore would be re-fetched by every later success.
    const imageNames = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png', 'g.png', 'h.png', 'i.png'];
    $isConnected.set(true);
    mocks.failFetches = true;
    renderHook(imageNames, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    $isConnected.set(false);
    await advance(35_000);

    // The backend answers again, but these rows never land in the cache (deleted, or filtered out
    // for this user).
    mocks.failFetches = false;
    mocks.cacheLands = false;
    act(() => {
      $isConnected.set(true);
    });
    await advance(THROTTLE_MS * 4);

    const fetchesAfterHeal = mocks.imageFetches.length;
    await advance(60_000);
    expect(mocks.imageFetches.length).toBe(fetchesAfterHeal);

    // Review finding (coverage): the settle assertion above cannot see a parked set that outlived
    // its restore. With no further signal, a second restore would re-install the same array
    // reference into pendingRanges while it holds EMPTY_ARRAY — React bails out, the effect never
    // re-runs, and the fetch count stays flat whether or not the set was cleared. A scroll to a
    // disjoint range can see it: its resetRetryBudget restores whatever is still parked, so the
    // stale range would ride along on every later scroll, re-requesting rows the user left behind
    // for the lifetime of the grid — a slower version of the request stream this PR removes.
    scrollTo({ startIndex: 6, endIndex: 8 });
    await advance(THROTTLE_MS * 4);
    const namesRequestedAfterScroll = new Set(mocks.imageFetches.slice(fetchesAfterHeal).flat());
    expect([...namesRequestedAfterScroll].sort()).toEqual(['g.png', 'h.png', 'i.png']);
  });

  it('does not turn a flapping socket into a request stream', async () => {
    // Review finding: re-arming on every reconnect made the budget per-reconnect rather than
    // per-outage. A socket that keeps completing a handshake while REST stays broken — a
    // crash-looping container, uvicorn accepting connections before startup finishes, a proxy
    // routing the websocket to a healthy replica and REST to a sick one — would then pin the
    // backoff at its shortest delay for as long as the flapping lasted. The re-arm is now floored
    // at one per RETRY_REARM_COOLDOWN_MS (60s) and only fires when there is something parked.
    $isConnected.set(true);
    mocks.failFetches = true;
    renderHook(IMAGE_NAMES, true);
    scrollTo({ startIndex: 0, endIndex: 2 });

    // Five minutes of flapping every 5s, REST failing throughout, no user input.
    for (let i = 0; i < 60; i++) {
      act(() => {
        $isConnected.set(false);
      });
      await advance(2_500);
      act(() => {
        $isConnected.set(true);
      });
      await advance(2_500);
    }

    // Design intent with no flapping at all is 12 requests (one bounded streak). Five minutes of
    // flapping buys at most five re-arms, each worth another bounded streak. Pre-fix this ran at
    // the flap rate and measured 240.
    expect(mocks.imageFetches.length).toBeLessThanOrEqual(80);
  });

  it('does not schedule a retry for a fetch that rejects after unmount', async () => {
    // Review finding: the unmount cleanup clears the pending timer, but a mutation still in flight
    // rejects afterwards, reaching onFetchFailure on a dead instance and arming a fresh timer of
    // up to 16s that no cleanup will ever reach. Triggered by closing the gallery panel or
    // switching tabs while the backend is down.
    mocks.manualFailure = true;
    renderHook(IMAGE_NAMES, true);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 2);
    expect(mocks.rejectPending.length).toBeGreaterThan(0);

    // Unmount with the request still in flight, then let it reject.
    act(() => {
      root!.unmount();
    });
    root = null;
    const timersAfterUnmount = vi.getTimerCount();

    for (const reject of mocks.rejectPending) {
      reject();
    }
    // Deliver the rejection without advancing the clock, so a backoff timer armed by it (>=1s)
    // is still pending and countable rather than already fired and cleared.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(vi.getTimerCount()).toBe(timersAfterUnmount);
  });

  it('does not accumulate ranges reported while disabled', async () => {
    // Review finding: the `!enabled` guard returned before the clear, so every range reported
    // while disabled stayed in pendingRanges and the first enabled pass scanned all of them.
    const imageNames = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png', 'g.png', 'h.png', 'i.png'];
    renderHook(imageNames, false);
    scrollTo({ startIndex: 0, endIndex: 2 });
    await advance(THROTTLE_MS * 2);
    scrollTo({ startIndex: 3, endIndex: 5 });
    await advance(THROTTLE_MS * 2);
    expect(mocks.imageFetches).toEqual([]);

    // Enable. In production `enabled` is `!isLoading`, so it flips as the names arrive — a new
    // array identity, which is what re-runs the fetch effect (`throttledFetchItems` is
    // referentially stable across callback changes, so `enabled` alone does not re-run it).
    // The pass that follows must cover the last reported viewport (d-f) and nothing else: the
    // earlier range (a-c), long scrolled past, must not still be sitting in pendingRanges.
    rerenderHook([...imageNames], true);
    await advance(THROTTLE_MS * 2);
    expect(mocks.imageFetches.flat().sort()).toEqual(['d.png', 'e.png', 'f.png']);

    scrollTo({ startIndex: 6, endIndex: 8 });
    await advance(THROTTLE_MS * 2);
    expect(mocks.imageFetches.flat().sort()).toEqual(['d.png', 'e.png', 'f.png', 'g.png', 'h.png', 'i.png']);
  });

  // Review finding: pinned to a single delay, this passed only on a lucky phase of the
  // throttle/backoff alignment. Sweeping it covers the batch in which the backoff retry and the
  // throttle's trailing edge land together — the interleaving in which an absolute clear discards
  // the restore.
  it.each([500, 600, 750, 1_000, 1_250])(
    'recovers a range that failed while the user was scrolling elsewhere (scroll at t=%dms)',
    async (delayBeforeScroll) => {
      // Review finding on the original retry: the catch (`prev.length > 0 ? prev : ranges`)
      // dropped the failed range whenever another range had been reported in the meantime — rows
      // the user had scrolled past stayed blank placeholders. The retry now merges the failed
      // ranges with whatever is pending instead of choosing one side, and the clear only fires
      // when the pending state is still the array the pass consumed, so both ranges end up
      // fetched with no further user input.
      const imageNames = ['a.png', 'b.png', 'c.png', 'd.png', 'e.png', 'f.png', 'g.png', 'h.png', 'i.png'];
      mocks.failFetches = true;
      renderHook(imageNames, true);
      scrollTo({ startIndex: 0, endIndex: 2 });
      await advance(delayBeforeScroll);

      // The backend recovers and the user scrolls to a disjoint range while the backoff retry for
      // the failed range is still pending.
      mocks.failFetches = false;
      scrollTo({ startIndex: 6, endIndex: 8 });
      scrollTo({ startIndex: 6, endIndex: 8 });
      await advance(10_000);

      // Both the failed range and the new one land, with no user input beyond the one scroll —
      // and nothing outside the reported ranges is fetched.
      expect([...mocks.cachedImageNames].sort()).toEqual(['a.png', 'b.png', 'c.png', 'g.png', 'h.png', 'i.png']);
    }
  );

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
