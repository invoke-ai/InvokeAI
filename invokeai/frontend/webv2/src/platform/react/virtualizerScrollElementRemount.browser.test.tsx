import { act, useRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { afterEach, describe, expect, it, vi } from 'vitest';

/**
 * Regression test for the locally patched scroll-desync in
 * react-hook-tanstack-virtual / @tanstack/virtual-core.
 *
 * When a scroll element is swapped (a list passing through an empty loading
 * state remounts its container), virtual-core restores the cached offset onto
 * the new element via `scrollTo`. If the new content is too short for that
 * offset, the browser clamps the scroll WITHOUT firing a scroll event, and
 * nothing ever reads the element's real position back — the virtualizer keeps
 * serving the range computed at the stale offset, which renders no rows at
 * all. This is the gallery going blank after a semantic search or cluster
 * click until a manual scroll. The patch reconciles the cached offset with
 * the element's actual position whenever the scroll element changes.
 */

const ROW_HEIGHT = 50;

/** The live virtualizer instance, so a test can assert its internal scroll bookkeeping. */
let instance: {
  isScrolling: boolean;
  scrollDirection: string | null;
  scrollToIndex: (index: number) => void;
} | null = null;

const VirtualList = ({ count, isListMounted }: { count: number; isListMounted: boolean }) => {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const virtualizer = useVirtualizer(
    {
      count,
      estimateSize: () => ROW_HEIGHT,
      getScrollElement: () => scrollRef.current,
      overscan: 0,
    },
    (snapshot, virtualizerInstance) => {
      instance = virtualizerInstance as unknown as typeof instance;

      return snapshot;
    }
  );

  if (!isListMounted) {
    return <p>loading</p>;
  }

  return (
    <div ref={scrollRef} data-testid="scroller" style={{ height: '200px', overflow: 'auto' }}>
      <div style={{ height: `${virtualizer.totalSize}px`, position: 'relative' }}>
        {virtualizer.virtualItems.map((item) => (
          <div
            key={item.key}
            data-index={item.index}
            style={{
              height: `${ROW_HEIGHT}px`,
              left: 0,
              position: 'absolute',
              top: 0,
              transform: `translateY(${item.start}px)`,
              width: '100%',
            }}
          >
            row {item.index}
          </div>
        ))}
      </div>
    </div>
  );
};

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const render = async (count: number, isListMounted: boolean) => {
  await act(async () => {
    root?.render(<VirtualList count={count} isListMounted={isListMounted} />);
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  });
};

const renderedIndexes = (): number[] =>
  [...(host?.querySelectorAll<HTMLElement>('[data-index]') ?? [])].map((row) => Number(row.dataset.index));

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  instance = null;
  root = null;
});

describe('useVirtualizer scroll element remount', () => {
  it('does not consume a programmatic scroll\u2019s pending intent while reconciling', async () => {
    // virtual-core ignores an incoming scroll event when `isScrolling &&
    // _intendedScrollOffset === null && offset === scrollOffset`. That guard
    // exists so a PROGRAMMATIC scroll's own event still gets through, and it
    // is armed by `_intendedScrollOffset` being non-null. If the
    // reconciliation writes scrollOffset AND clears the intent, all three
    // conditions hold, so every scrollToIndex swallows its own event and
    // leaves `isScrolling` false and `scrollDirection` stale — which is what
    // virtual-core steers dynamic re-measurement by.
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await render(1000, true);
    const scroller = host.querySelector<HTMLElement>('[data-testid="scroller"]');
    const internals = instance as unknown as {
      _intendedScrollOffset: number | null;
      scrollOffset: number | null;
    };

    // The state a programmatic scroll leaves behind: the element has moved,
    // the intent is recorded, and the cached offset is still stale because no
    // scroll event has been delivered yet. Staged and committed in one
    // synchronous block so the commit — and with it the reconciliation — runs
    // before the browser can deliver that event, which is the ordering every
    // scrollToIndex call site in this app produces.
    await act(() => {
      scroller!.scrollTop = 400 * ROW_HEIGHT;
      internals.scrollOffset = 0;
      internals._intendedScrollOffset = 400 * ROW_HEIGHT;
      root?.render(<VirtualList count={1000} isListMounted />);
    });

    // The reconciliation ran: the cached offset now matches the element...
    expect(internals.scrollOffset).toBe(400 * ROW_HEIGHT);
    // ...and it left the pending intent alone, so the guard stays armed.
    expect(internals._intendedScrollOffset).toBe(400 * ROW_HEIGHT);
  });

  it('recovers when the restored offset is clamped by shorter remounted content', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    // A long list, scrolled deep on the FIRST scroll element.
    await render(1000, true);
    const scroller = host.querySelector<HTMLElement>('[data-testid="scroller"]');

    await act(async () => {
      scroller!.scrollTop = 400 * ROW_HEIGHT;
      scroller!.dispatchEvent(new Event('scroll'));
      // End the scroll like the browser would; the reconciliation deliberately
      // stands down while a scroll is live.
      scroller!.dispatchEvent(new Event('scrollend'));
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
    expect(renderedIndexes()).toContain(400);

    // The container unmounts (a loading state) and remounts as a NEW element
    // holding far fewer, unscrollable rows — a search-results swap. The
    // core's offset restoration gets clamped to 0 by the browser with no
    // scroll event; without the reconciliation patch the virtualizer keeps
    // the stale deep offset and renders NO rows at all.
    await render(1000, false);
    await render(3, true);

    // The reconciliation notifies from a layout effect; the corrected render
    // can land a scheduler tick later.
    await vi.waitFor(() => {
      expect(renderedIndexes()).toEqual([0, 1, 2]);
    });
  });
});
