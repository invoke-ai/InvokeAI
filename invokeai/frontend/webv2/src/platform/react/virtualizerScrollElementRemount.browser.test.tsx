import { act, useRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { afterEach, describe, expect, it, vi } from 'vitest';

/**
 * On scroll-element replacement, browsers can clamp restored offsets without an event. Verify the local
 * virtualizer patch reconciles actual position instead of rendering a stale empty range.
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
    // Retain programmatic scroll intent during reconciliation; clearing it suppresses the matching event and
    // leaves scrolling state/direction stale.
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await render(1000, true);
    const scroller = host.querySelector<HTMLElement>('[data-testid="scroller"]');
    const internals = instance as unknown as {
      _intendedScrollOffset: number | null;
      scrollOffset: number | null;
    };

    // Set an own scrollTop value to model pre-event state without Chromium racing the assertion by delivering a
    // native scroll event.
    await act(() => {
      Object.defineProperty(scroller, 'scrollTop', {
        configurable: true,
        value: 400 * ROW_HEIGHT,
        writable: true,
      });
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
      // End the active scroll before testing reconciliation, which intentionally waits for idle.
      scroller!.dispatchEvent(new Event('scrollend'));
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });
    });
    expect(renderedIndexes()).toContain(400);

    // Remount with unscrollable results so the browser clamps the old offset without an event.
    await render(1000, false);
    await render(3, true);

    // Poll for the rerender scheduled by layout-effect reconciliation.
    await vi.waitFor(() => {
      expect(renderedIndexes()).toEqual([0, 1, 2]);
    });
  });
});
