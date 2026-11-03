import { act, useRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { afterEach, describe, expect, it, vi } from 'vitest';

/**
 * Browsers clamp an offset written past shorter content and report it late or never, e.g. on scroll-element
 * replacement. Verify the local virtualizer patch reconciles actual position instead of rendering a stale range.
 */

const ROW_HEIGHT = 50;

/** The live virtualizer instance's scroll bookkeeping, so a test can stage and assert it. */
type ScrollInternals = {
  _holdOffset: number;
  _reportedOverscroll: boolean;
  pendingWrite: { smooth: boolean; target: number } | null;
  scrollOffset: number | null;
};

let instance: ScrollInternals | null = null;

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
  /** Sets an own scrollTop value to model pre-event state without Chromium racing the assertion with a native event. */
  const stageScrollTop = (scroller: HTMLElement | null, offset: number) => {
    Object.defineProperty(scroller, 'scrollTop', { configurable: true, value: offset, writable: true });
  };

  it('still exposes the private virtual-core members the local patch reads', async () => {
    // patches/react-hook-tanstack-virtual.patch reads these members. If upstream renames one, the reconcile
    // quietly stops running, and the tests below still pass because they set these members themselves.
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await render(1000, true);
    const members = instance as unknown as Record<string, unknown>;

    expect(typeof members._holdOffset).toBe('number');
    expect(Object.hasOwn(members, 'pendingWrite')).toBe(true);
    expect(typeof members.maybeNotify).toBe('function');
  });

  it('does not consume a programmatic scroll\u2019s pending write while reconciling', async () => {
    // Clearing the pending write would make its scroll event match the reconciled offset and be dropped as
    // spurious, leaving scrolling state and an in-flight scrollToIndex without it.
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await render(1000, true);
    const scroller = host.querySelector<HTMLElement>('[data-testid="scroller"]');
    const internals = instance!;
    const pendingWrite = { smooth: false, target: 400 * ROW_HEIGHT };

    await act(() => {
      stageScrollTop(scroller, 400 * ROW_HEIGHT);
      internals.scrollOffset = 0;
      internals.pendingWrite = pendingWrite;
      root?.render(<VirtualList count={1000} isListMounted />);
    });

    // The reconciliation ran: the cached offset now matches the element...
    expect(internals.scrollOffset).toBe(400 * ROW_HEIGHT);
    // ...and it left the pending write for its scroll event.
    expect(internals.pendingWrite).toBe(pendingWrite);
  });

  it('keeps a held scroll correction instead of reconciling it away', async () => {
    // While a correction is held, the layout offset differs from the element by it on purpose; adopting the
    // element's offset would drop the correction and shift the rows under the reader.
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await render(1000, true);
    const scroller = host.querySelector<HTMLElement>('[data-testid="scroller"]');
    const internals = instance!;

    await act(() => {
      stageScrollTop(scroller, 400 * ROW_HEIGHT);
      internals.scrollOffset = 400 * ROW_HEIGHT + 100;
      internals._holdOffset = 100;
      // A reported rubber-band bounce keeps the hold from being written during the commit.
      internals._reportedOverscroll = true;
      root?.render(<VirtualList count={1000} isListMounted />);
    });

    expect(internals._holdOffset).toBe(100);
    expect(internals.scrollOffset).toBe(400 * ROW_HEIGHT + 100);
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
