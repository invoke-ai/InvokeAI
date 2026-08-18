import { act, Activity, useRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { usePreservedScrollOffset } from './usePreservedScrollOffset';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const VIEWPORT_STYLE = { height: '100px', overflow: 'auto', width: '100px' } as const;

/**
 * Stands in for a plain, non-virtualized list — the gallery's board list is
 * the real reproduction case — whose content is sized from actual DOM layout
 * rather than from a row count, so it does not survive the container being
 * taken out of layout unchanged. That is what defeats the browser's own
 * scroll restoration, and a container with fixed content would not reproduce
 * the bug at all. A virtualized list does not have this problem: its content
 * height is a pure function of row count and estimated sizes, so it survives
 * unaided and needs no help from this hook.
 */
const Scroller = ({
  contentHeight,
  contentWidth = 0,
  isPreserved,
}: {
  contentHeight: number;
  contentWidth?: number;
  isPreserved: boolean;
}) => {
  const preservedRef = useRef<HTMLDivElement>(null);
  const plainRef = useRef<HTMLDivElement>(null);

  usePreservedScrollOffset(isPreserved ? preservedRef : plainRef);

  return (
    <div ref={isPreserved ? preservedRef : undefined} data-testid="scroller" style={VIEWPORT_STYLE}>
      <div style={{ height: `${String(contentHeight)}px`, width: `${String(contentWidth)}px` }} />
    </div>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const render = async (mode: 'hidden' | 'visible', contentHeight: number, isPreserved: boolean, contentWidth = 0) => {
  await act(async () => {
    root?.render(
      <Activity mode={mode}>
        <Scroller contentHeight={contentHeight} contentWidth={contentWidth} isPreserved={isPreserved} />
      </Activity>
    );
    await new Promise((resolve) => {
      requestAnimationFrame(resolve);
    });
  });
};

const scroller = () => {
  const element = host?.querySelector<HTMLDivElement>('[data-testid="scroller"]');

  if (!element) {
    throw new Error('Expected the scroll container to be mounted.');
  }

  return element;
};

/** Scrolls and lets the real scroll event land, the way a user gesture would. */
const scrollTo = async (offset: number) => {
  const element = scroller();

  await act(async () => {
    element.scrollTop = offset;
    await new Promise((resolve) => {
      requestAnimationFrame(resolve);
    });
  });
};

/** Scrolls both axes and lets the real scroll event land. */
const scrollToBothAxes = async (top: number, left: number) => {
  const element = scroller();

  await act(async () => {
    element.scrollTop = top;
    element.scrollLeft = left;
    await new Promise((resolve) => {
      requestAnimationFrame(resolve);
    });
  });
};

/** Hides the container, collapses its measured content, then shows it again. */
const hideAndShow = async (isPreserved: boolean, contentWidth = 0) => {
  await render('hidden', 0, isPreserved, 0);
  await render('visible', 2000, isPreserved, contentWidth);
};

const mount = () => {
  host = document.createElement('div');
  host.style.cssText = 'height:200px;width:200px;';
  document.body.append(host);
  root = createRoot(host);
};

afterEach(async () => {
  await act(async () => {
    root?.unmount();
    await Promise.resolve();
  });
  host?.remove();
  host = null;
  root = null;
});

describe('preserved scroll offset', () => {
  it('keeps the offset across a keep-alive hide and show', async () => {
    mount();
    await render('visible', 2000, true);
    await scrollTo(500);

    const element = scroller();

    await hideAndShow(true);

    // The state, not the node: identity alone was never the thing at risk.
    //
    // Chrome restores this much on its own when the scrollable content happens
    // to survive, so this pins the contract rather than reproducing the bug —
    // the real reproduction is the `workbench-keep-alive-state` journey, which
    // runs the gallery's actual non-virtualized board list.
    expect(scroller()).toBe(element);
    expect(element.scrollTop).toBe(500);
  });

  it('keeps both axes across a keep-alive hide and show', async () => {
    // `Scrollable` installs this hook unconditionally regardless of its own
    // `orientation` prop, and `PreviewFilmstrip` is a horizontal `Scrollable`
    // inside the keep-alive-able Preview widget — a vertical-only fix would
    // leave that filmstrip snapping back to its start on every preset switch.
    mount();
    await render('visible', 2000, true, 2000);
    await scrollToBothAxes(500, 300);

    await hideAndShow(true, 2000);

    expect(scroller().scrollTop).toBe(500);
    expect(scroller().scrollLeft).toBe(300);
  });

  it('does not carry an offset into a genuinely new instance', async () => {
    mount();
    await render('visible', 2000, true);
    await scrollTo(500);

    await act(async () => {
      root?.unmount();
      await Promise.resolve();
    });
    root = createRoot(host as HTMLDivElement);
    await render('visible', 2000, true);

    expect(scroller().scrollTop).toBe(0);
  });
});
