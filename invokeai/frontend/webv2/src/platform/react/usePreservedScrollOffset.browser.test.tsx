import { act, Activity, useRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { usePreservedScrollOffset } from './usePreservedScrollOffset';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const VIEWPORT_STYLE = { height: '100px', overflow: 'auto', width: '100px' } as const;

/**
 * Use layout-dependent content so hiding changes its size; fixed-height virtualized content would not reproduce
 * lost browser restoration.
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

    // This asserts retained state; the real gallery reproduction lives in workbench-keep-alive-state.
    expect(scroller()).toBe(element);
    expect(element.scrollTop).toBe(500);
  });

  it('keeps both axes across a keep-alive hide and show', async () => {
    // Cover horizontal offsets too: Scrollable installs the hook for filmstrips regardless of orientation.
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
