import { act, Activity, useRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { usePreservedScrollOffset } from './usePreservedScrollOffset';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const VIEWPORT_STYLE = { height: '100px', overflow: 'auto', width: '100px' } as const;

/**
 * Stands in for a virtualized list: the scrollable content is sized from a
 * measurement, so it does not survive the container being taken out of layout.
 * That is what defeats the browser's own scroll restoration, and a container
 * with fixed content would not reproduce the bug at all.
 */
const Scroller = ({ contentHeight, isPreserved }: { contentHeight: number; isPreserved: boolean }) => {
  const preservedRef = useRef<HTMLDivElement>(null);
  const plainRef = useRef<HTMLDivElement>(null);

  usePreservedScrollOffset(isPreserved ? preservedRef : plainRef);

  return (
    <div ref={isPreserved ? preservedRef : undefined} data-testid="scroller" style={VIEWPORT_STYLE}>
      <div style={{ height: `${String(contentHeight)}px` }} />
    </div>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const render = async (mode: 'hidden' | 'visible', contentHeight: number, isPreserved: boolean) => {
  await act(async () => {
    root?.render(
      <Activity mode={mode}>
        <Scroller contentHeight={contentHeight} isPreserved={isPreserved} />
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

/** Hides the container, collapses its measured content, then shows it again. */
const hideAndShow = async (isPreserved: boolean) => {
  await render('hidden', 0, isPreserved);
  await render('visible', 2000, isPreserved);
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
    // runs the actual virtualized gallery.
    expect(scroller()).toBe(element);
    expect(element.scrollTop).toBe(500);
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
