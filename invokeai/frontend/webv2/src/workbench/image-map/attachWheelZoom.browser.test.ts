import { afterEach, describe, expect, it } from 'vitest';

import type { AxisRanges } from './imageMapViewport';

import { attachWheelZoom } from './attachWheelZoom';

/**
 * The gesture bookkeeping needs real pointer and touch events, so this runs in
 * the browser suite; the range arithmetic it drives is unit-tested against
 * `imageMapViewport` directly.
 */

let detach: (() => void) | null = null;
let host: HTMLElement | null = null;

const mountHost = (): HTMLElement => {
  const element = document.createElement('div');

  element.style.height = '400px';
  element.style.left = '0';
  element.style.position = 'fixed';
  element.style.top = '0';
  element.style.width = '400px';
  document.body.append(element);
  host = element;

  return element;
};

const attach = (element: HTMLElement, ranges: AxisRanges): { current: AxisRanges } => {
  const state = { current: ranges };

  detach = attachWheelZoom(element, {
    applyRanges: (next) => {
      state.current = next;
    },
    readRanges: () => state.current,
  });

  return state;
};

const spanOf = (ranges: AxisRanges): number => ranges.x[1] - ranges.x[0];

const touchAt = (element: HTMLElement, identifier: number, clientX: number): Touch =>
  new Touch({ clientX, clientY: 200, identifier, target: element });

const dispatchTouch = (element: HTMLElement, type: string, touches: Touch[]): void => {
  element.dispatchEvent(new TouchEvent(type, { bubbles: true, cancelable: true, touches }));
};

afterEach(() => {
  detach?.();
  detach = null;
  host?.remove();
  host = null;
});

describe('attachWheelZoom pinch bookkeeping', () => {
  it('re-baselines when a third finger lifts instead of jumping the viewport', () => {
    const element = mountHost();
    const state = attach(element, { x: [0, 100], y: [0, 100] });

    // Two fingers 100px apart establish the baseline.
    dispatchTouch(element, 'touchstart', [touchAt(element, 1, 100), touchAt(element, 2, 200)]);

    // A third finger suspends the pinch; plotly pans while it is down.
    dispatchTouch(element, 'touchstart', [
      touchAt(element, 1, 100),
      touchAt(element, 2, 200),
      touchAt(element, 3, 300),
    ]);
    dispatchTouch(element, 'touchmove', [touchAt(element, 1, 10), touchAt(element, 2, 390), touchAt(element, 3, 300)]);

    // Back to two fingers, now far apart. Measuring against the stale 100px
    // baseline would scale the view by ~1/3.8 in a single frame.
    dispatchTouch(element, 'touchend', [touchAt(element, 1, 10), touchAt(element, 2, 390)]);
    const spanBefore = spanOf(state.current);

    dispatchTouch(element, 'touchmove', [touchAt(element, 1, 10), touchAt(element, 2, 390)]);

    expect(spanOf(state.current)).toBeCloseTo(spanBefore, 6);
  });

  it('still pinches normally after the interruption', () => {
    const element = mountHost();
    const state = attach(element, { x: [0, 100], y: [0, 100] });

    dispatchTouch(element, 'touchstart', [touchAt(element, 1, 150), touchAt(element, 2, 250)]);
    const spanBefore = spanOf(state.current);

    // Fingers apart by 2x zooms in, halving the visible span.
    dispatchTouch(element, 'touchmove', [touchAt(element, 1, 100), touchAt(element, 2, 300)]);

    expect(spanOf(state.current)).toBeCloseTo(spanBefore / 2, 6);
  });
});

describe('attachWheelZoom wheel deltas', () => {
  it('zooms usefully for a line-mode wheel, as Firefox reports one', () => {
    const element = mountHost();
    const state = attach(element, { x: [0, 100], y: [0, 100] });

    element.dispatchEvent(
      new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 200, clientY: 200, deltaMode: 1, deltaY: 3 })
    );

    // Read as pixels this would be a 0.3% change, which is no zoom at all.
    expect(spanOf(state.current)).toBeGreaterThan(102);
  });

  it('does not let ctrl held over a real wheel jump the view', () => {
    const element = mountHost();
    const state = attach(element, { x: [0, 100], y: [0, 100] });

    element.dispatchEvent(
      new WheelEvent('wheel', {
        bubbles: true,
        cancelable: true,
        clientX: 200,
        clientY: 200,
        ctrlKey: true,
        deltaY: 100,
      })
    );

    // The trackpad-pinch gain on a mouse-sized delta would be ~2.7x.
    expect(spanOf(state.current)).toBeLessThan(125);
  });
});
