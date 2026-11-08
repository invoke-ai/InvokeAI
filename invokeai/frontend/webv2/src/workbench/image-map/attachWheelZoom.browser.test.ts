import { afterEach, describe, expect, it } from 'vitest';

import type { AxisRanges } from './imageMapViewport';

import { attachWheelZoom } from './attachWheelZoom';

/** Browser events verify gesture bookkeeping; unit tests cover viewport arithmetic. */

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

const attach = (element: HTMLElement, ranges: AxisRanges): { applied: number; current: AxisRanges } => {
  const state = { applied: 0, current: ranges };

  detach = attachWheelZoom(element, {
    applyRanges: (next) => {
      state.applied += 1;
      state.current = next;
    },
    readRanges: () => state.current,
  });

  return state;
};

/**
 * Zooms are coalesced into the next animation frame, so nothing is applied
 * until one passes. Two frames, because the first only guarantees the
 * callback has been scheduled by the dispatch above.
 */
const settleFrames = (): Promise<void> =>
  new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });

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
  it('re-baselines when a third finger lifts instead of jumping the viewport', async () => {
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
    await settleFrames();
    const spanBefore = spanOf(state.current);

    const appliedBefore = state.applied;

    dispatchTouch(element, 'touchmove', [touchAt(element, 1, 10), touchAt(element, 2, 390)]);
    await settleFrames();

    // "Nothing moved" has to mean the gesture ran and correctly produced no
    // movement — a zoom that never applied would satisfy the span check too.
    expect(state.applied).toBe(appliedBefore);
    expect(spanOf(state.current)).toBeCloseTo(spanBefore, 6);
  });

  it('still pinches normally after the interruption', async () => {
    const element = mountHost();
    const state = attach(element, { x: [0, 100], y: [0, 100] });

    dispatchTouch(element, 'touchstart', [touchAt(element, 1, 150), touchAt(element, 2, 250)]);
    const spanBefore = spanOf(state.current);

    // Fingers apart by 2x zooms in, halving the visible span.
    dispatchTouch(element, 'touchmove', [touchAt(element, 1, 100), touchAt(element, 2, 300)]);
    await settleFrames();

    expect(spanOf(state.current)).toBeCloseTo(spanBefore / 2, 6);
  });
});

describe('attachWheelZoom wheel deltas', () => {
  it('zooms usefully for a line-mode wheel, as Firefox reports one', async () => {
    const element = mountHost();
    const state = attach(element, { x: [0, 100], y: [0, 100] });

    element.dispatchEvent(
      new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 200, clientY: 200, deltaMode: 1, deltaY: 3 })
    );
    await settleFrames();

    // Read as pixels this would be a 0.3% change, which is no zoom at all.
    expect(spanOf(state.current)).toBeGreaterThan(102);
  });

  it('applies one zoom per frame however fast the wheel emits', async () => {
    // A wheel outruns a frame by an order of magnitude, and each applied zoom
    // is a plotly relayout over the whole scene: one per event queued
    // relayouts behind the cursor on a large map.
    const element = mountHost();
    const state = attach(element, { x: [0, 100], y: [0, 100] });

    for (let index = 0; index < 12; index++) {
      element.dispatchEvent(
        new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 200, clientY: 200, deltaY: -100 })
      );
    }

    expect(state.applied).toBe(0);

    await settleFrames();

    expect(state.applied).toBe(1);
    // Coalesced, not dropped: twelve zoom-ins compose into one much larger
    // step rather than collapsing to a single event's worth.
    expect(spanOf(state.current)).toBeLessThan(50);
  });

  it('drops a queued zoom when the map goes away before the frame runs', async () => {
    // The owning effect detaches and then purges the plotly div; a frame
    // that survived the detach would relayout a purged container.
    const element = mountHost();
    const state = attach(element, { x: [0, 100], y: [0, 100] });

    element.dispatchEvent(
      new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 200, clientY: 200, deltaY: -100 })
    );
    detach?.();
    detach = null;
    await settleFrames();

    expect(state.applied).toBe(0);
  });

  it('does not let ctrl held over a real wheel jump the view', async () => {
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
    await settleFrames();

    // An unapplied zoom would also satisfy the bound below, so check the
    // gesture actually landed before reading it.
    expect(state.applied).toBe(1);
    // The trackpad-pinch gain on a mouse-sized delta would be ~2.7x.
    expect(spanOf(state.current)).toBeLessThan(125);
  });
});
