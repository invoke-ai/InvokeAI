import { describe, expect, it } from 'vitest';

import {
  panBy,
  pinchZoomAtPoints,
  snapWheelZoom,
  WHEEL_ZOOM_SNAP_IDLE_MS,
  WHEEL_ZOOM_SNAP_POINTS,
  zoomAtPoint,
  type WheelZoomSnap,
} from './panZoom';

const clamp = (min: number, max: number) => (zoom: number) => Math.max(min, Math.min(max, zoom));

describe('panZoom', () => {
  it('keeps the anchored content point fixed when zoom changes', () => {
    const initial = { pan: { x: 30, y: -12 }, zoom: 1.5 };
    const anchor = { x: 200, y: 120 };
    const contentPoint = {
      x: (anchor.x - initial.pan.x) / initial.zoom,
      y: (anchor.y - initial.pan.y) / initial.zoom,
    };

    const next = zoomAtPoint(initial, 3, anchor, clamp(0.1, 20));

    expect(next.zoom).toBe(3);
    expect(next.zoom * contentPoint.x + next.pan.x).toBeCloseTo(anchor.x, 6);
    expect(next.zoom * contentPoint.y + next.pan.y).toBeCloseTo(anchor.y, 6);
  });

  it('applies the zoom constraint before calculating anchor translation', () => {
    const atMaximum = { pan: { x: -700, y: -525 }, zoom: 8 };

    const next = zoomAtPoint(atMaximum, 9.5, { x: 200, y: 150 }, clamp(1, 8));

    expect(next).toBe(atMaximum);
  });

  it('pans in screen space without changing zoom', () => {
    expect(panBy({ pan: { x: 5, y: 5 }, zoom: 2 }, { x: 10, y: -4 })).toEqual({
      pan: { x: 15, y: 1 },
      zoom: 2,
    });
  });

  it('scales a pinch by how far the fingers spread and follows their midpoint', () => {
    const start = { pan: { x: 0, y: 0 }, zoom: 1 };

    const next = pinchZoomAtPoints(
      start,
      { center: { x: 140, y: 60 }, distance: 200, startCenter: { x: 100, y: 50 }, startDistance: 100 },
      clamp(1, 8)
    );

    expect(next.zoom).toBe(2);
    expect(next.zoom * 100 + next.pan.x).toBeCloseTo(140, 6);
    expect(next.zoom * 50 + next.pan.y).toBeCloseTo(60, 6);
  });

  it('keeps panning with the fingers after the pinch is clamped at its zoom limit', () => {
    const start = { pan: { x: -100, y: -50 }, zoom: 4 };
    const gesture = { center: { x: 180, y: 90 }, startCenter: { x: 100, y: 50 }, startDistance: 100 };

    const atLimit = pinchZoomAtPoints(start, { ...gesture, distance: 200 }, clamp(1, 8));
    const pastLimit = pinchZoomAtPoints(start, { ...gesture, distance: 400 }, clamp(1, 8));

    expect(atLimit.zoom).toBe(8);
    expect(pastLimit.zoom).toBe(8);
    // Clamping zoom must still allow midpoint translation.
    expect(pastLimit.pan).toEqual(atLimit.pan);
    expect(pastLimit.zoom * 50 + pastLimit.pan.x).toBeCloseTo(180, 6);
  });

  it('leaves the transform untouched when the pinch has no measurable start', () => {
    const start = { pan: { x: 3, y: 7 }, zoom: 2 };

    expect(
      pinchZoomAtPoints(
        start,
        { center: { x: 10, y: 10 }, distance: 50, startCenter: { x: 0, y: 0 }, startDistance: 0 },
        clamp(1, 8)
      )
    ).toBe(start);
  });
});

describe('snapWheelZoom', () => {
  const options = { constrainZoom: clamp(0.1, 20), now: 0 };
  /** Wheel steps from `zoom`, each fed the previous result, as consecutive wheel events 16 ms apart are. */
  const wheel = (zoom: number, deltas: number[], snap: WheelZoomSnap | null = null, start = 0): WheelZoomSnap[] => {
    const steps: WheelZoomSnap[] = [];
    let state = snap;
    let current = zoom;
    for (const [index, deltaY] of deltas.entries()) {
      state = snapWheelZoom(state, current, deltaY, { ...options, now: start + index * 16 });
      current = state.zoom;
      steps.push(state);
    }
    return steps;
  };

  it('rests on 100% when a notch lands within tolerance of it, from below and from above', () => {
    // One mouse notch is a factor of about 1.16: 87% → 101.1% and 117% → 100.7%.
    expect(wheel(0.87, [-100])[0]!.zoom).toBe(1);
    expect(wheel(1.17, [100])[0]!.zoom).toBe(1);
  });

  it('stays on the snap point across small deltas until the intended zoom leaves its tolerance', () => {
    // Each trackpad step asks for about 0.75% more.
    const steps = wheel(1, [-5, -5, -5]);

    expect(steps.map((step) => step.zoom).slice(0, 2)).toEqual([1, 1]);
    // The third step takes the intended zoom past 102%, and zoom continues from there rather than from 100%.
    expect(steps[2]!.activeSnapPoint).toBeNull();
    expect(steps[2]!.zoom).toBeCloseTo(Math.exp(0.0015 * 15), 9);
  });

  it('lets a pause release the wheel movement left over inside a snap point', () => {
    // Two steps ask for about 101.5% while resting on 100%.
    const resting = wheel(1, [-5, -5]).at(-1)!;
    expect(resting).toMatchObject({ activeSnapPoint: 1, zoom: 1 });

    // Without a pause, one more step crosses 102% and leaves the snap point...
    expect(wheel(1, [-5], resting, resting.at + 16)[0]!.activeSnapPoint).toBeNull();
    // ...after a pause, the gesture starts again from the 100% shown, and the same step stays on it.
    expect(wheel(1, [-5], resting, resting.at + WHEEL_ZOOM_SNAP_IDLE_MS + 1)[0]).toMatchObject({
      activeSnapPoint: 1,
      zoom: 1,
    });
  });

  it('rests on every snap point', () => {
    for (const point of WHEEL_ZOOM_SNAP_POINTS) {
      expect(wheel(point * 0.99, [-1])[0]).toMatchObject({ activeSnapPoint: point, zoom: point });
    }
  });

  it('passes a snap point that a single step jumps over', () => {
    expect(wheel(0.9, [-150])[0]!.zoom).toBeCloseTo(0.9 * Math.exp(0.225), 9);
  });

  it('starts afresh when another change moved the zoom since the last step', () => {
    const [snapped] = wheel(1, [-1]);

    // Resting on 100%, then a preset set 130%: the next step starts from 130%, not from the remembered 100%.
    const next = snapWheelZoom(snapped!, 1.3, -1, { ...options, now: 16 });

    expect(next.activeSnapPoint).toBeNull();
    expect(next.zoom).toBeCloseTo(1.3 * Math.exp(0.0015), 9);
  });

  it('never rests on a snap point outside the zoom limits', () => {
    const next = snapWheelZoom(null, 0.26, 10, { constrainZoom: clamp(0.26, 20), now: 0 });

    expect(next.zoom).toBe(0.26);
  });
});
