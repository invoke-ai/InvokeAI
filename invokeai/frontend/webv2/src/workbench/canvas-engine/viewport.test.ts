import type { Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint } from '@workbench/canvas-engine/math/mat2d';
import { ZOOM_SNAP_CANDIDATES } from '@workbench/canvas-engine/math/snapping';
import { describe, expect, it, vi } from 'vitest';

import { createViewport } from './viewport';

const closeTo = (a: Vec2, b: Vec2, eps = 1e-6): void => {
  expect(a.x).toBeCloseTo(b.x, 6);
  expect(a.y).toBeCloseTo(b.y, 6);
  void eps;
};

describe('createViewport', () => {
  it('round-trips screen↔document with pan and zoom', () => {
    const vp = createViewport({ pan: { x: 30, y: -12 }, zoom: 1.5 });
    const doc: Vec2 = { x: 42, y: 17 };
    const screen = vp.documentToScreen(doc);
    // screen = zoom*doc + pan
    closeTo(screen, { x: 1.5 * 42 + 30, y: 1.5 * 17 - 12 });
    closeTo(vp.screenToDocument(screen), doc);
  });

  it('viewMatrix folds dpr into scale and translation (device = dpr * screen)', () => {
    const vp = createViewport({ pan: { x: 10, y: 5 }, zoom: 2 });
    const dpr = 2;
    const m = vp.viewMatrix(dpr);
    const doc: Vec2 = { x: 7, y: 9 };
    const device = applyToPoint(m, doc);
    const screen = vp.documentToScreen(doc);
    closeTo(device, { x: screen.x * dpr, y: screen.y * dpr });
  });

  it('zoomAtPoint keeps the anchored document point fixed on screen', () => {
    const vp = createViewport({ pan: { x: 0, y: 0 }, zoom: 1 });
    const anchor: Vec2 = { x: 200, y: 120 };
    const docUnderAnchorBefore = vp.screenToDocument(anchor);
    vp.zoomAtPoint(3, anchor);
    expect(vp.getZoom()).toBe(3);
    // The document point under the anchor must project back to the same screen anchor.
    closeTo(vp.documentToScreen(docUnderAnchorBefore), anchor);
  });

  it('wheelZoom snaps to a candidate when the exponential step lands near one', () => {
    const vp = createViewport({ pan: { x: 0, y: 0 }, zoom: 1 });
    const anchor: Vec2 = { x: 0, y: 0 };
    // A tiny negative deltaY nudges zoom just above 1; it should snap back to 1.
    vp.wheelZoom(-1, anchor);
    expect(vp.getZoom()).toBe(1);
    expect(ZOOM_SNAP_CANDIDATES).toContain(vp.getZoom());
  });

  it('wheelZoom zooms out on positive deltaY and in on negative deltaY', () => {
    const vpOut = createViewport({ zoom: 1 });
    vpOut.wheelZoom(400, { x: 0, y: 0 });
    expect(vpOut.getZoom()).toBeLessThan(1);

    const vpIn = createViewport({ zoom: 1 });
    vpIn.wheelZoom(-400, { x: 0, y: 0 });
    expect(vpIn.getZoom()).toBeGreaterThan(1);
  });

  it('fitToView centers the document and applies padding-limited zoom', () => {
    const vp = createViewport();
    const documentRect = { height: 100, width: 200, x: 0, y: 0 };
    const viewportSize = { height: 400, width: 400 };
    const padding = 50;
    vp.fitToView(documentRect, viewportSize, padding);

    // avail = 400 - 2*50 = 300; zoom = min(300/200, 300/100) = 1.5
    expect(vp.getZoom()).toBeCloseTo(1.5, 6);
    // Document center maps to viewport center.
    const docCenter: Vec2 = { x: 100, y: 50 };
    closeTo(vp.documentToScreen(docCenter), { x: 200, y: 200 });
  });

  it('fitToView is a no-op for a degenerate viewport', () => {
    const vp = createViewport({ zoom: 2 });
    vp.fitToView({ height: 100, width: 100, x: 0, y: 0 }, { height: 0, width: 0 });
    expect(vp.getZoom()).toBe(2);
  });

  it('panBy translates in screen space', () => {
    const vp = createViewport({ pan: { x: 5, y: 5 }, zoom: 2 });
    vp.panBy({ x: 10, y: -4 });
    expect(vp.getState().pan).toEqual({ x: 15, y: 1 });
  });

  it('setViewportSize clamps dpr to the max and notifies on change', () => {
    const vp = createViewport();
    const listener = vi.fn();
    vp.subscribe(listener);
    vp.setViewportSize(800, 600, 3);
    expect(vp.getDpr()).toBe(2);
    expect(vp.getViewportSize()).toEqual({ height: 600, width: 800 });
    expect(listener).toHaveBeenCalledTimes(1);
    // Identical call does not re-notify.
    vp.setViewportSize(800, 600, 3);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('notifies subscribers on zoom/pan and stops after unsubscribe', () => {
    const vp = createViewport();
    const listener = vi.fn();
    const unsubscribe = vp.subscribe(listener);
    vp.panBy({ x: 1, y: 0 });
    vp.zoomAtPoint(2, { x: 0, y: 0 });
    expect(listener).toHaveBeenCalledTimes(2);
    unsubscribe();
    vp.panBy({ x: 1, y: 0 });
    expect(listener).toHaveBeenCalledTimes(2);
  });
});
