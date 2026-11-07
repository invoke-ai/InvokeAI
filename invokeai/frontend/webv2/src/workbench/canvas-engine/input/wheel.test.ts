import type { ControlKeyState } from '@workbench/canvas-engine/input/pointerPipeline';
import type { WheelHandlerDeps } from '@workbench/canvas-engine/input/wheel';
import type { Tool } from '@workbench/canvas-engine/tools/tool';
import type { ToolId } from '@workbench/canvas-engine/types';

import { createWheelHandler } from '@workbench/canvas-engine/input/wheel';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { describe, expect, it, vi } from 'vitest';

const makeWheelEvent = (
  deltaY: number,
  opts: { clientX?: number; ctrlKey?: boolean; deltaX?: number; deltaMode?: number; timeStamp?: number } = {}
): WheelEvent =>
  ({
    altKey: false,
    clientX: opts.clientX ?? 10,
    clientY: 10,
    ctrlKey: opts.ctrlKey ?? false,
    deltaMode: opts.deltaMode ?? 0,
    deltaX: opts.deltaX ?? 0,
    deltaY,
    metaKey: false,
    preventDefault: vi.fn(),
    shiftKey: false,
    timeStamp: opts.timeStamp ?? 0,
  }) as unknown as WheelEvent;

/**
 * `controlKey` defaults to `down`: ctrl+wheel then models a mouse with the Control key held. `enableOnPinch` stands in
 * for a frontend listener that turns trackpad navigation on when a pinch is reported; `armed` says one listens.
 */
const createHarness = (
  tool: Tool | undefined,
  invertBrushSizeScroll = false,
  trackpadNavigation = false,
  input: { armed?: boolean; controlKey?: ControlKeyState; enableOnPinch?: boolean } = {}
) => {
  const viewport = createViewport();
  const stepActiveBrushSize = vi.fn();
  const invalidate = vi.fn();
  const state = { pinchReports: 0, trackpadNavigation };
  const deps: WheelHandlerDeps = {
    getActiveTool: () => tool,
    getControlKeyState: () => input.controlKey ?? 'down',
    getInputElement: () => ({ getBoundingClientRect: () => ({ left: 0, top: 0 }) }) as unknown as HTMLElement,
    getInvertBrushSizeScroll: () => invertBrushSizeScroll,
    getTrackpadNavigation: () => state.trackpadNavigation,
    invalidate,
    isPinchDetectionArmed: () => input.armed ?? false,
    reportTrackpadPinch: () => {
      state.pinchReports += 1;
      if (input.enableOnPinch) {
        state.trackpadNavigation = true;
      }
    },
    stepActiveBrushSize,
    viewport,
  };
  return { handler: createWheelHandler(deps), invalidate, state, stepActiveBrushSize, viewport };
};

const toolWithId = (id: ToolId): Tool => ({ id });

describe('wheel handler: ctrl+wheel brush size step', () => {
  it('grows the size on ctrl+wheel-up when brush is active', () => {
    const h = createHarness(toolWithId('brush'));
    h.handler(makeWheelEvent(-100, { ctrlKey: true }));
    expect(h.stepActiveBrushSize).toHaveBeenCalledWith(1);
  });

  it('shrinks the size on ctrl+wheel-down when eraser is active', () => {
    const h = createHarness(toolWithId('eraser'));
    h.handler(makeWheelEvent(100, { ctrlKey: true }));
    expect(h.stepActiveBrushSize).toHaveBeenCalledWith(-1);
  });

  it('inverts the step direction when the invert-scroll preference is on', () => {
    const grow = createHarness(toolWithId('brush'), true);
    grow.handler(makeWheelEvent(-100, { ctrlKey: true }));
    // Wheel-up normally grows (+1); inverted, it shrinks (-1).
    expect(grow.stepActiveBrushSize).toHaveBeenCalledWith(-1);

    const shrink = createHarness(toolWithId('eraser'), true);
    shrink.handler(makeWheelEvent(100, { ctrlKey: true }));
    // Wheel-down normally shrinks (-1); inverted, it grows (+1).
    expect(shrink.stepActiveBrushSize).toHaveBeenCalledWith(1);
  });

  it('swallows ctrl+wheel (no size step, no zoom) for non-paint tools', () => {
    const h = createHarness(toolWithId('view'));
    const wheelZoom = vi.spyOn(h.viewport, 'wheelZoom');
    h.handler(makeWheelEvent(-100, { ctrlKey: true }));
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
    expect(wheelZoom).not.toHaveBeenCalled();
  });
});

describe('wheel handler: plain wheel', () => {
  it('zooms the viewport around the pointer on plain wheel', () => {
    const h = createHarness(toolWithId('brush'));
    const wheelZoom = vi.spyOn(h.viewport, 'wheelZoom');
    h.handler(makeWheelEvent(-120));
    expect(wheelZoom).toHaveBeenCalledWith(-120, { x: 10, y: 10 });
    expect(h.invalidate).toHaveBeenCalledWith({ view: true });
  });
});

describe('wheel handler: trackpad navigation', () => {
  const trackpadHarness = (tool: Tool | undefined = toolWithId('brush')) => createHarness(tool, false, true);

  it('pans on both axes so content follows the fingers, without zooming', () => {
    const h = trackpadHarness();
    h.handler(makeWheelEvent(30, { deltaX: -12 }));
    expect(h.viewport.getState()).toEqual({ pan: { x: 12, y: -30 }, zoom: 1 });
    expect(h.invalidate).toHaveBeenCalledWith({ view: true });
  });

  it('converts line deltas at 16px per line and page deltas at the viewport size', () => {
    const lines = trackpadHarness();
    lines.handler(makeWheelEvent(3, { deltaMode: 1, deltaX: 1 }));
    expect(lines.viewport.getState().pan).toEqual({ x: -16, y: -48 });

    const pages = trackpadHarness();
    pages.viewport.setViewportSize(800, 600, 1);
    pages.handler(makeWheelEvent(1, { deltaMode: 2, deltaX: -0.5 }));
    expect(pages.viewport.getState().pan).toEqual({ x: 400, y: -600 });
  });

  it('zooms continuously around the pointer on ctrl+wheel (pinch)', () => {
    const h = trackpadHarness();
    const anchor = { x: 10, y: 10 };
    const documentUnderPointer = h.viewport.screenToDocument(anchor);

    // Spreading the fingers arrives as negative deltaY: zoom in, unsnapped.
    h.handler(makeWheelEvent(-10, { ctrlKey: true }));
    expect(h.viewport.getZoom()).toBeCloseTo(Math.exp(0.1), 10);
    const afterIn = h.viewport.screenToDocument(anchor);
    expect(afterIn.x).toBeCloseTo(documentUnderPointer.x, 10);
    expect(afterIn.y).toBeCloseTo(documentUnderPointer.y, 10);

    h.handler(makeWheelEvent(10, { ctrlKey: true }));
    expect(h.viewport.getZoom()).toBeCloseTo(1, 10);
    expect(h.invalidate).toHaveBeenCalledWith({ view: true });
  });

  it('bounds one ctrl+mouse-wheel notch to a 1.25x step either way', () => {
    const zoomIn = trackpadHarness();
    zoomIn.handler(makeWheelEvent(-100, { ctrlKey: true }));
    expect(zoomIn.viewport.getZoom()).toBeCloseTo(1.25, 10);

    const zoomOut = trackpadHarness();
    zoomOut.handler(makeWheelEvent(100, { ctrlKey: true }));
    expect(zoomOut.viewport.getZoom()).toBeCloseTo(0.8, 10);
  });

  it('does not step the brush size on ctrl+wheel', () => {
    const h = trackpadHarness(toolWithId('brush'));
    h.handler(makeWheelEvent(-100, { ctrlKey: true }));
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
  });
});

describe('wheel handler: trackpad pinch detection', () => {
  const pinch = (timeStamp: number, deltaY = -8, clientX = 10) =>
    makeWheelEvent(deltaY, { clientX, ctrlKey: true, timeStamp });
  const pinchHarness = (controlKey: ControlKeyState = 'up', enableOnPinch = true, armed = true) =>
    createHarness(toolWithId('brush'), false, false, { armed, controlKey, enableOnPinch });

  it('swallows the start of a pinch run, then zooms by the whole run at the pointer of the step that completes it', () => {
    const h = pinchHarness();
    h.handler(pinch(0, -8, 10));
    h.handler(pinch(16, -6, 20));
    // Candidates are neither a brush resize nor, yet, a zoom.
    expect(h.state.pinchReports).toBe(0);
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
    expect(h.viewport.getZoom()).toBe(1);

    const anchor = { x: 30, y: 10 };
    const documentUnderPointer = h.viewport.screenToDocument(anchor);
    h.handler(pinch(32, -10, anchor.x));
    expect(h.state.pinchReports).toBe(1);
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
    // 8 + 6 + 10 pixels of spreading at the 0.01 pinch gain: the first steps are not lost.
    expect(h.viewport.getZoom()).toBeCloseTo(Math.exp(0.24), 10);
    const after = h.viewport.screenToDocument(anchor);
    expect(after.x).toBeCloseTo(documentUnderPointer.x, 10);
    expect(after.y).toBeCloseTo(documentUnderPointer.y, 10);
  });

  it('bounds each step of the detected run as a live pinch step is bounded', () => {
    const h = pinchHarness();
    // 30px steps would each zoom by e^0.3; one ctrl+wheel step zooms by at most 1.25x.
    for (const t of [0, 16, 32]) {
      h.handler(pinch(t, -30));
    }
    expect(h.viewport.getZoom()).toBeCloseTo(1.25 ** 3, 10);
  });

  it('starts a fresh run, without the expired steps, once the time window has passed', () => {
    const h = pinchHarness();
    h.handler(pinch(0, -20));
    h.handler(pinch(400));
    h.handler(pinch(416));
    expect(h.state.pinchReports).toBe(0);
    h.handler(pinch(432));
    expect(h.state.pinchReports).toBe(1);
    expect(h.viewport.getZoom()).toBeCloseTo(Math.exp(0.24), 10);
  });

  it('detects a pinch delivered in lines, measuring its steps in pixels', () => {
    const h = pinchHarness();
    // Half a line is 8px, far below a mouse notch.
    for (const t of [0, 16, 32]) {
      h.handler(makeWheelEvent(-0.5, { ctrlKey: true, deltaMode: 1, timeStamp: t }));
    }
    expect(h.state.pinchReports).toBe(1);
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
    expect(h.viewport.getZoom()).toBeCloseTo(Math.exp(0.24), 10);
  });

  it('keeps swallowing candidates while a listener that leaves the setting off still listens', () => {
    const h = pinchHarness('up', false);
    for (const t of [0, 16, 32]) {
      h.handler(pinch(t));
    }
    expect(h.state.pinchReports).toBe(1);
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
    expect(h.viewport.getZoom()).toBe(1);
  });

  it('sizes the brush from pinches and reports nothing once nobody listens (the setting was decided)', () => {
    const h = pinchHarness('up', true, false);
    for (const t of [0, 16, 32]) {
      h.handler(pinch(t));
    }
    expect(h.state.pinchReports).toBe(0);
    expect(h.stepActiveBrushSize).toHaveBeenCalledTimes(3);
  });

  it('treats ctrl+wheel with the Control key down or unknown as a brush resize, never a pinch', () => {
    for (const controlKey of ['down', 'unknown'] as const) {
      const h = pinchHarness(controlKey);
      for (let i = 0; i < 10; i++) {
        h.handler(pinch(i * 16));
      }
      expect(h.state.pinchReports).toBe(0);
      expect(h.state.trackpadNavigation).toBe(false);
      expect(h.stepActiveBrushSize).toHaveBeenCalledTimes(10);
    }
  });

  it('treats mouse notches, in pixels or 3-line steps, as a brush resize, never a pinch', () => {
    const notch = pinchHarness();
    const lines = pinchHarness();
    for (let i = 0; i < 10; i++) {
      notch.handler(pinch(i * 16, -100));
      lines.handler(makeWheelEvent(-3, { ctrlKey: true, deltaMode: 1, timeStamp: i * 16 }));
    }
    expect(notch.state.pinchReports + lines.state.pinchReports).toBe(0);
    expect(notch.stepActiveBrushSize).toHaveBeenCalledTimes(10);
    expect(lines.stepActiveBrushSize).toHaveBeenCalledTimes(10);
  });

  it('needs the run within the time window, and any other wheel event breaks it', () => {
    const slow = pinchHarness();
    slow.handler(pinch(0));
    slow.handler(pinch(200));
    slow.handler(pinch(400));
    expect(slow.state.pinchReports).toBe(0);

    const broken = pinchHarness();
    broken.handler(pinch(0));
    broken.handler(pinch(16));
    broken.handler(makeWheelEvent(-8, { timeStamp: 24 }));
    broken.handler(pinch(32));
    expect(broken.state.pinchReports).toBe(0);
  });

  it('does not report pinches once trackpad navigation is on', () => {
    const h = createHarness(toolWithId('brush'), false, true, { armed: true, controlKey: 'up' });
    for (const t of [0, 16, 32]) {
      h.handler(pinch(t));
    }
    expect(h.state.pinchReports).toBe(0);
  });
});
