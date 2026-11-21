import type { ScrollInput } from '@workbench/canvas-engine/engineStores';
import type { ControlKeyState } from '@workbench/canvas-engine/input/pointerPipeline';
import type { Tool } from '@workbench/canvas-engine/tools/tool';
import type { ToolId } from '@workbench/canvas-engine/types';

import { createWheelInput } from '@workbench/canvas-engine/input/wheel';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { describe, expect, it, vi } from 'vitest';

interface WheelInit {
  ctrlKey?: boolean;
  deltaMode?: number;
  deltaX?: number;
  wheelDeltaY?: number;
}

const ANCHOR = { x: 10, y: 10 };

const createHarness = (
  opts: {
    controlKey?: ControlKeyState;
    gestureActive?: boolean;
    invertBrushSizeScroll?: boolean;
    scrollInput?: ScrollInput;
    tool?: ToolId;
    trackpadDetected?: boolean;
  } = {}
) => {
  const viewport = createViewport();
  const stepActiveBrushSize = vi.fn();
  const invalidate = vi.fn();
  const detected = { value: opts.trackpadDetected ?? false };
  const tool: Tool = { id: opts.tool ?? 'brush' };
  const input = createWheelInput({
    getActiveTool: () => tool,
    getControlKeyState: () => opts.controlKey ?? 'down',
    getInputElement: () => ({ getBoundingClientRect: () => ({ left: 0, top: 0 }) }) as unknown as HTMLElement,
    getInvertBrushSizeScroll: () => opts.invertBrushSizeScroll ?? false,
    getScrollInput: () => opts.scrollInput ?? 'auto',
    invalidate,
    isGestureActive: () => opts.gestureActive ?? false,
    stepActiveBrushSize,
    trackpadDetected: {
      get: () => detected.value,
      set: (value) => {
        detected.value = value;
      },
    },
    viewport,
  });
  // Each event starts its own stream unless a test says otherwise.
  let timeStamp = 0;
  const wheel = (deltaY: number, init: WheelInit = {}): void => {
    timeStamp += 1000;
    input.onWheel({
      clientX: ANCHOR.x,
      clientY: ANCHOR.y,
      ctrlKey: init.ctrlKey ?? false,
      deltaMode: init.deltaMode ?? 0,
      deltaX: init.deltaX ?? 0,
      deltaY,
      preventDefault: vi.fn(),
      timeStamp,
      wheelDeltaY: init.wheelDeltaY ?? 0,
    } as unknown as WheelEvent);
  };
  const notch = (direction: 1 | -1, init: WheelInit = {}) =>
    wheel(100 * direction, { wheelDeltaY: -120 * direction, ...init });
  const smooth = (deltaY: number, init: WheelInit = {}) => wheel(deltaY, { wheelDeltaY: -3 * deltaY, ...init });
  const wheelZoom = vi.spyOn(viewport, 'wheelZoom');
  return { detected, input, invalidate, notch, smooth, stepActiveBrushSize, viewport, wheel, wheelZoom };
};

describe('wheel input: mouse wheel', () => {
  it('zooms around the pointer, whatever the event looks like', () => {
    const h = createHarness({ scrollInput: 'mouse' });
    h.notch(-1);
    h.smooth(-6, { deltaX: 2 });
    expect(h.wheelZoom.mock.calls).toEqual([
      [-100, ANCHOR],
      [-6, ANCHOR],
    ]);
    expect(h.invalidate).toHaveBeenCalledWith({ view: true });
    expect(h.detected.value).toBe(false);
  });

  it('steps the brush or eraser size on ctrl+wheel, inverted by the preference', () => {
    const brush = createHarness({ scrollInput: 'mouse' });
    brush.notch(-1, { ctrlKey: true });
    brush.notch(1, { ctrlKey: true });
    expect(brush.stepActiveBrushSize.mock.calls).toEqual([[1], [-1]]);

    const inverted = createHarness({ invertBrushSizeScroll: true, scrollInput: 'mouse', tool: 'eraser' });
    inverted.notch(-1, { ctrlKey: true });
    expect(inverted.stepActiveBrushSize).toHaveBeenCalledWith(-1);
  });

  it('swallows ctrl+wheel for other tools', () => {
    const h = createHarness({ scrollInput: 'mouse', tool: 'view' });
    h.notch(-1, { ctrlKey: true });
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
    expect(h.wheelZoom).not.toHaveBeenCalled();
    expect(h.viewport.getState()).toEqual({ pan: { x: 0, y: 0 }, zoom: 1 });
  });
});

describe('wheel input: trackpad', () => {
  it('pans on both axes so content follows the fingers, a notch included', () => {
    const h = createHarness({ scrollInput: 'trackpad', tool: 'view' });
    h.smooth(30, { deltaX: -12 });
    h.notch(1);
    expect(h.viewport.getState()).toEqual({ pan: { x: 12, y: -130 }, zoom: 1 });
    expect(h.invalidate).toHaveBeenCalledWith({ view: true });
  });

  it('zooms continuously around the pointer on ctrl+wheel, at most 1.25x per event, without sizing the brush', () => {
    const h = createHarness({ scrollInput: 'trackpad' });
    const documentUnderPointer = h.viewport.screenToDocument(ANCHOR);
    h.smooth(-10, { ctrlKey: true });
    expect(h.viewport.getZoom()).toBeCloseTo(Math.exp(0.1), 10);
    const after = h.viewport.screenToDocument(ANCHOR);
    expect(after.x).toBeCloseTo(documentUnderPointer.x, 10);
    expect(after.y).toBeCloseTo(documentUnderPointer.y, 10);

    h.notch(1, { ctrlKey: true });
    expect(h.viewport.getZoom()).toBeCloseTo(Math.exp(0.1) / 1.25, 10);
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
  });
});

describe('wheel input: auto', () => {
  it('zooms on a notch and sizes the brush on ctrl+notch, even after a trackpad was detected', () => {
    const h = createHarness({ trackpadDetected: true });
    h.notch(-1);
    h.notch(-1, { ctrlKey: true });
    expect(h.wheelZoom.mock.calls).toEqual([[-100, ANCHOR]]);
    expect(h.stepActiveBrushSize).toHaveBeenCalledWith(1);
  });

  it('zooms on smooth scrolling until a trackpad is detected, then pans', () => {
    const h = createHarness();
    h.smooth(-6);
    expect(h.wheelZoom).toHaveBeenCalledWith(-6, ANCHOR);

    h.detected.value = true;
    h.smooth(-6);
    expect(h.wheelZoom).toHaveBeenCalledOnce();
    expect(h.viewport.getState().pan).toEqual({ x: 0, y: 6 });
  });

  it('detects a trackpad from a two-axis event and pans by that event', () => {
    const h = createHarness();
    h.smooth(6, { deltaX: 2 });
    expect(h.detected.value).toBe(true);
    expect(h.wheelZoom).not.toHaveBeenCalled();
    expect(h.viewport.getState().pan).toEqual({ x: -2, y: -6 });
  });

  it('detects a pinch, ctrl+wheel with Control known up, and zooms by it instead of sizing the brush', () => {
    const h = createHarness({ controlKey: 'up' });
    // Chromium stamps each pinch step as one whole wheel tick.
    h.wheel(-10, { ctrlKey: true, wheelDeltaY: 120 });
    expect(h.detected.value).toBe(true);
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
    expect(h.viewport.getZoom()).toBeCloseTo(Math.exp(0.1), 10);
  });

  it('zooms by a pinch whose first step is large, which Chromium also stamps as a whole tick', () => {
    const h = createHarness({ controlKey: 'up' });
    h.wheel(-30, { ctrlKey: true, wheelDeltaY: 120 });
    expect(h.stepActiveBrushSize).not.toHaveBeenCalled();
    expect(h.viewport.getZoom()).toBeCloseTo(1.25, 10);
  });

  it('sizes the brush on a ctrl+notch that arrives with Control up, as Sticky Keys sends it', () => {
    const h = createHarness({ controlKey: 'up' });
    h.notch(-1, { ctrlKey: true });
    expect(h.stepActiveBrushSize).toHaveBeenCalledWith(1);
    expect(h.detected.value).toBe(false);
  });

  it('sizes the brush on smooth ctrl+wheel while Control may be held', () => {
    for (const controlKey of ['down', 'unknown'] as const) {
      const h = createHarness({ controlKey });
      h.smooth(-4, { ctrlKey: true });
      expect(h.detected.value).toBe(false);
      expect(h.stepActiveBrushSize).toHaveBeenCalledWith(1);
    }
  });

  it('pans on a sideways-only wheel before detection, without detecting, in pixels or lines', () => {
    const h = createHarness();
    h.wheel(0, { deltaX: 8 });
    h.wheel(0, { deltaMode: 1, deltaX: 1 });
    expect(h.viewport.getState().pan).toEqual({ x: -24, y: 0 });
    expect(h.wheelZoom).not.toHaveBeenCalled();
    expect(h.detected.value).toBe(false);
  });

  it('measures line-mode deltas at 16px per line', () => {
    const h = createHarness({ trackpadDetected: true });
    h.wheel(3, { deltaMode: 1 });
    expect(h.wheelZoom).toHaveBeenCalledWith(48, ANCHOR);

    const trackpad = createHarness({ scrollInput: 'trackpad' });
    trackpad.wheel(3, { deltaMode: 1, deltaX: 1 });
    expect(trackpad.viewport.getState().pan).toEqual({ x: -16, y: -48 });
  });
});

const pinchGesture = (h: ReturnType<typeof createHarness>, type: 'gesturestart' | 'gesturechange', scale: number) => {
  const event = { clientX: ANCHOR.x, clientY: ANCHOR.y, preventDefault: vi.fn(), scale, type };
  (type === 'gesturestart' ? h.input.onPinchGestureStart : h.input.onPinchGestureChange)(event as unknown as Event);
  return event.preventDefault;
};

describe('wheel input: Safari pinch gesture', () => {
  it('leaves the view alone while a pointer gesture is under way, as an iPad touch pinch mid-stroke', () => {
    const h = createHarness({ gestureActive: true });
    expect(pinchGesture(h, 'gesturestart', 1)).toHaveBeenCalled();
    expect(pinchGesture(h, 'gesturechange', 2)).toHaveBeenCalled();
    expect(h.viewport.getZoom()).toBe(1);
    expect(h.detected.value).toBe(false);
  });

  it('zooms by the gesture scale from its start, around the pointer, and detects a trackpad in any mode', () => {
    const h = createHarness({ scrollInput: 'mouse' });
    h.viewport.zoomAtPoint(2, ANCHOR);
    const documentUnderPointer = h.viewport.screenToDocument(ANCHOR);
    expect(pinchGesture(h, 'gesturestart', 1)).toHaveBeenCalled();
    pinchGesture(h, 'gesturechange', 1.5);
    expect(pinchGesture(h, 'gesturechange', 0.5)).toHaveBeenCalled();
    expect(h.viewport.getZoom()).toBeCloseTo(1, 10);
    const after = h.viewport.screenToDocument(ANCHOR);
    expect(after.x).toBeCloseTo(documentUnderPointer.x, 10);
    expect(after.y).toBeCloseTo(documentUnderPointer.y, 10);
    expect(h.detected.value).toBe(true);
  });
});
