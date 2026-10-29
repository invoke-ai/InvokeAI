import type { WheelHandlerDeps } from '@workbench/canvas-engine/input/wheel';
import type { Tool, ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { ToolId } from '@workbench/canvas-engine/types';

import { createWheelHandler } from '@workbench/canvas-engine/input/wheel';
import { createViewport } from '@workbench/canvas-engine/viewport';
import { describe, expect, it, vi } from 'vitest';

const makeWheelEvent = (deltaY: number, opts: { ctrlKey?: boolean } = {}): WheelEvent =>
  ({
    altKey: false,
    clientX: 10,
    clientY: 10,
    ctrlKey: opts.ctrlKey ?? false,
    deltaY,
    metaKey: false,
    preventDefault: vi.fn(),
    shiftKey: false,
  }) as unknown as WheelEvent;

const createHarness = (tool: Tool | undefined, invertBrushSizeScroll = false) => {
  const viewport = createViewport();
  const stepActiveBrushSize = vi.fn();
  const invalidate = vi.fn();
  const deps: WheelHandlerDeps = {
    getActiveTool: () => tool,
    getInputElement: () => ({ getBoundingClientRect: () => ({ left: 0, top: 0 }) }) as unknown as HTMLElement,
    getInvertBrushSizeScroll: () => invertBrushSizeScroll,
    getToolContext: () => ({}) as ToolContext,
    invalidate,
    stepActiveBrushSize,
    viewport,
  };
  return { handler: createWheelHandler(deps), invalidate, stepActiveBrushSize, viewport };
};

const toolWithId = (id: ToolId, onWheel?: Tool['onWheel']): Tool => ({ id, onWheel });

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
  it('zooms the viewport when the active tool defines no onWheel', () => {
    const h = createHarness(toolWithId('brush'));
    const wheelZoom = vi.spyOn(h.viewport, 'wheelZoom');
    h.handler(makeWheelEvent(-120));
    expect(wheelZoom).toHaveBeenCalledWith(-120, { x: 10, y: 10 });
    expect(h.invalidate).toHaveBeenCalledWith({ view: true });
  });

  it('routes to the active tool onWheel when present', () => {
    const onWheel = vi.fn();
    const h = createHarness(toolWithId('view', onWheel));
    const wheelZoom = vi.spyOn(h.viewport, 'wheelZoom');
    h.handler(makeWheelEvent(-120));
    expect(onWheel).toHaveBeenCalledTimes(1);
    expect(wheelZoom).not.toHaveBeenCalled();
  });
});
