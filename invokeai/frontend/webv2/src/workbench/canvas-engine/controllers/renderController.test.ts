import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { RasterMemoryBudgetController } from './rasterMemoryBudgetController';
import { RenderController } from './renderController';

const createHarness = () => {
  let frame: FrameRequestCallback | null = null;
  const render = vi.fn();
  const backend = createTestStubRasterBackend();
  const rasterMemory = new RasterMemoryBudgetController();
  const controller = new RenderController({
    applyCursor: vi.fn(),
    backend,
    rasterMemory,
    cancelFrame: () => {
      frame = null;
    },
    getInputHandlers: () => ({
      onKeyDown: vi.fn(),
      onKeyUp: vi.fn(),
      onFocusIn: vi.fn(),
      onPointerCancel: vi.fn(),
      onPointerDown: vi.fn(),
      onPointerEnter: vi.fn(),
      onPointerLeave: vi.fn(),
      onPointerMove: vi.fn(),
      onPointerUp: vi.fn(),
      onWheel: vi.fn(),
      reset: vi.fn(),
    }),
    isEngineDisposed: () => false,
    onPageHide: vi.fn(),
    onVisibilityChange: vi.fn(),
    onWindowBlur: vi.fn(),
    render,
    requestFrame: (callback) => {
      frame = callback;
      return 1;
    },
    setViewportReady: vi.fn(),
    updateAnimation: vi.fn(),
    updateCursor: vi.fn(),
  });

  return {
    controller,
    currentFrame: () => frame,
    rasterMemory,
    render,
    runFrame: () => {
      const scheduled = frame;
      frame = null;
      scheduled?.(0);
    },
  };
};

describe('RenderController', () => {
  it('owns scheduling lifecycle and disposes idempotently', () => {
    const { controller, currentFrame, render, runFrame } = createHarness();
    controller.scheduler.invalidate({ overlay: true });
    expect(currentFrame()).not.toBeNull();
    runFrame();
    expect(render).toHaveBeenCalledOnce();
    controller.dispose();
    controller.dispose();
    controller.scheduler.invalidate({ all: true });
    expect(currentFrame()).toBeNull();
  });

  it('reuses the viewport mask scratch, accounts its backing bytes and releases them on detach', () => {
    const { controller, rasterMemory } = createHarness();
    const scratch = controller.getMaskScratch(20, 10);
    const bytes = () => rasterMemory.snapshot().detachedBytes;
    expect(bytes()).toBe(800);
    expect(controller.getMaskScratch(30, 5)).toBe(scratch);
    expect(bytes()).toBe(1200);

    controller.resize(15, 8);
    expect(scratch).toMatchObject({ height: 8, width: 15 });
    expect(bytes()).toBe(480);

    controller.detach();
    expect(scratch).toMatchObject({ height: 0, width: 0 });
    expect(bytes()).toBe(0);
    expect(controller.getMaskScratch(20, 10)).not.toBe(scratch);
    controller.dispose();
    expect(bytes()).toBe(0);
  });
});
