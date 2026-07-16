import { describe, expect, it, vi } from 'vitest';

import { RenderController } from './renderController';

describe('RenderController', () => {
  it('owns scheduling lifecycle and disposes idempotently', () => {
    let frame: FrameRequestCallback | null = null;
    const render = vi.fn();
    const controller = new RenderController({
      applyCursor: vi.fn(),
      cancelFrame: () => {
        frame = null;
      },
      clearPreview: vi.fn(),
      getInputHandlers: () =>
        ({
          onKeyDown: vi.fn(),
          onKeyUp: vi.fn(),
          onPointerCancel: vi.fn(),
          onPointerDown: vi.fn(),
          onPointerEnter: vi.fn(),
          onPointerLeave: vi.fn(),
          onPointerMove: vi.fn(),
          onPointerUp: vi.fn(),
          onWheel: vi.fn(),
          reset: vi.fn(),
        }) as never,
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

    controller.scheduler.invalidate({ overlay: true });
    expect(frame).not.toBeNull();
    const scheduled = frame as unknown as FrameRequestCallback;
    frame = null;
    scheduled(0);
    expect(render).toHaveBeenCalledOnce();
    controller.dispose();
    controller.dispose();
    controller.scheduler.invalidate({ all: true });
    expect(frame).toBeNull();
  });
});
