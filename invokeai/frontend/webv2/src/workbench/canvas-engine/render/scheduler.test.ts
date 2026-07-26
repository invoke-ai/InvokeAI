import type { LayerDamage, RenderFlags } from '@workbench/canvas-engine/types';

import { describe, expect, it, vi } from 'vitest';

import { createRenderScheduler } from './scheduler';

/** A controllable fake rAF: `flush()` runs the pending callback, if any. */
const createFakeRaf = () => {
  let nextHandle = 1;
  const callbacks = new Map<number, FrameRequestCallback>();
  return {
    cancelFrame: (handle: number): void => {
      callbacks.delete(handle);
    },
    /** Runs all currently-queued frame callbacks (like the browser firing a frame). */
    flush: (): void => {
      const queued = [...callbacks.entries()];
      callbacks.clear();
      for (const [, cb] of queued) {
        cb(performance.now());
      }
    },
    pendingCount: (): number => callbacks.size,
    requestFrame: (cb: FrameRequestCallback): number => {
      const handle = nextHandle++;
      callbacks.set(handle, cb);
      return handle;
    },
  };
};

describe('createRenderScheduler', () => {
  it('coalesces multiple invalidations into a single render with merged flags', () => {
    const raf = createFakeRaf();
    const render = vi.fn<(flags: RenderFlags) => void>();
    const scheduler = createRenderScheduler({ cancelFrame: raf.cancelFrame, render, requestFrame: raf.requestFrame });

    scheduler.invalidate({ view: true });
    scheduler.invalidate({ layers: ['a'] });
    scheduler.invalidate({ layers: ['b'] });
    scheduler.invalidate({ overlay: true });

    // Only one frame is scheduled despite four invalidations.
    expect(raf.pendingCount()).toBe(1);
    expect(render).not.toHaveBeenCalled();

    raf.flush();

    expect(render).toHaveBeenCalledTimes(1);
    const flags = render.mock.calls[0]![0];
    expect(flags.view).toBe(true);
    expect(flags.overlay).toBe(true);
    expect(flags.all).toBe(false);
    expect([...flags.layers].sort()).toEqual(['a', 'b']);
  });

  it('resets flags after the render callback runs', () => {
    const raf = createFakeRaf();
    const render = vi.fn<(flags: RenderFlags) => void>();
    const scheduler = createRenderScheduler({ cancelFrame: raf.cancelFrame, render, requestFrame: raf.requestFrame });

    scheduler.invalidate({ layers: ['a'], view: true });
    raf.flush();

    scheduler.invalidate({ layers: ['b'] });
    raf.flush();

    expect(render).toHaveBeenCalledTimes(2);
    const second = render.mock.calls[1]![0];
    // No leftover 'a' or view flag from the first frame.
    expect(second.view).toBe(false);
    expect([...second.layers]).toEqual(['b']);
  });

  it('does not schedule a new frame when nothing was invalidated', () => {
    const raf = createFakeRaf();
    const render = vi.fn<(flags: RenderFlags) => void>();
    createRenderScheduler({ cancelFrame: raf.cancelFrame, render, requestFrame: raf.requestFrame });

    expect(raf.pendingCount()).toBe(0);
    raf.flush();
    expect(render).not.toHaveBeenCalled();
  });

  it('accumulates invalidations while paused and flushes them on resume', () => {
    const raf = createFakeRaf();
    const render = vi.fn<(flags: RenderFlags) => void>();
    const scheduler = createRenderScheduler({ cancelFrame: raf.cancelFrame, render, requestFrame: raf.requestFrame });

    scheduler.pause();
    scheduler.invalidate({ layers: ['a'] });
    scheduler.invalidate({ view: true });

    // Nothing scheduled while paused.
    expect(raf.pendingCount()).toBe(0);

    scheduler.resume();
    expect(raf.pendingCount()).toBe(1);
    raf.flush();

    expect(render).toHaveBeenCalledTimes(1);
    const flags = render.mock.calls[0]![0];
    expect(flags.view).toBe(true);
    expect([...flags.layers]).toEqual(['a']);
  });

  it('pause cancels an already-scheduled frame; resume reschedules the same pending flags', () => {
    const raf = createFakeRaf();
    const render = vi.fn<(flags: RenderFlags) => void>();
    const scheduler = createRenderScheduler({ cancelFrame: raf.cancelFrame, render, requestFrame: raf.requestFrame });

    scheduler.invalidate({ view: true });
    expect(raf.pendingCount()).toBe(1);

    scheduler.pause();
    expect(raf.pendingCount()).toBe(0);
    raf.flush();
    expect(render).not.toHaveBeenCalled();

    scheduler.resume();
    raf.flush();
    expect(render).toHaveBeenCalledTimes(1);
    expect(render.mock.calls[0]![0].view).toBe(true);
  });

  it('resume with no pending work does not schedule a frame', () => {
    const raf = createFakeRaf();
    const render = vi.fn<(flags: RenderFlags) => void>();
    const scheduler = createRenderScheduler({ cancelFrame: raf.cancelFrame, render, requestFrame: raf.requestFrame });

    scheduler.pause();
    scheduler.resume();
    expect(raf.pendingCount()).toBe(0);
  });

  it('dispose cancels a pending frame and ignores further invalidations', () => {
    const raf = createFakeRaf();
    const cancelFrame = vi.fn(raf.cancelFrame);
    const render = vi.fn<(flags: RenderFlags) => void>();
    const scheduler = createRenderScheduler({ cancelFrame, render, requestFrame: raf.requestFrame });

    scheduler.invalidate({ view: true });
    expect(raf.pendingCount()).toBe(1);

    scheduler.dispose();
    expect(cancelFrame).toHaveBeenCalledTimes(1);
    expect(raf.pendingCount()).toBe(0);

    scheduler.invalidate({ all: true });
    raf.flush();
    expect(render).not.toHaveBeenCalled();
    expect(raf.pendingCount()).toBe(0);
  });

  it('invalidations from within the render callback schedule a fresh frame', () => {
    const raf = createFakeRaf();
    let reentered = false;
    const scheduler = createRenderScheduler({
      cancelFrame: raf.cancelFrame,
      render: () => {
        if (!reentered) {
          reentered = true;
          scheduler.invalidate({ overlay: true });
        }
      },
      requestFrame: raf.requestFrame,
    });

    scheduler.invalidate({ view: true });
    raf.flush();
    // The re-entrant invalidate scheduled another frame rather than being lost.
    expect(raf.pendingCount()).toBe(1);
  });

  it('retains pending invalidations and retries when the host frame scheduler throws', () => {
    const raf = createFakeRaf();
    const render = vi.fn<(flags: RenderFlags) => void>();
    let shouldThrow = true;
    const scheduler = createRenderScheduler({
      cancelFrame: raf.cancelFrame,
      render,
      requestFrame: (callback) => {
        if (shouldThrow) {
          throw new Error('host scheduler unavailable');
        }
        return raf.requestFrame(callback);
      },
    });

    expect(() => scheduler.invalidate({ view: true })).not.toThrow();
    expect(raf.pendingCount()).toBe(0);

    shouldThrow = false;
    scheduler.invalidate({ overlay: true });
    raf.flush();

    expect(render).toHaveBeenCalledOnce();
    expect(render.mock.calls[0]![0]).toMatchObject({ overlay: true, view: true });
  });

  it('reflects paused state via isPaused', () => {
    const raf = createFakeRaf();
    const scheduler = createRenderScheduler({
      cancelFrame: raf.cancelFrame,
      render: () => {},
      requestFrame: raf.requestFrame,
    });
    expect(scheduler.isPaused).toBe(false);
    scheduler.pause();
    expect(scheduler.isPaused).toBe(true);
    scheduler.resume();
    expect(scheduler.isPaused).toBe(false);
  });
});

describe('damage coalescing', () => {
  const setup = () => {
    const raf = createFakeRaf();
    const render = vi.fn();
    const scheduler = createRenderScheduler({
      cancelFrame: raf.cancelFrame,
      render,
      requestFrame: raf.requestFrame,
    });
    return { raf, render, scheduler };
  };

  const flagsOf = (render: ReturnType<typeof vi.fn>) => render.mock.calls[0]![0] as RenderFlags;

  const damage = (layerId: string, x: number): LayerDamage => ({
    layerId,
    rect: { height: 10, width: 10, x, y: 0 },
  });

  it('carries damage through when every invalidation names its region', () => {
    const { raf, render, scheduler } = setup();
    scheduler.invalidate({ damage: damage('a', 0), layers: ['a'] });
    scheduler.invalidate({ damage: damage('a', 40), layers: ['a'] });
    raf.flush();
    expect(flagsOf(render).damage).toEqual([damage('a', 0), damage('a', 40)]);
  });

  it('keeps overlay-only invalidations from widening the frame', () => {
    // The overlay is a separate canvas redrawn whole; it says nothing about
    // which composited pixels changed.
    const { raf, render, scheduler } = setup();
    scheduler.invalidate({ damage: damage('a', 0), layers: ['a'] });
    scheduler.invalidate({ overlay: true });
    raf.flush();
    expect(flagsOf(render).damage).toEqual([damage('a', 0)]);
  });

  it.each([
    ['a bare layer invalidation', { layers: ['a'] }],
    ['a viewport change', { view: true } as const],
    ['a forced full repaint', { all: true } as const],
  ])('widens to a full repaint after %s', (_label, payload) => {
    const { raf, render, scheduler } = setup();
    scheduler.invalidate({ damage: damage('a', 0), layers: ['a'] });
    scheduler.invalidate(payload);
    raf.flush();
    expect(flagsOf(render).damage).toBeNull();
  });

  it('stays widened once something in the frame could not name its damage', () => {
    const { raf, render, scheduler } = setup();
    scheduler.invalidate({ view: true });
    scheduler.invalidate({ damage: damage('a', 0), layers: ['a'] });
    raf.flush();
    expect(flagsOf(render).damage).toBeNull();
  });

  it('ignores damage that does not name the single invalidated layer', () => {
    // Damage describes one layer's region; anything else in the same payload
    // changed without a region, so the frame cannot be narrowed.
    const { raf, render, scheduler } = setup();
    scheduler.invalidate({ damage: damage('a', 0), layers: ['a', 'b'] });
    raf.flush();
    expect(flagsOf(render).damage).toBeNull();
  });

  it('ignores damage naming a different layer than the one invalidated', () => {
    const { raf, render, scheduler } = setup();
    scheduler.invalidate({ damage: damage('b', 0), layers: ['a'] });
    raf.flush();
    expect(flagsOf(render).damage).toBeNull();
  });

  it('starts each frame with a fresh damage set', () => {
    const { raf, render, scheduler } = setup();
    scheduler.invalidate({ all: true });
    raf.flush();
    expect(flagsOf(render).damage).toBeNull();

    render.mockClear();
    scheduler.invalidate({ damage: damage('a', 0), layers: ['a'] });
    raf.flush();
    expect(flagsOf(render).damage).toEqual([damage('a', 0)]);
  });
});
