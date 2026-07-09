import type { RenderFlags } from '@workbench/canvas-engine/types';

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
