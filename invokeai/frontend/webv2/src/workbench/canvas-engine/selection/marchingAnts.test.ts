import type { StubRasterSurface } from '@workbench/canvas-engine/render/raster.testStub';
import type { Mat2d } from '@workbench/canvas-engine/types';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createAntsAnimator, drawMarchingAnts } from '@workbench/canvas-engine/selection/marchingAnts';
import { describe, expect, it, vi } from 'vitest';

const IDENTITY: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
const fakePath = (id: string): Path2D => ({ id }) as unknown as Path2D;

describe('drawMarchingAnts', () => {
  it('strokes each path twice (two-tone), applies a dash and the animated offset', () => {
    const surface = createTestStubRasterBackend().createSurface(50, 50) as StubRasterSurface;
    drawMarchingAnts(surface.ctx, IDENTITY, { paths: [fakePath('p')], phase: 3 });

    const log = surface.callLog;
    const ops = log.map((e) => e.op);
    expect(ops.filter((op) => op === 'stroke')).toHaveLength(2);
    expect(ops).toContain('setLineDash');
    expect(ops).toContain('setTransform');

    const strokeStyles = log.filter((e) => e.op === 'set' && e.args[0] === 'strokeStyle').map((e) => e.args[1]);
    expect(strokeStyles).toContain('#000000');
    expect(strokeStyles).toContain('#ffffff');
    // The light run is offset from the dark run so the ants appear to crawl.
    const offsets = log.filter((e) => e.op === 'set' && e.args[0] === 'lineDashOffset').map((e) => e.args[1]);
    expect(new Set(offsets).size).toBeGreaterThan(1);
  });

  it('is a no-op with no paths', () => {
    const surface = createTestStubRasterBackend().createSurface(10, 10) as StubRasterSurface;
    drawMarchingAnts(surface.ctx, IDENTITY, { paths: [], phase: 0 });
    expect(surface.callLog).toHaveLength(0);
  });
});

/** A controllable rAF pair + clock for driving the animator deterministically. */
const createDriver = () => {
  let nextHandle = 1;
  const callbacks = new Map<number, () => void>();
  let clock = 0;
  return {
    advance: (ms: number): void => {
      clock += ms;
    },
    cancelFrame: (handle: number): void => {
      callbacks.delete(handle);
    },
    /** Runs every queued frame callback once (like the browser firing a frame). */
    flush: (): void => {
      const queued = [...callbacks.entries()];
      callbacks.clear();
      for (const [, cb] of queued) {
        cb();
      }
    },
    now: (): number => clock,
    pending: (): number => callbacks.size,
    requestFrame: (cb: () => void): number => {
      const handle = nextHandle++;
      callbacks.set(handle, cb);
      return handle;
    },
  };
};

describe('createAntsAnimator', () => {
  it('steps on the throttle boundary and reschedules each frame while running', () => {
    const driver = createDriver();
    const onStep = vi.fn();
    const animator = createAntsAnimator({
      cancelFrame: driver.cancelFrame,
      intervalMs: 100,
      now: driver.now,
      onStep,
      requestFrame: driver.requestFrame,
    });

    animator.start();
    expect(animator.isRunning).toBe(true);
    expect(driver.pending()).toBe(1);

    // First frame steps immediately (start primes lastStep one interval back).
    driver.flush();
    expect(onStep).toHaveBeenCalledTimes(1);
    // It rescheduled itself.
    expect(driver.pending()).toBe(1);

    // A frame before the interval elapses does not step, but keeps polling.
    driver.advance(40);
    driver.flush();
    expect(onStep).toHaveBeenCalledTimes(1);
    expect(driver.pending()).toBe(1);

    // Crossing the interval steps again.
    driver.advance(70);
    driver.flush();
    expect(onStep).toHaveBeenCalledTimes(2);
  });

  it('stop cancels the pending frame and halts rescheduling (no timer leak)', () => {
    const driver = createDriver();
    const onStep = vi.fn();
    const animator = createAntsAnimator({
      cancelFrame: driver.cancelFrame,
      intervalMs: 100,
      now: driver.now,
      onStep,
      requestFrame: driver.requestFrame,
    });

    animator.start();
    animator.stop();
    expect(animator.isRunning).toBe(false);
    expect(driver.pending()).toBe(0);

    // Any straggler frame that still fires must not step or reschedule.
    driver.advance(1000);
    driver.flush();
    expect(onStep).not.toHaveBeenCalled();
    expect(driver.pending()).toBe(0);
  });

  it('start is idempotent (no double scheduling)', () => {
    const driver = createDriver();
    const animator = createAntsAnimator({
      cancelFrame: driver.cancelFrame,
      now: driver.now,
      onStep: vi.fn(),
      requestFrame: driver.requestFrame,
    });
    animator.start();
    animator.start();
    expect(driver.pending()).toBe(1);
  });
});
